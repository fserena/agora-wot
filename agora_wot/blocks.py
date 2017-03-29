"""
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=#
  Ontology Engineering Group
        http://www.oeg-upm.net/
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=#
  Copyright (C) 2016 Ontology Engineering Group.
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=#
  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

            http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=#
"""
import itertools
from abc import abstractmethod
from urlparse import urljoin

import networkx as nx
import requests
from rdflib import ConjunctiveGraph
from rdflib import Graph
from rdflib import RDF
from rdflib.term import Node, BNode, URIRef
from shortuuid import uuid

from agora_wot.evaluate import find_params, evaluate
from agora_wot.ns import CORE, WOT, MAP
from agora_wot.utils import encode_rdict

__author__ = 'Fernando Serena'


def describe(graph, elm, filters=[], trace=None):
    def desc(obj=False):
        tp = (None, None, elm) if obj else (elm, None, None)
        for (s, p, o) in graph.triples(tp):
            triple = (s, p, o)
            ext_node = s if obj else o
            if triple not in trace:
                trace.add(triple)
                yield triple
            if ext_node not in trace:
                if isinstance(ext_node, BNode):
                    ignore = any(list(graph.triples((ext_node, x, None))) for x in filters)
                    if not ignore:
                        trace.add(ext_node)
                        for t in describe(graph, ext_node, filters=filters, trace=trace):
                            yield t

    if trace is None:
        trace = set([])
    for t in itertools.chain(desc()):
        yield t


class TD(object):
    def __init__(self, resource, id=None):
        # type: (Resource) -> None
        self.__resource = resource
        self.__id = id if id is not None else uuid()
        self.__access_mappings = set([])
        self.__rdf_sources = set([])
        self.__vars = set([])
        self.__endpoints = set([])

    @staticmethod
    def from_graph(graph, node, node_map):
        if node in node_map:
            return node_map[node]

        try:
            r_node = list(graph.objects(subject=node, predicate=WOT.describes)).pop()
        except IndexError:
            raise ValueError

        resource = Resource.from_graph(graph, r_node, node_map=node_map)
        td = TD(resource)

        try:
            td.__id = list(graph.objects(node, WOT.identifier)).pop().toPython()
        except IndexError:
            pass

        for mr_node in graph.objects(node, MAP.hasAccessMapping):
            mr = AccessMapping.from_graph(graph, mr_node, node_map=node_map)
            td.add_access_mapping(mr)

        for rs_node in graph.objects(node, MAP.fromRDFSource):
            rdf_source = RDFSource.from_graph(graph, rs_node, node_map=node_map)
            td.add_rdf_source(rdf_source)

        node_map[node] = td
        return td

    @staticmethod
    def from_types(types=[], id=None, uri=None):
        if types:
            r = Resource(uri=None, types=types)
            return TD(r, id)

    def add_access_mapping(self, am):
        self.__access_mappings.add(am)
        self.__vars = reduce(lambda x, y: set.union(x, y), [mr.vars for mr in self.__access_mappings], set([]))
        self.__endpoints = set([mr.endpoint for mr in self.__access_mappings])

    def add_rdf_source(self, source):
        self.__rdf_sources.add(source)

    def endpoint_mappings(self, e):
        return reduce(lambda x, y: set.union(x, y),
                      map(lambda x: x.mappings, filter(lambda x: x.endpoint == e, self.__access_mappings)), set([]))

    def clone(self, id=None, **kwargs):
        new = TD(self.__resource, id=id)
        for am in self.access_mappings:
            am_end = am.endpoint
            href = am_end.evaluate_href(**{'$' + k: kwargs[k] for k in kwargs})
            e = Endpoint(href=href, media=am_end.media, whref=am_end.whref,
                         intercept=am_end.intercept)
            clone_am = AccessMapping(e)
            for m in am.mappings:
                clone_am.mappings.add(m)
            new.add_access_mapping(clone_am)
        for s in self.rdf_sources:
            new.add_rdf_source(s)
        return new

    @property
    def id(self):
        # type: (None) -> str
        return self.__id

    @property
    def access_mappings(self):
        # type: (None) -> iter
        return frozenset(self.__access_mappings)

    @property
    def base(self):
        # type: (None) -> iter
        return frozenset(self.__endpoints)

    @property
    def rdf_sources(self):
        # type: (None) -> iter
        return frozenset(self.__rdf_sources)

    @property
    def resource(self):
        # type: (None) -> Resource
        return self.__resource

    @property
    def vars(self):
        # type: (None) -> iter
        return frozenset(self.__vars)


class Resource(object):
    def __init__(self, uri=None, types=None):
        # type: (any, iter) -> None
        self.__graph = ConjunctiveGraph()
        self.__node = uri
        if self.__node is None:
            self.__node = BNode()
        if types is None:
            types = []
        self.__types = set(types)
        for t in self.__types:
            self.__graph.add((self.__node, RDF.type, URIRef(t)))

    @staticmethod
    def from_graph(graph, node, node_map):
        if node in node_map:
            return node_map[node]

        r = Resource()
        for t in describe(graph, node):
            r.__graph.add(t)

        r.__node = node
        r.__types = set(graph.objects(node, RDF.type))

        node_map[node] = r
        return r

    @property
    def types(self):
        return set(self.__graph.objects(subject=self.__node, predicate=RDF.type))

    @property
    def graph(self):
        return self.__graph

    @property
    def node(self):
        return self.__node


class AccessMapping(object):
    def __init__(self, endpoint):
        # type: (Endpoint) -> None

        self.mappings = set([])
        self.__vars = set([])

        self.__endpoint = endpoint
        self.__find_vars()

    def __find_vars(self):
        if self.__endpoint:
            ref = self.__endpoint.href
            for param in find_params(str(ref)):
                self.__vars.add(param)

    @staticmethod
    def from_graph(graph, node, node_map):
        # type: (Graph, Node) -> iter
        if node in node_map:
            return node_map[node]

        e_node = list(graph.objects(node, MAP.mapsResourcesFrom)).pop()
        endpoint = Endpoint.from_graph(graph, e_node, node_map=node_map)

        am = AccessMapping(endpoint)

        try:
            for m in graph.objects(node, MAP.hasMapping):
                am.mappings.add(Mapping.from_graph(graph, m, node_map=node_map))
        except IndexError:
            pass

        node_map[node] = am
        return am

    @property
    def vars(self):
        return frozenset(self.__vars)

    @property
    def endpoint(self):
        return self.__endpoint


class RDFSource(object):
    def __init__(self, endpoint):
        # type: (Endpoint) -> None
        self.__vars = set([])
        self.endpoint = endpoint
        self.__find_vars()

    def __find_vars(self):
        if self.endpoint:
            ref = self.endpoint.href
            for param in find_params(str(ref)):
                self.__vars.add(param)

    @staticmethod
    def from_graph(graph, node, node_map):
        # type: (Graph, Node) -> iter
        endpoint = Endpoint.from_graph(graph, node, node_map=node_map)
        rsource = RDFSource(endpoint)
        return rsource

    @staticmethod
    def from_uri(uri):
        e = Endpoint(href=uri, media="text/turtle")
        rdf_source = RDFSource(e)
        return rdf_source

    @property
    def vars(self):
        return frozenset(self.__vars)


class Endpoint(object):
    def __init__(self, href=None, media=None, whref=None, order=0, intercept=None, response_headers=None):
        self.href = href
        self.whref = whref
        self.order = order
        # self.mappings = set([])
        self.media = media or 'application/json'
        self.intercept = intercept
        self.response_headers = response_headers

    @staticmethod
    def from_graph(graph, node, node_map):
        # type: (Graph, Node) -> iter
        if node in node_map:
            return node_map[node]

        endpoint = Endpoint()
        try:
            endpoint.media = list(graph.objects(node, WOT.mediaType)).pop()
        except IndexError:
            pass

        try:
            endpoint.order = list(graph.objects(node, MAP.order)).pop()
        except IndexError:
            pass

        try:
            endpoint.href = list(graph.objects(node, WOT.href)).pop()
        except IndexError:
            whref = list(graph.objects(node, WOT.withHRef)).pop()
            endpoint.whref = whref

        node_map[node] = endpoint
        return endpoint

    def __add__(self, other):
        endpoint = Endpoint()
        if isinstance(other, Endpoint):
            # endpoint.mappings.update(other.mappings)
            endpoint.media = other.media
            other = other.whref if other.href is None else other.href

        endpoint.href = urljoin(self.href + '/', other, allow_fragments=True)
        return endpoint

    def evaluate_href(self, graph=None, subject=None, **kwargs):
        href = self.href
        for v in filter(lambda x: x in href, kwargs):
            href = href.replace(v, kwargs[v])
        return evaluate(href, graph=graph, subject=subject)

    def invoke(self, graph=None, subject=None, **kwargs):
        href = self.evaluate_href(graph=graph, subject=subject, **kwargs)
        print u'getting {}'.format(href)
        if self.intercept:
            r =self.intercept(href)
        else:
            r = requests.get(href, headers={'Accept': self.media})
        if self.response_headers is not None:
            r.headers.update(self.response_headers)
        return r


class Mapping(object):
    def __init__(self, key=None, predicate=None, transform=None, path=None, limit=None):
        self.key = key
        if predicate is not None:
            predicate = URIRef(predicate)
        self.predicate = predicate
        self.transform = transform
        self.path = path
        self.limit = limit

    @staticmethod
    def from_graph(graph, node, node_map):
        if node in node_map:
            return node_map[node]

        mapping = Mapping()

        try:
            mapping.predicate = list(graph.objects(node, MAP.predicate)).pop()
            mapping.key = list(graph.objects(node, MAP.key)).pop().toPython()
        except IndexError:
            pass

        try:
            mapping.path = list(graph.objects(node, MAP.jsonPath)).pop()
        except IndexError:
            pass

        try:
            mapping.transform = create_transform(graph, list(graph.objects(node, MAP.valuesTransformedBy)).pop(),
                                                 node_map)
        except IndexError:
            pass

        node_map[node] = mapping
        return mapping


def create_transform(graph, node, node_map):
    if list(graph.triples((node, RDF.type, WOT.ThingDescription))):
        return ResourceTransform.from_graph(graph, node, node_map=node_map)


class Transform(object):
    def attach(self, data):
        def wrapper(*args, **kwargs):
            return self.apply(data, *args, **kwargs)

        return wrapper

    @abstractmethod
    def apply(self, data, *args, **kwargs):
        pass


class ResourceTransform(Transform):
    def __init__(self, td):
        # type: (TD) -> None
        self.td = td

    @staticmethod
    def from_graph(graph, node, node_map):
        td = TD.from_graph(graph, node, node_map)
        transform = ResourceTransform(td)
        return transform

    def apply(self, data, *args, **kwargs):
        def merge(x, y):
            z = y.copy()
            z.update(x)
            return z

        if not isinstance(data, dict):
            uri_provider = kwargs['uri_provider']
            if not isinstance(data, list):
                data = [data]
            vars = kwargs['vars']
            parent_item = kwargs.get('$item', None) if '$parent' in vars else None
            base_rdict = {"$parent": parent_item} if parent_item is not None else {}
            res = [uri_provider(self.td.resource.node, encode_rdict(merge({"$item": v}, base_rdict))) for v in data]
            return res  # [:min(3, len(res))]
        return data


class TED(object):
    def __init__(self):
        self.__ecosystem = Ecosystem()

    @staticmethod
    def from_graph(graph):
        ted = TED()
        ted.__ecosystem = Ecosystem.from_graph(graph)
        return ted

    @property
    def ecosystem(self):
        return self.__ecosystem


class Ecosystem(object):
    def __init__(self):
        # type: (None) -> None

        self.__resources = set([])
        self.__tds = set([])
        self.__roots = set([])
        self.__root_tds = set([])

    @staticmethod
    def from_graph(graph):
        eco = Ecosystem()

        try:
            node = list(graph.subjects(RDF.type, CORE.Ecosystem)).pop()
        except IndexError:
            raise ValueError('Ecosystem node not found')

        node_block_map = {}
        root_nodes = set([])
        for r_node in graph.objects(node, CORE.hasComponent):
            try:
                td_node = list(graph.subjects(predicate=WOT.describes, object=r_node)).pop()
                td = TD.from_graph(graph, td_node, node_map=node_block_map)
                eco.add_component_from_td(td)
            except IndexError:
                resource = Resource.from_graph(graph, r_node, node_map=node_block_map)
                eco.add_component(resource)

            root_nodes.add(r_node)

        for td_node, r_node in graph.subject_objects(predicate=WOT.describes):
            if r_node not in root_nodes:
                td = TD.from_graph(graph, td_node, node_map=node_block_map)
                eco.__tds.add(td)
                eco.__resources.add(td.resource)

        return eco

    @property
    def resources(self):
        return frozenset(self.__resources)

    @property
    def tds(self):
        return self.__tds

    def add_component_from_td(self, td):
        # type: (TD) -> None
        self.__tds.add(td)
        self.__root_tds.add(td)
        self.__roots.add(td)
        self.__resources.add(td.resource)
        self.network()

    def add_component(self, resource):
        # type: (Resource) -> None
        self.__roots.add(resource)
        self.__resources.add(resource)

    @property
    def roots(self):
        return frozenset(self.__roots)

    @property
    def root_types(self):
        types = set([])
        for root in self.__roots:
            resource = root.resource if isinstance(root, TD) else root
            types.update(resource.types)
        return frozenset(types)

    def network(self):
        network = nx.DiGraph()
        children = set([])
        for td in self.__tds:
            transforming_mapping_sets = map(
                lambda am: filter(lambda m: isinstance(m.transform, ResourceTransform), am.mappings),
                td.access_mappings)
            transforming_mappings = reduce(lambda x, y: x.union(y), transforming_mapping_sets, set([]))
            for m in transforming_mappings:
                child_td = m.transform.td
                children.add(child_td)
                network.add_edge(td.id, child_td.id)

        for ch in children:
            self.__tds.add(ch)

        # print network.edges()
        return network

    def tds_by_type(self, t):
        for td in self.__root_tds:
            if t in td.resource.types:
                yield td
