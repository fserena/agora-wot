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
import logging
from abc import abstractmethod
from urllib import urlencode
from urlparse import urljoin, urlparse, parse_qs, urlunparse

import networkx as nx
import requests
from StringIO import StringIO
from rdflib import ConjunctiveGraph
from rdflib import Graph
from rdflib import RDF
from rdflib.term import Node, BNode, URIRef, Literal
from shortuuid import uuid

from agora_wot.evaluate import find_params, evaluate
from agora_wot.ns import CORE, WOT, MAP
from agora_wot.utils import encode_rdict

__author__ = 'Fernando Serena'

log = logging.getLogger('agora.wot')


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
                if isinstance(ext_node, BNode) and not list(graph.subjects(CORE.describes, ext_node)):
                    ignore = any(list(graph.triples((ext_node, x, None))) for x in filters)
                    if not ignore:
                        trace.add(ext_node)
                        for t in describe(graph, ext_node, filters=filters, trace=trace):
                            yield t

    if trace is None:
        trace = set([])
    for t in itertools.chain(desc()):
        yield t


def bound_graph():
    g = ConjunctiveGraph()
    g.bind('core', CORE)
    g.bind('wot', WOT)
    g.bind('map', MAP)
    return g


def load_component(uri, graph, trace=None):
    if trace is None:
        trace = []

    if uri in trace:
        return

    response = requests.get(uri)
    trace.append(uri)
    if response.status_code == 200:
        comp_ttl = response.content
        comp_g = Graph()
        comp_g.parse(StringIO(comp_ttl), format='turtle')
        for s, p, o in comp_g:
            graph.add((s, p, o))

    try:
        td_uri = list(comp_g.objects(uri, CORE.describedBy)).pop()
        load_component(td_uri, graph, trace=trace)

        for s, p, o in comp_g:
            if p != RDF.type:
                if isinstance(o, URIRef):
                    load_component(o, graph, trace=trace)
                if isinstance(s, URIRef):
                    load_component(s, graph, trace=trace)
    except IndexError:
        pass

    try:
        bp_uri = list(comp_g.objects(uri, CORE.describes)).pop()
        load_component(bp_uri, graph, trace=trace)
    except IndexError:
        pass


class TD(object):
    def __init__(self, resource, id=None):
        # type: (Resource) -> None
        self.__resource = resource
        self.__id = id if id is not None else uuid()
        self.__access_mappings = set([])
        self.__rdf_sources = set([])
        self.__vars = set([])
        self.__endpoints = set([])
        self.__node = None

    @staticmethod
    def from_graph(graph, node, node_map):
        if node in node_map:
            return node_map[node]

        try:
            r_node = list(graph.objects(subject=node, predicate=CORE.describes)).pop()
        except IndexError:
            raise ValueError

        resource = Resource.from_graph(graph, r_node, node_map=node_map)

        td = TD(resource)
        td.__node = node
        node_map[node] = td

        try:
            td.__id = list(graph.objects(node, CORE.identifier)).pop().toPython()
        except IndexError:
            pass

        for mr_node in graph.objects(node, MAP.hasAccessMapping):
            mr = AccessMapping.from_graph(graph, mr_node, node_map=node_map)
            td.add_access_mapping(mr)

        for rs_node in graph.objects(node, MAP.fromRDFSource):
            rdf_source = RDFSource.from_graph(graph, rs_node, node_map=node_map)
            td.add_rdf_source(rdf_source)

        return td

    def to_graph(self, graph=None, node=None, td_nodes=None, th_nodes=None):
        if graph is None:
            graph = bound_graph()
        if node is None:
            node = self.node

        resource_node = self.resource.node
        if th_nodes:
            resource_node = th_nodes.get(resource_node, resource_node)

        graph.add((node, RDF.type, CORE.ThingDescription))
        graph.add((node, CORE.describes, resource_node))
        graph.add((node, CORE.identifier, Literal(self.id)))
        for am in self.access_mappings:
            am_node = BNode()
            graph.add((node, MAP.hasAccessMapping, am_node))
            am.to_graph(graph=graph, node=am_node, td_nodes=td_nodes)

        for rdfs in self.rdf_sources:
            r_node = BNode()
            graph.add((node, MAP.fromRDFSource, r_node))
            rdfs.to_graph(graph=graph, node=r_node)

        r_node = self.resource.node
        if not (th_nodes and r_node in th_nodes):
            r_graph = self.resource.to_graph()
            for s, p, o in r_graph:
                ss = th_nodes.get(s, s) if th_nodes else s
                oo = th_nodes.get(o, o) if th_nodes else o
                graph.add((ss, p, oo))

        return graph

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
        new.__node = self.node
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
        return self.__vars

    @property
    def node(self):
        return self.__node


class Resource(object):
    def __init__(self, uri=None, types=None):
        # type: (any, iter) -> None
        self.__graph = bound_graph()
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

        for prefix, ns in graph.namespaces():
            r.__graph.bind(prefix, ns)

        r.__node = node
        r.__types = set(graph.objects(node, RDF.type))

        node_map[node] = r
        return r

    def to_graph(self, graph=None):
        if graph is not None:
            graph.__iadd__(self.graph)

        return self.graph

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

    def to_graph(self, graph=None, node=None, td_nodes=None):
        if graph is None:
            graph = bound_graph()
        if node is None:
            node = BNode()

        graph.add((node, RDF.type, MAP.AccessMapping))

        e_node = BNode()
        graph.add((node, MAP.mapsResourcesFrom, e_node))
        self.endpoint.to_graph(graph=graph, node=e_node)

        for m in self.mappings:
            m_node = BNode()
            m.to_graph(graph=graph, node=m_node, td_nodes=td_nodes)
            graph.add((node, MAP.hasMapping, m_node))

        return graph

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

    def to_graph(self, graph=None, node=None):
        if graph is None:
            graph = bound_graph()
        if node is None:
            node = BNode()

        graph.add((node, RDF.type, WOT.Endpoint))
        if self.href:
            graph.add((node, WOT.href, Literal(self.href)))
        if self.order:
            graph.add((node, WOT.href, Literal(self.order)))
        if self.whref:
            graph.add((node, WOT.withHRef, Literal(self.whref)))
        graph.add((node, WOT.mediaType, Literal(self.media)))

        return graph

    def __add__(self, other):
        endpoint = Endpoint()
        if isinstance(other, Endpoint):
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

        filtered_query = {}
        href_parse = urlparse(href)
        for v, vals in parse_qs(href_parse[4]).items():
            if not any(map(lambda x: x.startswith('$'), vals)):
                filtered_query[v] = vals.pop()

        query_str = urlencode(filtered_query)
        href_parse = list(href_parse[:])
        href_parse[4] = query_str
        href = urlunparse(tuple(href_parse))

        log.debug(u'getting {}'.format(href))
        if self.intercept:
            r = self.intercept(href)
        else:
            r = requests.get(href, headers={'Accept': self.media})
        if self.response_headers is not None:
            r.headers.update(self.response_headers)
        return r


class Mapping(object):
    def __init__(self, key=None, predicate=None, transform=None, path=None, limit=None, root=False):
        self.key = key
        if predicate is not None:
            predicate = URIRef(predicate)
        self.predicate = predicate
        self.transform = transform
        self.path = path
        self.root = root
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
            mapping.root = list(graph.objects(node, MAP.rootMode)).pop()
        except IndexError:
            pass

        try:
            mapping.transform = create_transform(graph, list(graph.objects(node, MAP.valuesTransformedBy)).pop(),
                                                 node_map)
        except IndexError:
            pass

        node_map[node] = mapping
        return mapping

    def to_graph(self, graph=None, node=None, td_nodes=None):
        if graph is None:
            graph = bound_graph()
        if node is None:
            node = BNode()

        graph.add((node, RDF.type, MAP.Mapping))
        graph.add((node, MAP.predicate, URIRef(self.predicate)))
        graph.add((node, MAP.key, Literal(self.key)))
        graph.add((node, MAP.rootMode, Literal(self.root)))
        if self.path:
            graph.add((node, MAP.jsonPath, Literal(self.path)))
        if self.transform:
            transform_node = td_nodes.get(self.transform.td, None) if td_nodes else None
            if transform_node:
                graph.add((node, MAP.valuesTransformedBy, transform_node))

        return graph


def create_transform(graph, node, node_map):
    # if isinstance(node, URIRef):
    #     load_component(node, graph)

    if list(graph.triples((node, RDF.type, CORE.ThingDescription))):
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
        if node in node_map:
            td = node_map[node]
        else:
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
            vars = self.td.vars
            parent_item = kwargs.get('$item', None) if '$parent' in vars else None
            base_rdict = {"$parent": parent_item} if parent_item is not None else {}
            for var in filter(lambda x: x in kwargs, vars):
                base_rdict[var] = kwargs[var]
            res = [uri_provider(self.td.resource.node, encode_rdict(merge({"$item": v}, base_rdict))) for v in data]
            return res
        return data


class TED(object):
    def __init__(self):
        self.__ecosystem = Ecosystem()

    @staticmethod
    def from_graph(graph):
        ted = TED()
        ted.__ecosystem = Ecosystem.from_graph(graph)
        return ted

    def to_graph(self, graph=None, node=None, td_nodes=None, th_nodes=None):
        if graph is None:
            graph = bound_graph()
        if node is None:
            node = BNode()

        eco_node = BNode()
        self.ecosystem.to_graph(graph=graph, node=eco_node, td_nodes=td_nodes, th_nodes=th_nodes)
        graph.add((node, RDF.type, CORE.TED))
        graph.add((node, CORE.describes, eco_node))

        if td_nodes:
            for td_node in td_nodes.values():
                graph.add((node, CORE.usesTD, td_node))

        return graph

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
        td_nodes_dict = {}

        load_trace = []

        for _, ext_td in graph.subject_objects(MAP.valuesTransformedBy):
            if isinstance(ext_td, URIRef):
                load_component(ext_td, graph, trace=load_trace)

        for r_node in graph.objects(node, CORE.hasComponent):
            if isinstance(r_node, URIRef) and not list(graph.objects(r_node, RDF.type)):
                load_component(r_node, graph, trace=load_trace)
            try:
                td_node = list(graph.subjects(predicate=CORE.describes, object=r_node)).pop()
                td = TD.from_graph(graph, td_node, node_map=node_block_map)
                eco.add_component_from_td(td)
                td_nodes_dict[r_node] = td
            except IndexError:
                resource = Resource.from_graph(graph, r_node, node_map=node_block_map)
                eco.add_component(resource)

            root_nodes.add(r_node)

        for td_node, r_node in graph.subject_objects(predicate=CORE.describes):
            if (td_node, RDF.type, CORE.ThingDescription) in graph and r_node not in root_nodes:
                td = TD.from_graph(graph, td_node, node_map=node_block_map)
                eco.__tds.add(td)
                eco.__resources.add(td.resource)
                td_nodes_dict[r_node] = td

        for td in eco.__tds:
            for s, p, o in td.resource.graph.triples((None, None, None)):
                if o in td_nodes_dict:
                    td.vars.update(td_nodes_dict[o].vars)

        return eco

    def to_graph(self, graph=None, node=None, td_nodes=None, th_nodes=None):
        if graph is None:
            graph = bound_graph()
        if node is None:
            node = BNode()
        if td_nodes is None:
            td_nodes = {td: BNode() for td in self.tds}

        graph.add((node, RDF.type, CORE.Ecosystem))
        roots = self.roots

        for td in self.tds:
            if td in roots:
                graph.add(
                    (node, CORE.hasComponent, td.resource.node if not th_nodes else th_nodes.get(td.resource.node)))

            td_node = td_nodes[td]
            td.to_graph(graph=graph, node=td_node, td_nodes=td_nodes, th_nodes=th_nodes)

        for root in filter(lambda x: not isinstance(x, TD), self.roots):
            graph.add((node, CORE.hasComponent, root.node))
            root.to_graph(graph)

        return graph

    @property
    def resources(self):
        return frozenset(self.__resources)

    @property
    def tds(self):
        return self.__tds

    @property
    def endpoints(self):
        yielded = []
        for td in self.__tds:
            for am in td.access_mappings:
                e = am.endpoint
                if e not in yielded:
                    yielded.append(e)
                    yield e

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

        return network

    def tds_by_type(self, t):
        for td in self.__root_tds:
            if t in td.resource.types:
                yield td

    def resources_by_type(self, t):
        try:
            for r in filter(lambda x: isinstance(x.node, URIRef), self.__roots):
                yield r
        except Exception:
            pass
