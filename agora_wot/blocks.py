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
from StringIO import StringIO
from abc import abstractmethod
from urlparse import urljoin, urlparse, parse_qs, urlunparse

import networkx as nx
import requests
from SPARQLWrapper import SPARQLWrapper, JSON
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
    else:
        return  # should inform that

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
        for _, ext_td_uri in list(comp_g.subject_objects(MAP.valuesTransformedBy)):
            if isinstance(ext_td_uri, URIRef):
                load_component(ext_td_uri, graph, trace=trace)
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
        self.__access_mappings = set()
        self.__rdf_sources = set()
        self.__vars = set()
        self.__endpoints = set()
        self.__node = None
        self.__td_ext = set()

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

        try:
            for ext in set(graph.objects(node, CORE.extends)):
                if ext in node_map:
                    ext_td = node_map[ext]
                else:
                    ext_td = TD.from_graph(graph, ext, node_map)
                td.__td_ext.add(ext_td)
        except IndexError:
            pass

        for mr_node in graph.objects(node, MAP.hasAccessMapping):
            mr = AccessMapping.from_graph(graph, mr_node, node_map=node_map)
            td.add_access_mapping(mr)

        for ext_td in td.__td_ext:
            for am in ext_td.access_mappings:
                td.add_access_mapping(am, own=False)

        for rs_node in graph.objects(node, MAP.fromRDFSource):
            rdf_source = RDFSource.from_graph(graph, rs_node, node_map=node_map)
            td.add_rdf_source(rdf_source)

        return td

    def to_graph(self, graph=None, node=None, td_nodes=None, th_nodes=None):
        if graph is None:
            graph = bound_graph()
        if node is None:
            node = td_nodes.get(self, BNode()) if td_nodes else (self.node or BNode())

        resource_node = self.resource.node
        if th_nodes:
            resource_node = th_nodes.get(resource_node, resource_node)

        graph.add((node, RDF.type, CORE.ThingDescription))
        graph.add((node, CORE.describes, resource_node))
        graph.add((node, CORE.identifier, Literal(self.id)))
        for ext in self.extends:
            graph.add((node, CORE.extends, td_nodes.get(ext, ext)))

        for am in self.__access_mappings:
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

    def add_access_mapping(self, am, own=True):
        if own:
            self.__access_mappings.add(am)
        self.__vars = reduce(lambda x, y: set.union(x, y), [mr.vars for mr in self.access_mappings], set([]))
        self.__endpoints = set([mr.endpoint for mr in self.access_mappings])

    def add_rdf_source(self, source):
        self.__rdf_sources.add(source)

    def endpoint_mappings(self, e):
        return reduce(lambda x, y: set.union(x, y),
                      map(lambda x: x.mappings, filter(lambda x: x.endpoint == e, self.access_mappings)), set([]))

    def clone(self, id=None, **kwargs):
        new = TD(self.__resource, id=id)
        new.__node = self.node
        for am in self.__access_mappings:
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

        for td in self.extends:
            new.__td_ext.add(td)
        return new

    @property
    def id(self):
        # type: (None) -> str
        return self.__id

    @property
    def access_mappings(self):
        # type: (None) -> iter
        all_am = set()
        for ext in self.extends:
            all_am.update(ext.access_mappings)
        all_am.update(self.__access_mappings)
        return frozenset(all_am)

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

    @property
    def extends(self):
        return self.__td_ext


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
        self.__order = None
        self.__vars = set([])

        self.__endpoint = endpoint
        self.__find_vars()

    def __find_vars(self):
        if self.__endpoint:
            ref = self.__endpoint.href
            for param in find_params(str(ref)):
                self.__vars.add(param)

        for m in self.mappings:
            if isinstance(m.transform, ResourceTransform):
                self.__vars.update(m.transform.td.vars)

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
                am.__find_vars()
        except IndexError:
            pass

        try:
            am.order = list(graph.objects(node, MAP.order)).pop().toPython()
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

        e_node = self.endpoint.node or BNode()
        graph.add((node, MAP.mapsResourcesFrom, e_node))
        self.endpoint.to_graph(graph=graph, node=e_node)

        for m in self.mappings:
            m_node = BNode()
            m.to_graph(graph=graph, node=m_node, td_nodes=td_nodes)
            graph.add((node, MAP.hasMapping, m_node))

        if self.__order is not None:
            graph.add((node, MAP.order, Literal(self.order)))

        return graph

    @property
    def vars(self):
        return frozenset(self.__vars)

    @property
    def endpoint(self):
        return self.__endpoint

    @property
    def order(self):
        return self.__order if self.__order is not None else 1000

    @order.setter
    def order(self, o):
        self.__order = o


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
    def __init__(self, href=None, media=None, whref=None, intercept=None, response_headers=None):
        self.href = href
        self.whref = whref
        self.media = media or 'application/json'
        self.intercept = intercept
        self.node = None
        self.response_headers = response_headers

    @staticmethod
    def from_graph(graph, node, node_map):
        # type: (Graph, Node) -> iter

        if node in node_map:
            return node_map[node]

        endpoint = Endpoint()
        endpoint.node = node
        try:
            endpoint.media = list(graph.objects(node, WOT.mediaType)).pop()
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
            node = self.node or BNode()

        graph.add((node, RDF.type, WOT.Endpoint))
        if self.href:
            graph.add((node, WOT.href, Literal(self.href)))
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

        href_parse = urlparse(href)
        query_list = []
        for v, vals in parse_qs(href_parse[4], keep_blank_values=1).items():
            if not any(map(lambda x: x.startswith('$'), vals)):
                for val in vals:
                    if val:
                        query_list.append(u'{}={}'.format(v, val))
                    else:
                        query_list.append(v)
        query_str = '&'.join(query_list)

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
    def __init__(self, key=None, predicate=None, transform=None, path=None, limit=None, root=False, target_class=None,
                 target_datatype=None):
        self.key = key
        if predicate is not None:
            predicate = URIRef(predicate)
        self.predicate = predicate
        self.transform = transform
        self.path = path
        self.root = root
        self.limit = limit
        self.target_class = target_class
        self.target_datatype = target_datatype

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
            mapping.target_class = list(graph.objects(node, MAP.targetClass)).pop()
        except IndexError:
            try:
                mapping.target_datatype = list(graph.objects(node, MAP.targetDatatype)).pop()
            except IndexError:
                pass

        try:
            mapping.transform = create_transform(graph, list(graph.objects(node, MAP.valuesTransformedBy)).pop(),
                                                 node_map, target=mapping.target_class or mapping.target_datatype)
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
        if self.target_class:
            graph.add((node, MAP.targetClass, URIRef(self.target_class)))
        elif self.target_datatype:
            graph.add((node, MAP.targetDatatype, URIRef(self.target_datatype)))
        if self.transform:
            if isinstance(self.transform, ResourceTransform):
                transform_node = td_nodes.get(self.transform.td, None) if td_nodes else None
            else:
                transform_node = BNode()
                self.transform.to_graph(graph=graph, node=transform_node)

            if transform_node:
                graph.add((node, MAP.valuesTransformedBy, transform_node))

        return graph


def create_transform(graph, node, node_map, target=None):
    if list(graph.triples((node, RDF.type, CORE.ThingDescription))):
        return ResourceTransform.from_graph(graph, node, node_map=node_map, target=target)
    if list(graph.triples((node, RDF.type, MAP.StringReplacement))):
        return StringReplacement.from_graph(graph, node)
    if list(graph.triples((node, RDF.type, MAP.SPARQLQuery))):
        return SPARQLQuery.from_graph(graph, node)


class Transform(object):
    def attach(self, data):
        def wrapper(*args, **kwargs):
            return self.apply(data, *args, **kwargs)

        return wrapper

    @abstractmethod
    def apply(self, data, *args, **kwargs):
        pass


class ResourceTransform(Transform):
    def __init__(self, td, target=None):
        # type: (TD) -> None
        self.td = td
        self.target = target

    @staticmethod
    def from_graph(graph, node, node_map, target=None):
        if node in node_map:
            td = node_map[node]
        else:
            td = TD.from_graph(graph, node, node_map)
        transform = ResourceTransform(td, target=target)
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
            if self.target:
                base_rdict['$target'] = self.target.n3(self.td.resource.graph.namespace_manager)
            for var in filter(lambda x: x in kwargs, vars):
                base_rdict[var] = kwargs[var]
            res = [uri_provider(self.td.resource.node, encode_rdict(merge({"$item": v}, base_rdict))) for v in data]
            return res
        return data


class StringReplacement(Transform):
    def __init__(self, match, replace):
        # type: (TD) -> None
        self.match = match
        self.replace = replace

    @staticmethod
    def from_graph(graph, node):
        match = ''
        try:
            match = list(graph.objects(node, MAP.match)).pop()
        except IndexError:
            pass

        replace = ''
        try:
            replace = list(graph.objects(node, MAP['replace'])).pop()
        except IndexError:
            pass

        transform = StringReplacement(match, replace)
        return transform

    def to_graph(self, graph=None, node=None, **kwargs):
        if graph is None:
            graph = bound_graph()
        if node is None:
            node = BNode()

        graph.add((node, RDF.type, MAP.StringReplacement))
        graph.add((node, MAP.match, Literal(self.match)))
        graph.add((node, MAP['replace'], Literal(self.replace)))

        return graph

    def apply(self, data, *args, **kwargs):
        return [data.replace(self.match, self.replace)]


class SPARQLQuery(Transform):
    def __init__(self, query, host):
        self.query = query
        self.host = host

    @staticmethod
    def from_graph(graph, node):
        query = ''
        try:
            query = list(graph.objects(node, MAP.queryText)).pop()
        except IndexError:
            pass

        host = ''
        try:
            host = list(graph.objects(node, MAP.sparqlHost)).pop()
        except IndexError:
            pass

        transform = SPARQLQuery(query, host)
        return transform

    def to_graph(self, graph=None, node=None, **kwargs):
        if graph is None:
            graph = bound_graph()
        if node is None:
            node = BNode()

        graph.add((node, RDF.type, MAP.SPARQLQuery))
        graph.add((node, MAP.queryText, Literal(self.query)))
        graph.add((node, MAP.sparqlHost, Literal(self.host)))

        return graph

    def apply(self, data, *args, **kwargs):
        query = self.query.replace('$item', data)

        sparql = SPARQLWrapper(self.host)
        sparql.setReturnFormat(JSON)

        sparql.setQuery(query)

        solutions = []
        try:
            results = sparql.query().convert()

            for result in results["results"]["bindings"]:
                r = result[result.keys().pop()]["value"]
                solutions.append(r)
        except Exception:
            pass

        return solutions


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
        graph.add((node, RDF.type, CORE.ThingEcosystemDescription))
        graph.add((node, CORE.describes, eco_node))

        if td_nodes:
            for td_node in td_nodes.values():
                graph.add((node, CORE.usesTD, td_node))

        return graph

    @property
    def ecosystem(self):
        return self.__ecosystem

    @ecosystem.setter
    def ecosystem(self, eco):
        self.__ecosystem = eco


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

        for _, ext_td in graph.subject_objects(CORE.extends):
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
                eco.add_td(td)
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

    def __remove_td_by_id(self, td):
        for prev_td in filter(lambda atd: atd.id == td.id, self.__tds):
            self.__tds.remove(prev_td)
            if prev_td in self.__root_tds:
                self.__root_tds.remove(prev_td)
            if prev_td in self.__roots:
                self.__roots.remove(prev_td)
            self.__resources.remove(prev_td.resource)

    def add_component_from_td(self, td):
        # type: (TD) -> None
        if td not in self.__tds:
            self.__remove_td_by_id(td)
            self.__tds.add(td)
            self.__resources.add(td.resource)
            self.network()
        self.__root_tds.add(td)
        self.__roots.add(td)

    def add_component(self, resource):
        # type: (Resource) -> None
        self.__roots.add(resource)
        self.__resources.add(resource)

    def add_td(self, td):
        # type: (TD) -> None
        if td not in self.__tds:
            self.__remove_td_by_id(td)
            self.__tds.add(td)
            self.__resources.add(td.resource)
            self.network()

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
            self.add_td(ch)

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
