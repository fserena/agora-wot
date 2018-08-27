"""
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=#
  Ontology Engineering Group
        http://www.oeg-upm.net/
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=#
  Copyright (C) 2017 Ontology Engineering Group.
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
import traceback
from SPARQLWrapper import SPARQLWrapper, JSON
from abc import abstractmethod
from rdflib import RDF, Graph
from rdflib.term import BNode, Literal, URIRef, Node
from shortuuid import uuid

from agora_wot.blocks.endpoint import Endpoint
from agora_wot.blocks.evaluate import find_params
from agora_wot.blocks.rdfsource import RDFSource
from agora_wot.blocks.resource import Resource
from agora_wot.blocks.utils import bound_graph, encode_rdict
from agora_wot.ns import CORE, MAP

__author__ = 'Fernando Serena'


class TD(object):
    def __init__(self, resource, id=None):
        # type: (Resource, str) -> TD
        self.__resource = resource
        self.__id = id if id is not None else uuid()
        self.__access_mappings = set()
        self.__rdf_sources = set()
        self.__vars = set()
        self.__endpoints = set()
        self.__node = None
        self.__td_ext = set()

    @staticmethod
    def from_graph(graph, node, node_map, **kwargs):
        if node in node_map:
            return node_map[node]

        try:
            r_node = list(graph.objects(subject=node, predicate=CORE.describes)).pop()
        except IndexError:
            raise ValueError('No described thing')

        if kwargs.get('loader', None) and not set(graph.triples((r_node, None, None))):
            graph.__iadd__(kwargs['loader'](r_node))

        resource = Resource.from_graph(graph, r_node, node_map=node_map, **kwargs)

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
                    if kwargs.get('loader', None) and not set(graph.triples((ext, None, None))):
                        graph.__iadd__(kwargs['loader'](ext))

                    ext_td = TD.from_graph(graph, ext, node_map, **kwargs)
                    node_map[ext] = ext_td
                td.__td_ext.add(ext_td)
        except (IndexError, ValueError):
            pass

        for ext_td in td.__td_ext:
            for am in ext_td.access_mappings:
                td.add_access_mapping(am, own=False)

        for mr_node in graph.objects(node, MAP.hasAccessMapping):
            try:
                mr = AccessMapping.from_graph(graph, mr_node, node_map=node_map, **kwargs)
                td.add_access_mapping(mr)
            except Exception as e:
                traceback.print_exc()
                print e.message

        for rs_node in graph.objects(node, MAP.fromRDFSource):
            rdf_source = RDFSource.from_graph(graph, rs_node, node_map=node_map)
            td.add_rdf_source(rdf_source)

        return td

    def to_graph(self, graph=None, node=None, td_nodes=None, th_nodes=None, abstract=False, **kwargs):
        if td_nodes is None:
            td_nodes = {}

        if node is None:
            node = td_nodes.get(self, None) or self.node or BNode()

        if graph is None:
            graph = bound_graph(str(node))

        if (node, None, None) in graph:
            return graph

        resource_node = self.resource.node
        if th_nodes:
            resource_node = th_nodes.get(resource_node, resource_node)

        graph.add((node, RDF.type, CORE.ThingDescription))
        graph.add((node, CORE.describes, resource_node))
        graph.add((node, CORE.identifier, Literal(self.id)))
        for ext in self.extends:
            graph.add((node, CORE.extends, td_nodes.get(ext, ext.node)))

        if not abstract:
            for am in self.__access_mappings:
                if td_nodes:
                    if am not in td_nodes:
                        td_nodes[am] = BNode()
                    am_node = td_nodes[am]
                else:
                    am_node = BNode()
                graph.add((node, MAP.hasAccessMapping, am_node))
                am.to_graph(graph=graph, node=am_node, td_nodes=td_nodes, **kwargs)

            for rdfs in self.rdf_sources:
                if td_nodes:
                    if rdfs not in td_nodes:
                        td_nodes[rdfs] = BNode()
                    r_node = td_nodes[rdfs]
                else:
                    r_node = BNode()
                graph.add((node, MAP.fromRDFSource, r_node))
                rdfs.to_graph(graph=graph, node=r_node)

            r_node = self.resource.node
            if not (th_nodes is None or r_node in th_nodes):
                r_graph = self.resource.to_graph(**kwargs)
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

    def remove_access_mapping(self, am):
        self.__access_mappings.remove(am)
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
    def direct_vars(self):
        # type: (None) -> iter
        return reduce(lambda x, y: x.union(y.endpoint_vars), self.access_mappings, set())

    @property
    def node(self):
        return self.__node

    @property
    def extends(self):
        return self.__td_ext

    @property
    def endpoints(self):
        return self.__endpoints


class AccessMapping(object):
    def __init__(self, endpoint):
        # type: (Endpoint) -> None

        self.mappings = set([])
        self.__order = None
        self.__vars = set([])

        self.__endpoint = endpoint
        self.__find_vars()
        self.id = uuid()

    def __find_endpoint_vars(self):
        if self.__endpoint:
            ref = self.__endpoint.href
            for param in find_params(str(ref)):
                yield param

    def __find_vars(self):
        if self.__endpoint:
            self.__vars.update(set(self.__find_endpoint_vars()))

        for m in self.mappings:
            if isinstance(m.transform, ResourceTransform):
                self.__vars.update(m.transform.td.vars)

    @staticmethod
    def from_graph(graph, node, node_map, **kwargs):
        # type: (Graph, Node) -> iter
        if node in node_map:
            return node_map[node]

        try:
            e_node = list(graph.objects(node, MAP.mapsResourcesFrom)).pop()
        except IndexError:
            pass

        endpoint = Endpoint.from_graph(graph, e_node, node_map=node_map)

        am = AccessMapping(endpoint)

        try:
            for m in graph.objects(node, MAP.hasMapping):
                am.mappings.add(Mapping.from_graph(graph, m, node_map=node_map, **kwargs))
                am.__find_vars()
        except IndexError:
            pass

        try:
            am.order = list(graph.objects(node, MAP.order)).pop().toPython()
        except IndexError:
            pass

        try:
            am.id = list(graph.objects(node, CORE.identifier)).pop()
        except IndexError:
            pass

        node_map[node] = am
        return am

    def to_graph(self, graph=None, node=None, td_nodes=None, **kwargs):
        if node is None:
            node = BNode()
        if graph is None:
            graph = bound_graph(identifier=str(node))

        if (node, None, None) in graph:
            return graph

        graph.add((node, RDF.type, MAP.AccessMapping))
        graph.add((node, CORE.identifier, Literal(self.id)))

        if td_nodes:
            if self.endpoint not in td_nodes:
                td_nodes[self.endpoint] = BNode()
            e_node = td_nodes[self.endpoint]
        else:
            e_node = self.endpoint.node or BNode()

        graph.add((node, MAP.mapsResourcesFrom, e_node))
        self.endpoint.to_graph(graph=graph, node=e_node)

        for m in self.mappings:
            if td_nodes:
                if m not in td_nodes:
                    td_nodes[m] = BNode()
                m_node = td_nodes[m]
            else:
                m_node = BNode()
            m.to_graph(graph=graph, node=m_node, td_nodes=td_nodes, **kwargs)
            graph.add((node, MAP.hasMapping, m_node))

        if self.__order is not None:
            graph.add((node, MAP.order, Literal(self.order)))

        return graph

    @property
    def vars(self):
        return frozenset(self.__vars)

    @property
    def endpoint_vars(self):
        return frozenset(self.__find_endpoint_vars())

    @property
    def endpoint(self):
        return self.__endpoint

    @property
    def order(self):
        return self.__order if self.__order is not None else 1000

    @order.setter
    def order(self, o):
        self.__order = o


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
        self.id = uuid()

    @staticmethod
    def from_graph(graph, node, node_map, **kwargs):
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
                                                 node_map, target=mapping.target_class or mapping.target_datatype,
                                                 **kwargs)
        except IndexError:
            pass

        try:
            mapping.id = list(graph.objects(node, CORE.identifier)).pop()
        except IndexError:
            pass

        node_map[node] = mapping
        return mapping

    def to_graph(self, graph=None, node=None, td_nodes=None, **kwargs):
        if node is None:
            node = BNode()
        if graph is None:
            graph = bound_graph()

        if (node, None, None) in graph:
            return graph

        graph.add((node, RDF.type, MAP.Mapping))
        graph.add((node, CORE.identifier, Literal(self.id)))
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
                transform_node = td_nodes.get(self.transform.td, None) if td_nodes else self.transform.td.node
            elif isinstance(self.transform, URITransform):
                transform_node = self.transform.uri
            else:
                transform_node = BNode()
                self.transform.to_graph(graph=graph, node=transform_node, **kwargs)

            if transform_node:
                graph.add((node, MAP.valuesTransformedBy, transform_node))

        return graph


def create_transform(graph, node, node_map, target=None, fetch=False, loader=None):
    if isinstance(node, URIRef) and not list(graph.objects(node, RDF.type)):
        if fetch:
            if not loader:
                graph.load(node, format='application/ld+json')
            else:
                f_graph = loader(node)
                graph.__iadd__(f_graph)
        else:
            return URITransform(node)
    if list(graph.triples((node, RDF.type, CORE.ThingDescription))):
        return ResourceTransform.from_graph(graph, node, node_map=node_map, target=target, fetch=fetch, loader=loader)
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


class URITransform(Transform):
    def __init__(self, uri):
        self.uri = uri


class ResourceTransform(Transform):
    def __init__(self, td, target=None):
        # type: (TD) -> None
        self.td = td
        self.target = target

    @staticmethod
    def from_graph(graph, node, node_map, target=None, fetch=False, **kwargs):
        if node in node_map:
            td = node_map[node]
        else:
            td = TD.from_graph(graph, node, node_map, fetch=fetch, **kwargs)
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
        if node is None:
            node = BNode()
        if graph is None:
            graph = bound_graph(str(node))

        if (node, None, None) in graph:
            return graph

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
        if node is None:
            node = BNode()
        if graph is None:
            graph = bound_graph(str(node))

        if (node, None, None) in graph:
            return graph

        graph.add((node, RDF.type, MAP.SPARQLQuery))
        graph.add((node, MAP.queryText, Literal(self.query)))
        graph.add((node, MAP.sparqlHost, Literal(self.host)))

        return graph

    def apply(self, data, *args, **kwargs):
        if not isinstance(data, list):
            data = [data]
        solutions = set()
        for elm in data:
            try:
                query = self.query.replace('$item', elm)
            except TypeError:
                continue

            sparql = SPARQLWrapper(self.host)
            sparql.setReturnFormat(JSON)

            sparql.setQuery(query)

            try:
                results = sparql.query().convert()

                for result in results["results"]["bindings"]:
                    r = result[result.keys().pop()]["value"]
                    solutions.add(r)
            except Exception:
                pass

        return list(solutions)
