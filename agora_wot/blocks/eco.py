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
import logging
from StringIO import StringIO

import networkx as nx
import requests
from rdflib import Graph
from rdflib import RDF
from rdflib.term import BNode, URIRef

from agora_wot.blocks.resource import Resource
from agora_wot.blocks.td import TD, ResourceTransform
from agora_wot.blocks.utils import bound_graph
from agora_wot.ns import CORE, MAP

__author__ = 'Fernando Serena'

log = logging.getLogger('agora.wot.blocks')


def request_loader(uri):
    try:
        response = requests.get(uri, headers={'Accept': 'text/turtle'})
        if response.status_code == 200:
            comp_ttl = response.content
            comp_g = Graph()
            comp_g.parse(StringIO(comp_ttl), format='turtle')
            return comp_g
    except:
        pass


def load_component(uri, graph, trace=None, loader=None, namespaces=None):
    if trace is None:
        trace = []

    if uri in trace:
        return

    if not isinstance(uri, URIRef):
        return

    if not namespaces:
        namespaces = dict(graph.namespaces()).values()

    # abort if candidate uri belongs to a namespace
    if any([ns in uri for ns in namespaces]):
        return

    log.debug(u'Fetching required component: {}...'.format(uri))

    if loader is None:
        loader = request_loader

    comp_g = loader(uri)
    trace.append(uri)
    if not comp_g:
        return

    if (uri, None, None) not in graph:
        for s, p, o in comp_g:
            graph.add((s, p, o))

    try:
        td_uri = list(comp_g.objects(uri, CORE.describedBy)).pop()
        load_component(td_uri, graph, trace=trace, loader=loader, namespaces=namespaces)

        for s, p, o in comp_g:
            if p != RDF.type:
                load_component(o, graph, trace=trace, loader=loader, namespaces=namespaces)
                load_component(s, graph, trace=trace, loader=loader, namespaces=namespaces)
    except IndexError:
        pass

    try:
        for _, ext_td_uri in list(comp_g.subject_objects(MAP.valuesTransformedBy)):
            load_component(ext_td_uri, graph, trace=trace, loader=loader, namespaces=namespaces)
    except IndexError:
        pass

    try:
        bp_uri = list(comp_g.objects(uri, CORE.describes)).pop()
        load_component(bp_uri, graph, trace=trace, loader=loader, namespaces=namespaces)
    except IndexError:
        pass


class Ecosystem(object):
    def __init__(self):
        # type: (None) -> None

        self.__resources = set([])
        self.__tds = set([])
        self.__roots = set([])
        self.__root_tds = set([])
        self.__node_td_map = {}
        self.__td_id_map = {}
        self.node = BNode()

    @staticmethod
    def from_graph(graph, loader=None, fetch=True, **kwargs):
        eco = Ecosystem()

        try:
            node = list(graph.subjects(RDF.type, CORE.Ecosystem)).pop()
            eco.node = node
        except IndexError:
            raise ValueError('Ecosystem node not found')

        node_block_map = {}
        root_nodes = set([])
        td_nodes_dict = {}

        load_trace = []
        namespaces = dict(graph.namespaces()).values()

        for r_node in graph.objects(node, CORE.hasComponent):
            if fetch and not list(graph.objects(r_node, RDF.type)):
                load_component(r_node, graph, trace=load_trace, loader=loader, namespaces=namespaces)

            root_nodes.add(r_node)

        if fetch:
            for _, ext_td in graph.subject_objects(CORE.extends):
                load_component(ext_td, graph, trace=load_trace, loader=loader, namespaces=namespaces)

            for _, ext_td in graph.subject_objects(MAP.valuesTransformedBy):
                load_component(ext_td, graph, trace=load_trace, loader=loader, namespaces=namespaces)

        for r_node in root_nodes:
            try:
                td_node = list(graph.subjects(predicate=CORE.describes, object=r_node)).pop()
                td = TD.from_graph(graph, td_node, node_map=node_block_map, loader=loader, **kwargs)
                eco.add_root_from_td(td)
                td_nodes_dict[r_node] = td
            except IndexError:
                resource = Resource.from_graph(graph, r_node, node_map=node_block_map)
                eco.add_root(resource)

        for td_node, r_node in graph.subject_objects(predicate=CORE.describes):
            if (td_node, RDF.type, CORE.ThingDescription) in graph and r_node not in root_nodes:
                td = TD.from_graph(graph, td_node, node_map=node_block_map, loader=loader, **kwargs)
                eco.add_td(td)
                eco.__resources.add(td.resource)
                td_nodes_dict[r_node] = td

        for td in eco.__tds:
            for s, p, o in td.resource.graph.triples((None, None, None)):
                if o in td_nodes_dict:
                    td.vars.update(td_nodes_dict[o].vars)

        return eco

    def to_graph(self, graph=None, node=None, td_nodes=None, th_nodes=None, abstract=False, **kwargs):
        if node is None:
            node = self.node or BNode()
        if graph is None:
            graph = bound_graph(str(node))

        if td_nodes is None:
            td_nodes = {td: td.node for td in self.tds}
        if th_nodes is None:
            th_nodes = {r: r.node for r in self.resources}

        graph.add((node, RDF.type, CORE.Ecosystem))
        roots = self.roots
        for td in self.tds:
            if td in roots:
                graph.add(
                    (node, CORE.hasComponent,
                     td.resource.node if not th_nodes else th_nodes.get(td.resource.node, td.resource.node)))

            if not abstract:
                td_node = td_nodes.get(td, None) if td_nodes else None
                for ext in td.extends:
                    ext.to_graph(graph=graph, node=ext.node, td_nodes=td_nodes, th_nodes=th_nodes, **kwargs)
                td.to_graph(graph=graph, node=td_node, td_nodes=td_nodes, th_nodes=th_nodes, **kwargs)

        for root in filter(lambda x: not isinstance(x, TD), self.roots):
            graph.add((node, CORE.hasComponent, root.node))
            root.to_graph(graph, **kwargs)

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

    def add_root_from_td(self, td):
        # type: (TD) -> None
        self.add_td(td)
        self.__root_tds.add(td)
        self.__roots.add(td)

    def add_root(self, root):
        # type: (object) -> None
        if isinstance(root, TD):
            self.add_root_from_td(root)
        else:
            self.__roots.add(root)
            self.__resources.add(root)

    def add_td(self, td, update_network=True):
        # type: (TD) -> None
        if td not in self.__tds:
            self.__remove_td_by_id(td)
            self.__tds.add(td)
            self.__resources.add(td.resource)
            self.__node_td_map[td.resource.node] = td
            self.__td_id_map[td.id] = td
            if update_network:
                self.network()

    @property
    def roots(self):
        return frozenset(self.__roots)

    @property
    def root_resources(self):
        return frozenset([root.resource if isinstance(root, TD) else root for root in self.__roots])

    @property
    def td_root_resources(self):
        return frozenset([root.resource for root in self.__roots if isinstance(root, TD)])

    @property
    def non_td_root_resources(self):
        return frozenset([root for root in self.__roots if not isinstance(root, TD)])

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
            network.add_node(td.id)
            transforming_mapping_sets = map(
                lambda am: filter(lambda m: isinstance(m.transform, ResourceTransform), am.mappings),
                td.access_mappings)
            transforming_mappings = reduce(lambda x, y: x.union(y), transforming_mapping_sets, set([]))
            for m in transforming_mappings:
                child_td = m.transform.td
                children.add(child_td)
                network.add_edge(td.id, child_td.id)

            for s, p, o in td.resource.graph:
                if o in self.__node_td_map:
                    child_td = self.__node_td_map[o]
                    children.add(child_td)
                    network.add_edge(td.id, child_td.id)

        for ch in children:
            self.add_td(ch, update_network=False)

        td_id_map = {td.id: td for td in self.__tds}

        for td in self.__tds:
            try:
                suc_td_ids = reduce(lambda x, y: x.union(set(y)), nx.dfs_successors(network, td.id).values(), set())
                suc_tds = map(lambda id: td_id_map[id], suc_td_ids)
                for std in suc_tds:
                    for var in std.vars:
                        td.vars.add(var)
            except KeyError:
                pass

        return network

    def tds_by_type(self, t):
        for td in self.__root_tds:
            if t in td.resource.types:
                yield td

    def resources_by_type(self, t):
        try:
            for r in filter(lambda x: isinstance(x.node, URIRef), self.__roots):
                if not isinstance(r, TD):
                    if t in r.types:
                        yield r
        except Exception:
            pass

    def root_vars(self, td):
        def __follow_suc(td_id):
            s_td = self.__td_id_map[td_id]
            td_vars = s_td.direct_vars
            if not td_vars:
                suc_vars = map(lambda x: __follow_suc(x), bfs_succ_dict.get(td_id, []))
                if suc_vars:
                    td_vars = set.union(*suc_vars)
            return td_vars

        bfs_succ = list(nx.bfs_successors(self.network(), td.id))
        bfs_succ_dict = dict(bfs_succ)
        vars = __follow_suc(td.id)

        return vars