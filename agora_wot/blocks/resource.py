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

from rdflib import Graph
from rdflib import RDF
from rdflib.term import BNode, URIRef

from agora_wot.blocks.utils import bound_graph, describe

__author__ = 'Fernando Serena'


class Resource(object):
    def __init__(self, uri=None, types=None):
        # type: (any, iter) -> None
        self.__graph = bound_graph(identifier=uri)
        self.__node = uri
        if self.__node is None:
            self.__node = BNode()
        if types is None:
            types = []
        self.__types = set(types)
        for t in self.__types:
            self.__graph.add((self.__node, RDF.type, URIRef(t)))

    @staticmethod
    def from_graph(graph, node, node_map, fetch=False, loader=None):
        if node in node_map:
            return node_map[node]

        r = Resource(uri=node)

        if fetch and loader:
            f_graph = loader(r.node)
            r.__graph.__iadd__(f_graph)

        for s, p, o in describe(graph, node):
            if p != RDF.type or isinstance(o, URIRef):
                r.__graph.add((s, p, o))

        for prefix, ns in graph.namespaces():
            r.__graph.bind(prefix, ns)

        r.__node = node
        r.__types = filter(lambda t: isinstance(t, URIRef), set(graph.objects(node, RDF.type)))

        node_map[node] = r
        return r

    def to_graph(self, graph=None, abstract=False, fetch=True):
        base_g = self.graph if fetch else self.__graph
        res_g = Graph(identifier=self.node) if graph is None else graph
        for prefix, uri in base_g.namespaces():
            res_g.bind(prefix, uri)

        if abstract:
            for t in base_g.triples((self.node, RDF.type, None)):
                if isinstance(t[2], URIRef):
                    res_g.add(t)
        else:
            res_g.__iadd__(base_g)

        return res_g

    @property
    def types(self):
        if self.__types:
            return self.__types

        if not self.__graph and isinstance(self.__node, URIRef):
            try:
                self.__graph.load(self.__node, format='application/ld+json')
            except Exception as e:
                print  e.message
        return filter(lambda x: isinstance(x, URIRef),
                      set(self.__graph.objects(subject=self.__node, predicate=RDF.type)))

    @property
    def graph(self):
        if not self.__graph and isinstance(self.__node, URIRef):
            try:
                self.__graph.load(self.__node, format='application/ld+json')
            except Exception as e:
                print e.message
        return self.__graph

    @property
    def node(self):
        return self.__node
