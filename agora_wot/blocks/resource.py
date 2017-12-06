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

    def to_graph(self, graph=None, abstract=False):
        if graph is not None:
            if abstract:
                for t in self.graph.triples((self.node, RDF.type, None)):
                    graph.add(t)
            else:
                graph.__iadd__(self.graph)
        elif abstract:
            graph = Graph()
            for t in self.graph.triples((self.node, RDF.type, None)):
                graph.add(t)
            return graph

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
