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

from rdflib import RDF
from rdflib.term import BNode

from agora_wot.blocks.eco import Ecosystem
from agora_wot.blocks.td import TD
from agora_wot.blocks.utils import bound_graph
from agora_wot.ns import CORE

__author__ = 'Fernando Serena'

log = logging.getLogger('agora.wot.blocks')


class TED(object):
    def __init__(self, node=None):
        self.node = node
        self.__ecosystem = Ecosystem()

    @staticmethod
    def from_graph(graph, **kwargs):
        ted = TED()
        ted.__ecosystem = Ecosystem.from_graph(graph, **kwargs)
        return ted

    def to_graph(self, graph=None, node=None, td_nodes=None, **kwargs):
        if node is None:
            node = self.node or BNode()
        if graph is None:
            graph = bound_graph(str(node))

        eco_node = self.ecosystem.node
        self.ecosystem.to_graph(graph=graph, node=eco_node, td_nodes=td_nodes, **kwargs)
        graph.add((node, RDF.type, CORE.ThingEcosystemDescription))
        graph.add((node, CORE.describes, eco_node))

        if td_nodes:
            for td_node in td_nodes.values():
                if isinstance(td_node, TD):
                    graph.add((node, CORE.usesTD, td_node))

        return graph

    @property
    def ecosystem(self):
        return self.__ecosystem

    @ecosystem.setter
    def ecosystem(self, eco):
        self.__ecosystem = eco
