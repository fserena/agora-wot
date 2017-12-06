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

from rdflib import Graph
from rdflib.term import Node

from agora_wot.blocks.endpoint import Endpoint
from agora_wot.blocks.evaluate import find_params

__author__ = 'Fernando Serena'

log = logging.getLogger('agora.wot.blocks')


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
