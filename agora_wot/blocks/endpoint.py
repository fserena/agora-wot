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
from urlparse import urljoin, urlparse, parse_qs, urlunparse

import requests
from rdflib import Graph
from rdflib import RDF
from rdflib.term import Node, BNode, Literal

from agora_wot.blocks.evaluate import evaluate
from agora_wot.blocks.utils import bound_graph
from agora_wot.ns import WOT, CORE
from shortuuid import uuid

__author__ = 'Fernando Serena'

log = logging.getLogger('agora.wot.blocks')


class Endpoint(object):
    def __init__(self, href=None, media=None, whref=None, intercept=None, response_headers=None):
        self.href = href
        self.id = uuid()
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
            endpoint.id = list(graph.objects(node, CORE.identifier)).pop()
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
        if node is None:
            node = self.node or BNode()
        if graph is None:
            graph = bound_graph(identifier=str(node))

        if (node, None, None) in graph:
            return graph

        graph.add((node, RDF.type, WOT.Link))
        if self.href:
            graph.add((node, WOT.href, Literal(self.href)))
        if self.whref:
            graph.add((node, WOT.withHRef, Literal(self.whref)))
        graph.add((node, WOT.mediaType, Literal(self.media)))
        graph.add((node, CORE.identifier, Literal(self.id)))

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
                    elif not v.startswith('$'):
                        query_list.append(v)
        query_str = '&'.join(query_list)

        href_parse = list(href_parse[:])
        href_parse[4] = query_str
        href = urlunparse(tuple(href_parse))

        log.debug(u'getting {}'.format(href))
        if self.intercept:
            r = self.intercept(href)
        else:
            r = requests.get(href, headers={'Accept': self.media}, timeout=300)
        if self.response_headers is not None:
            r.headers.update(self.response_headers)
        return r
