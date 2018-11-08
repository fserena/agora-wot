"""
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=#
  Copyright (C) 2018 Fernando Serena
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
import base64
import copy
import itertools
import logging
import urlparse
from urllib import quote

from rdflib import ConjunctiveGraph, Graph, URIRef
from rdflib.term import BNode

from agora_wot.ns import CORE, WOT, MAP

__author__ = 'Fernando Serena'

log = logging.getLogger('agora.wot.blocks')


def encode_rdict(rd):
    sorted_keys = sorted(rd.keys())
    sorted_fields = []
    for k in sorted_keys:
        sorted_fields.append('"%s": "%s"' % (str(k), str(rd[k])))
    str_rd = '{' + ','.join(sorted_fields) + '}'
    return base64.b64encode(str_rd)


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


def bound_graph(identifier=None):
    g = ConjunctiveGraph(identifier=identifier)
    g.bind('core', CORE)
    g.bind('wot', WOT)
    g.bind('map', MAP)
    return g


def iriToUri(iri):
    # type: (str) -> str
    parts = urlparse.urlparse(iri)
    return urlparse.urlunparse(
        part.encode('utf-8') if parti == 1 else quote(part.encode('utf-8'))
        for parti, part in enumerate(parts)
    )


def get_ns(fountain):
    # type: (AbstractFountain) -> NamespaceManager
    g = Graph()
    prefixes = fountain.prefixes  # type:
    for prefix, ns in prefixes.items():
        g.bind(prefix, ns)
    return g.namespace_manager


def fltr(node, prefixes):
    # type: (object, iter) -> object
    if isinstance(node, dict):
        retVal = {}
        for key in node:
            if any([key.startswith(p) for p in prefixes]):
                child = fltr(node[key], prefixes)
                if child is not None:
                    retVal[key] = copy.deepcopy(child)
        if retVal:
            return retVal
        else:
            return None
    elif isinstance(node, list):
        retVal = []
        for entry in node:
            child = fltr(entry, prefixes)
            if child:
                retVal.append(child)
        if retVal:
            return retVal
        else:
            return None
    else:
        return node


def n3(t, ns):
    if isinstance(t, URIRef):
        t_n3 = t.n3(ns)
    else:
        t_n3 = t
    return t_n3
