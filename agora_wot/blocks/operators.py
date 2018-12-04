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
from urlparse import urlparse

from agora.engine.plan.agp import extend_uri
from rdflib import URIRef, Graph
from rdflib.term import Node

__author__ = 'Fernando Serena'


def lslug(url, **kwargs):
    # type: (basestring) -> basestring
    return urlparse(url, allow_fragments=True).path.split('/')[-1]


def _build_pred_uri(pred_str, graph):
    pred_uri = URIRef(extend_uri(pred_str, dict(graph.namespaces())))
    return pred_uri


def objectValue(pred_str, graph, subject, **kwargs):
    # type: (str, Graph, Node) -> iter[str]
    pred_uri = _build_pred_uri(pred_str)
    return list(graph.objects(subject, pred_uri)).pop().toPython()


def filterObjects(pred_str, pattern, graph, subject, **kwargs):
    # type: (iter, str) -> str
    pred_uri = _build_pred_uri(pred_str, graph)
    objects = list(graph.objects(subject, pred_uri))
    return filter(lambda o: pattern in o, objects).pop().toPython()
