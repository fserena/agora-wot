# coding=utf-8
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
import json
import logging
from os.path import commonprefix
from threading import Lock

from agora import RedisCache
from agora.engine.plan.graph import AGORA
from agora.server import Server
from flask import request
from gunicorn.app.base import BaseApplication
from gunicorn.six import iteritems
from pyld import jsonld
from rdflib import Graph, RDF, ConjunctiveGraph, BNode
from rdflib import URIRef, XSD, Literal
from rdflib.namespace import NamespaceManager

from agora_wot.gateway.proxy import Proxy

__author__ = 'Fernando Serena'

log = logging.getLogger('agora.ted.publish')

ACCEPT_MIMES = ['text/turtle', 'text/html', 'application/ld+json']
SKOLEM_BASE = 'http://bnodes'


class Application(BaseApplication):
    def init(self, **kwargs):
        pass

    def __init__(self, dispatcher, options=None):
        self.lock = Lock()
        self.dispatcher = dispatcher
        self.options = options or {}
        self.init()
        super(Application, self).__init__()

    def load_config(self):
        config = dict([(key, value) for key, value in iteritems(self.options)
                       if key in self.cfg.settings and value is not None])
        for key, value in iteritems(config):
            self.cfg.set(key.lower(), value)

            # self.cfg.set('worker_class', 'gateway.server.worker.Worker')

    def load(self):
        return self.dispatcher


class Dispatcher(object):
    def __init__(self, gw, server, proxy):
        self.gw = gw
        self.server = server
        self.proxy = proxy
        self.lock = Lock()

    def get_application(self, environ):
        adapter = self.gw.url_map.bind_to_environ(environ)
        if adapter.test():
            return self.gw
        else:
            prefix = commonprefix([self.proxy.path, environ['PATH_INFO']])
            if prefix != '/':
                environ['PATH_INFO'] = '/' + environ['PATH_INFO'].replace(prefix, '').lstrip('/')
            return self.server

    def __call__(self, environ, start_response):
        app = self.get_application(environ)
        return app(environ, start_response)


def request_wants_turtle():
    best = request.accept_mimetypes \
        .best_match(ACCEPT_MIMES)
    return best == 'text/turtle' and \
           request.accept_mimetypes[best] > \
           request.accept_mimetypes['application/ld+json']


def include_prefix(term, prefixes, context):
    try:
        prefix, ns = filter(lambda (prefix, ns): ns in term, prefixes).pop()
        prefixes.remove((prefix, ns))
        context[prefix] = str(ns)
    except IndexError:
        pass


def n3_context(graph, term, prefixes, object=None):
    context = {}
    n3 = term.n3(graph.namespace_manager)
    if not n3.startswith('<'):
        include_prefix(term, prefixes, context)
        key = n3.split(':')[1]
        if object is not None:
            type = object.datatype if isinstance(object, Literal) else '@id'
            if type and type != XSD.string:
                include_prefix(type, prefixes, context)
                context[key] = {
                    '@type': type.n3(graph.namespace_manager) if isinstance(type, URIRef) else type,
                    '@id': n3
                }
            else:
                context.update({key: n3})
        else:
            context.update({key: n3})
    return context


def build_graph_context(g):
    context = {}
    mapped_uris = set()
    prefixes = list(g.namespaces())
    for s, p, o in g:
        term, object = (p, o) if p != RDF.type else (o, None)
        if term not in mapped_uris:
            term_ctx = n3_context(g, term, prefixes, object=object)
            if term_ctx:
                context.update(term_ctx)
                mapped_uris.add(term)
    return context


def skolemize(g):
    bn_map = {}
    skolem = ConjunctiveGraph()
    for s, p, o in g:
        if isinstance(s, BNode):
            if s not in bn_map:
                bn_map[s] = URIRef('/'.join([SKOLEM_BASE, str(s)]))
            s = bn_map[s]
        if isinstance(o, BNode):
            if o not in bn_map:
                bn_map[o] = URIRef('/'.join([SKOLEM_BASE, str(o)]))
            o = bn_map[o]
        skolem.add((s, p, o))
    return skolem


def serialize_in_json(g, uri):
    context = build_graph_context(g)
    cg = skolemize(g)
    ted_nquads = cg.serialize(format='nquads')
    ld = jsonld.from_rdf(ted_nquads)
    type = list(cg.objects(uri, RDF.type)).pop()
    ld = jsonld.frame(ld, {'context': context, '@type': str(type)})
    return json.dumps(jsonld.compact(ld, context), indent=3, sort_keys=True)


def build(proxy, server=None, cache=None, import_name=__name__):
    # type: (Proxy, Server, RedisCache, str) -> Server

    if server is None:
        server = Server(import_name)

    gateway = Server('gateway')

    def serialize(g, uri):
        def match_ns(term):
            filter_ns = [ns for ns in rev_ns if ns in term]
            if filter_ns:
                ns = filter_ns.pop()
                if rev_ns[ns]:
                    found_ns.append((rev_ns[ns], ns))
                del rev_ns[ns]

        rev_ns = {ns: prefix for prefix, ns in g.namespaces()}
        found_ns = []
        for s, p, o in g:
            match_ns(s)
            match_ns(p)
            match_ns(o)

        g.namespace_manager = NamespaceManager(Graph())
        for (prefix, ns) in found_ns:
            g.bind(prefix, ns)

        format = 'text/turtle' if request_wants_turtle() else 'application/ld+json'
        if format == 'text/turtle':
            g_str = g.serialize(format='turtle')
        else:
            g_str = serialize_in_json(g, uri=uri)
        gw_host = proxy.host + '/'
        if 'localhost' in gw_host and gw_host != request.host_url:
            g_str = g_str.replace(gw_host, request.host_url)
        return g_str

    @gateway.get(proxy.path, produce_types=('text/turtle', 'text/html', 'application/ld+json'))
    def get_proxy():
        g = Graph()
        base = proxy.base
        if request.args:
            base += '?' + request.query_string
        proxy_uri = URIRef(base)
        g.bind('agora', AGORA)
        g.add((proxy_uri, RDF.type, AGORA.Gateway))
        seeds = proxy.instantiate_seeds(**request.args)
        seed_uris = set(reduce(lambda x, y: x + y, seeds.values(), []))
        for s_uri in seed_uris:
            r_uri = URIRef(s_uri)
            g.add((proxy_uri, AGORA.hasSeed, r_uri))

        return serialize(g, proxy_uri)

    @gateway.get(proxy.path + '/<path:rid>', produce_types=('text/turtle', 'text/html', 'application/ld+json'))
    def get_gw_resource(rid):
        r_uri = URIRef(proxy.base + '/' + rid)
        if cache is not None:
            g, ttl = cache.create(gid=r_uri, loader=proxy.load, format='text/turtle')
            headers = {'Cache-Control': 'max-age={}'.format(ttl)}
        else:
            g, headers = proxy.load(r_uri, **request.args)
        return serialize(g, uri=URIRef(r_uri.replace('=', '%3D'))), headers

    return Dispatcher(gateway, server, proxy)
