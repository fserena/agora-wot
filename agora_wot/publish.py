# coding=utf-8
"""
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=#
  Ontology Engineering Group
        http://www.oeg-upm.net/
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=#
  Copyright (C) 2016 Ontology Engineering Group.
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
from threading import Lock

from os.path import commonprefix

from flask import request
from rdflib import Graph, RDF
from rdflib import URIRef

from agora.engine.plan.graph import AGORA
from agora.server import Server
from agora_wot.proxy import Proxy

__author__ = 'Fernando Serena'

log = logging.getLogger('agora.ted.publish')


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


def build(proxy, server=None, import_name=__name__):
    # type: (Proxy, Server, str) -> Server

    if server is None:
        server = Server(import_name)

    gateway = Server('gateway')

    def serialize(g):
        turtle = g.serialize(format='turtle')
        gw_host = proxy.host + '/'
        if 'localhost' in gw_host and gw_host != request.host_url:
            turtle = turtle.replace(gw_host, request.host_url)
        return turtle

    @gateway.get(proxy.path, produce_types=('text/turtle', 'text/html'))
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

        return serialize(g)

    @gateway.get(proxy.path + '/<path:rid>', produce_types=('text/turtle', 'text/html'))
    def get_gw_resource(rid):
        g, headers = proxy.load(proxy.base + '/' + rid, **request.args)
        return serialize(g)

    return Dispatcher(gateway, server, proxy)
