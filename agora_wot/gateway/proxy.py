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
import logging
import sys

from agora.collector.execution import parse_rdf
from agora.collector.http import RDF_MIMES, http_get, extract_ttl
from agora.collector.wrapper import ResourceWrapper
from agora.engine.utils import Wrapper
from rdflib import Graph, ConjunctiveGraph, RDF
from rdflib.term import URIRef

from agora_wot.blocks.td import TD
from agora_wot.gateway.lifting import Lifter
from agora_wot.utils import encode_rdict, get_ns

__author__ = 'Fernando Serena'

log = logging.getLogger('agora.wot.proxy')


def _remove_type_redundancy(fountain, types, ns):
    known_types = fountain.types
    n3_types = {t.n3(ns): t for t in types}
    n3_filtered = filter(
        lambda t: t in known_types and not set.intersection(set(fountain.get_type(t)['sub']),
                                                            n3_types.keys()),
        n3_types.keys())
    return map(lambda t: n3_types.get(t), n3_filtered)


def process_arguments(**kwargs):
    return {k: v.pop() if isinstance(v, list) else v for k, v in kwargs.items()}


def url_for(wrapper, rvars):
    def f(tid, b64=None, **kwargs):
        if kwargs:
            var_dict = {}
            for var in filter(lambda x: x in kwargs, rvars):
                var_dict[var] = kwargs[var]
            b64 = encode_rdict(var_dict)
        if b64 is not None:
            b64 = unicode(b64)
        return wrapper.url_for('describe_resource', tid=tid, b64=b64)

    return f


def _parametrize_resource_uri(uri, **kwargs):
    if kwargs:
        uri = '{}?{}'.format(uri, '&'.join(['{}={}'.format(k, kwargs[k]) for k in kwargs]))
    return URIRef(uri)


def _load_remote(uri, format=None):
    for fmt in sorted(RDF_MIMES.keys(), key=lambda x: x != format):
        result = http_get(uri, format=fmt)
        if result is not None and not isinstance(result, bool):
            content, headers = result
            if not isinstance(content, Graph):
                g = ConjunctiveGraph()
                parse_rdf(g, content, fmt, headers)
                return g, headers


class Proxy(object):
    def __init__(self, ted, fountain, server_name='proxy', url_scheme='http', server_port=None, path=''):
        self.__ted = ted
        self.__fountain = fountain
        self.__seeds = set([])
        self.__wrapper = ResourceWrapper(server_name=server_name, url_scheme=url_scheme, server_port=server_port,
                                         path=path)
        self.__rvars = {t.resource.node: t.vars for t in ted.ecosystem.tds}
        self.__ndict = {t.resource.node: t.id for t in ted.ecosystem.tds}

        self.__wrapper.intercept('{}/<tid>'.format(path))(self.describe_resource)
        self.__wrapper.intercept('{}/<tid>/<b64>'.format(path))(self.describe_resource)
        self.__interceptor = None

        ns = get_ns(self.__fountain)

        for root in ted.ecosystem.roots:
            if isinstance(root, TD):
                if root.vars:
                    continue
                uri = URIRef(self.url_for(tid=root.id))
                resource = root.resource
            else:
                uri = root.node
                resource = root

            for t in resource.types:
                self.__seeds.add((uri, t.n3(ns)))

        self.__lifter = Lifter(self.__ted, self.__wrapper, self.url_for)

    def url_for(self, tid, b64=None, **kwargs):
        tid = self.__ndict.get(tid, tid)
        return url_for(self.__wrapper, self.__rvars)(tid, b64, **kwargs)

    def instantiate_seed(self, root, ns, **kwargs):
        if root in self.__ted.ecosystem.roots:
            uri = URIRef(self.url_for(root.id, **kwargs))
            for t in root.resource.types:
                t_n3 = t.n3(ns)
                self.__seeds.add((uri, t_n3))
                yield uri, t_n3

    @property
    def interceptor(self):
        return self.__interceptor

    @interceptor.setter
    def interceptor(self, i):
        self.__interceptor = i

    @property
    def parameters(self):
        params = set()
        for ty in self.ecosystem.root_types:
            for td in self.ecosystem.tds_by_type(ty):
                var_params = set([v.lstrip('$') for v in td.vars])
                params.update(var_params)

        return params

    def instantiate_seeds(self, **kwargs):
        seeds = {}
        kwargs = process_arguments(**kwargs)
        if self.interceptor:
            kwargs = self.interceptor(**kwargs)

        fountain = self.fountain
        ns = get_ns(fountain)

        root_types = reduce(lambda x, y: x.union(_remove_type_redundancy(fountain, y.types, ns)),
                            self.ecosystem.root_resources,
                            set())
        n3_root_types = {t.n3(ns): t for t in root_types}
        for ty in root_types:
            for td in self.ecosystem.tds_by_type(ty):
                try:
                    var_params = set([v.lstrip('$') for v in td.vars if v not in ['$item', '$parent']])
                    params = {'$' + v: kwargs[v] for v in var_params if v in kwargs}

                    if var_params and not params:
                        continue

                    for seed, t in self.instantiate_seed(td, ns, **params):
                        if t in n3_root_types:
                            if t not in seeds:
                                seeds[t] = []
                            seeds[t].append(seed)
                except KeyError:
                    pass

            for r in self.ecosystem.resources_by_type(ty):
                t = ty.n3(ns)
                if t in n3_root_types:
                    if t not in seeds:
                        seeds[t] = []
                    seeds[t].append(r.node)
        return seeds

    @property
    def fountain(self):
        return Wrapper(self.__fountain)

    @property
    def namespace_manager(self):
        prefixes = self.fountain.prefixes
        g = Graph()
        for prefix, uri in prefixes.items():
            g.bind(prefix, uri)
        return g.namespace_manager

    @property
    def ecosystem(self):
        return self.__ted.ecosystem

    @property
    def seeds(self):
        return frozenset(self.__seeds)

    @property
    def base(self):
        return self.__wrapper.base

    @property
    def host(self):
        return self.__wrapper.host

    @property
    def path(self):
        return self.__wrapper.path

    def load(self, uri, format=None, **kwargs):
        kwargs = {k: kwargs[k].pop() for k in kwargs}
        result = self.__wrapper.load(uri, **kwargs)

        if result is None:
            result = _load_remote(uri, format=format)
            base_graph = result[0]
            base_ttl = extract_ttl(result[1])
            types = base_graph.objects(URIRef(uri), RDF.type)
            enrichments = reduce(lambda x, y: y.union(x),
                                 map(lambda t: set(self.ecosystem.enrichments_by_type(t)), types), set())
            result = self.enrich_resource(uri, enrichments, base_graph, base_ttl)
        return result

    def enrich_resource(self, r_uri, enrichments, base_graph, base_ttl):
        min_ttl = sys.maxint
        for e in enrichments:
            base_graph, ttl = self.__lifter.lift(e.td.id, self.fountain, URIRef(r_uri), base_graph=base_graph,
                                                 replace=e.replace)
            if ttl < min_ttl:
                min_ttl = ttl
        result = (base_graph, {'Cache-Control': 'max-age={}'.format(int(min(min_ttl, base_ttl)))})
        return result

    def __gen_uri_args(self, tid, b64, **kwargs):
        if b64 is not None:
            b64 = b64.replace('%3D', '=')
            resource_args = eval(base64.b64decode(b64))
        else:
            resource_args = kwargs
        base_r_uri = self.url_for(tid=tid, b64=b64)
        r_uri = _parametrize_resource_uri(base_r_uri, **kwargs)
        return r_uri, resource_args

    def describe_resource(self, tid, b64=None, **kwargs):
        r_uri, resource_args = self.__gen_uri_args(tid, b64, **kwargs)
        g, ttl = self.__lifter.lift(tid, self.fountain, r_uri, b64, resource_args)
        return g, {'Cache-Control': 'max-age={}'.format(ttl)}

    def clear_seeds(self):
        self.__seeds.clear()
        for root in self.ecosystem.roots:
            resource = root.resource if isinstance(root, TD) else root
            for t in resource.types:
                t_n3 = t.n3(self.namespace_manager)
                self.fountain.delete_type_seeds(t_n3)
