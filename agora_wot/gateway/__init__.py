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
import hashlib
from abc import abstractmethod
from agora.collector.execution import parse_rdf
from agora.collector.http import http_get, RDF_MIMES
from agora.engine.utils import Wrapper
from agora.server.fountain import build as bn
from agora.server.fragment import build as bf
from agora.server.planner import build as bp
from agora.server.sparql import build as bs
from rdflib import Graph, ConjunctiveGraph

from agora_wot.gateway.proxy import Proxy
from agora_wot.gateway.publish import build as bpp

__author__ = 'Fernando Serena'


class AbstractGateway(object):
    @abstractmethod
    def query(self, query, stop_event=None, scholar=False, **kwargs):
        # type: (str, object, bool, **any) -> iter
        raise NotImplementedError

    @abstractmethod
    def fragment(self, query, stop_event=None, scholar=False, **kwargs):
        # type: (str, object, bool, **any) -> iter
        raise NotImplementedError

    @abstractmethod
    def shutdown(self):
        # type: (str) -> None
        raise NotImplementedError


class Gateway(AbstractGateway):
    def __init__(self, agora, ted, cache=None, server_name='localhost', port=5000, path='/gw', id='default',
                 static_fountain=False, serverless=False, **kwargs):
        self.ted = ted
        self.agora = agora
        self.port = port
        self.path = path
        self.server_name = server_name
        proxy_fountain = Wrapper(self.agora.fountain) if static_fountain else self.agora.fountain
        self.__static_fountain = static_fountain
        self.proxy = Proxy(ted, proxy_fountain, server_name=server_name, server_port=port, path=path)
        self.cache = cache
        self.id = id
        # self.loader = self.proxy.load
        self.scholars = {}
        self.__sch_init_kwargs = kwargs.copy()

        if not serverless:
            self.server = bs(self.agora, query_function=self.query, import_name=__name__)
            bf(self.agora, server=self.server, fragment_function=self.fragment)
            bp(self.agora.planner, server=self.server)
            bn(self.agora.fountain, server=self.server)
            self.server = bpp(self.proxy, server=self.server, cache=cache)

    @property
    def loader(self):
        def wrapper(uri, format=None, **kwargs):
            result = self.proxy.load(uri, format)
            if result is None:
                for fmt in sorted(RDF_MIMES.keys(), key=lambda x: x != format):
                    result = http_get(uri, format=fmt)
                    if result is not None and not isinstance(result, bool):
                        content, headers = result
                        if not isinstance(content, Graph):
                            g = ConjunctiveGraph()
                            parse_rdf(g, content, fmt, headers)
                            result = g, headers
                        break
            return result

        return wrapper

    @property
    def static_fountain(self):
        return self.__static_fountain

    @property
    def interceptor(self):
        return self.proxy.interceptor

    @interceptor.setter
    def interceptor(self, i):
        self.proxy.interceptor = i

    def _scholar(self, force_seed, **kwargs):
        from agora.collector.scholar import Scholar

        scholar_id = 'd'
        required_params = self.proxy.parameters
        if required_params:
            m = hashlib.md5()
            for k in sorted(required_params):
                m.update(k + str(kwargs.get(k, '')))
            scholar_id = m.digest().encode('base64').strip()

        scholar_id = '/'.join([self.id, scholar_id])

        if scholar_id not in self.scholars.keys():
            scholar = Scholar(planner=self.agora.planner, cache=self.cache,
                              loader=self.loader, persist_mode=True,
                              id=scholar_id, force_seed=force_seed, **self.__sch_init_kwargs)
            self.scholars[scholar_id] = scholar

        return self.scholars[scholar_id]

    def query(self, query, stop_event=None, scholar=False, **kwargs):
        if self.interceptor:
            kwargs = self.interceptor(**kwargs)

        force_seed = self.proxy.instantiate_seeds(**kwargs)
        collector = self._scholar(force_seed, **kwargs) if scholar else None

        if not self.static_fountain:
            proxy_fountain = Wrapper(self.agora.fountain)
            query_proxy = Proxy(self.ted, proxy_fountain, server_name=self.server_name, server_port=self.port,
                                path=self.path)
        else:
            query_proxy = self.proxy

        return self.agora.query(query,
                                cache=self.cache,
                                loader=query_proxy.load,
                                stop_event=stop_event,
                                collector=collector,
                                force_seed=force_seed,
                                **kwargs)

    def fragment(self, query, stop_event=None, scholar=False, follow_cycles=True, **kwargs):
        if self.interceptor:
            kwargs = self.interceptor(**kwargs)

        force_seed = self.proxy.instantiate_seeds(**kwargs)
        collector = self._scholar(force_seed, **kwargs) if scholar else None

        if not self.static_fountain:
            proxy_fountain = Wrapper(self.agora.fountain)
            fragment_proxy = Proxy(self.ted, proxy_fountain, server_name=self.server_name, server_port=self.port,
                                   path=self.path)
        else:
            fragment_proxy = self.proxy

        return self.agora.fragment_generator(query,
                                             cache=self.cache,
                                             loader=fragment_proxy.load,
                                             stop_event=stop_event,
                                             collector=collector,
                                             force_seed=force_seed,
                                             follow_cycles=follow_cycles)

    def shutdown(self):
        for scholar in self.scholars.values():
            try:
                scholar.shutdown()

            except Exception as e:
                print e.message

        self.scholars.clear()
