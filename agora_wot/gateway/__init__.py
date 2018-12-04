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
import hashlib
from abc import abstractmethod

from agora.collector.execution import filter_resource
from agora.engine.plan.agp import extend_uri
from agora.engine.utils import Wrapper, Semaphore
from agora.server.fountain import build as bn
from agora.server.fragment import build as bf
from agora.server.planner import build as bp
from agora.server.sparql import build as bs
from rdflib import Graph, URIRef

from agora_wot.gateway.proxy import Proxy
from agora_wot.gateway.publish import build as bpp

__author__ = 'Fernando Serena'


class AbstractDataGateway(object):
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
        # type: () -> None
        raise NotImplementedError


class DataGateway(AbstractDataGateway):
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
        self.scholars = {}
        self.__sch_init_kwargs = kwargs.copy()
        self.__stop = Semaphore()

        if not serverless:
            self.server = bs(self.agora, query_function=self.query, import_name=__name__)
            bf(self.agora, server=self.server, fragment_function=self.fragment)
            bp(self.agora.planner, server=self.server)
            bn(self.agora.fountain, server=self.server)
            self.server = bpp(self.proxy, server=self.server, cache=cache)

    @property
    def loader(self):
        def wrapper(uri, format=None, **kwargs):
            return self.proxy.load(uri, format)

        return wrapper

    def load(self, uri, format=None, filter=False, **kwargs):
        res_g, headers = self.loader(uri, format=format, **kwargs)

        if filter:
            prefixes = self.agora.fountain.prefixes
            fg = Graph(identifier=uri)
            for prefix, u in prefixes.items():
                fg.bind(prefix, u)
            filter_types = map(lambda x: extend_uri(x, prefixes), self.agora.fountain.types)
            properties = self.agora.fountain.properties
            filter_predicates = map(lambda x: extend_uri(x, prefixes), properties)
            inverses = map(lambda x: set(self.agora.fountain.get_property(x)['inverse']), properties)
            inverses = reduce(lambda x, y: y.union(x), inverses, set())
            inverses = map(lambda x: extend_uri(x, prefixes), inverses)
            filter_resource(URIRef(uri), res_g, fg, types=filter_types or [], predicates=filter_predicates or [],
                            inverses=inverses or [])
            res_g = fg
        return res_g, headers

    @property
    def static_fountain(self):
        return self.__static_fountain

    @property
    def interceptor(self):
        return self.proxy.interceptor

    @interceptor.setter
    def interceptor(self, i):
        self.proxy.interceptor = i

    def seeds(self, **kwargs):
        seeds = self.proxy.instantiate_seeds(**kwargs)
        seeds = set(reduce(lambda x, y: x + y, seeds.values(), []))
        return seeds

    def typed_seeds(self, **kwargs):
        seeds = self.proxy.instantiate_seeds(**kwargs)
        for t, uris in seeds.items():
            for uri in uris:
                yield (uri, t)

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

        if stop_event is None:
            stop_event = self.__stop or Semaphore()

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

        if stop_event is None:
            stop_event = self.__stop or Semaphore()

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

    def __enter__(self):
        self.__stop.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()
        self.__stop.__exit__(exc_type, exc_val, exc_tb)
