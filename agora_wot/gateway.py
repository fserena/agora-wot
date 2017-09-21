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
import hashlib

from agora.collector.scholar import Scholar
from agora.server.fountain import build as bn
from agora.server.fragment import build as bf
from agora.server.planner import build as bp
from agora.server.sparql import build as bs
from agora_wot import Proxy

from agora_wot.publish import build as bpp

__author__ = 'Fernando Serena'


class Gateway(object):
    def __init__(self, agora, ted, cache=None, server_name='localhost', port=5000, path='/gw', id='default',
                 **kwargs):
        self.agora = agora
        self.proxy = Proxy(ted, self.agora.fountain, server_name=server_name, server_port=port, path=path)
        self.cache = cache
        self.id = id
        self.scholars = {}
        self.__sch_init_kwargs = kwargs.copy()

        self.server = bs(self.agora, query_function=self.query, import_name=__name__)
        bf(self.agora, server=self.server, fragment_function=self.fragment)
        bp(self.agora.planner, server=self.server)
        bn(self.agora.fountain, server=self.server)
        self.server = bpp(self.proxy, server=self.server)

    @property
    def interceptor(self):
        return self.proxy.interceptor

    @interceptor.setter
    def interceptor(self, i):
        self.proxy.interceptor = i

    def _scholar(self, **kwargs):
        self.proxy.clear_seeds()
        force_seeds = self.proxy.instantiate_seeds(**kwargs)

        scholar_id = 'd'
        required_params = self.proxy.parameters
        if required_params:
            m = hashlib.md5()
            for k in sorted(required_params):
                m.update(k + str(kwargs.get(k, '')))
            scholar_id = m.digest().encode('base64').strip()

        scholar_id = '/'.join([self.id, scholar_id])

        if scholar_id not in self.scholars.keys():
            scholar = Scholar(planner=self.agora.planner, cache=self.cache, path='fragments',
                              loader=self.proxy.load, persist_mode=True,
                              id=scholar_id, force_seed=force_seeds, **self.__sch_init_kwargs)
            self.scholars[scholar_id] = scholar

        return self.scholars[scholar_id]

    def query(self, query, stop_event=None, **kwargs):
        if self.interceptor:
            kwargs = self.interceptor(**kwargs)
        return self.agora.query(query, stop_event=stop_event, collector=self._scholar(**kwargs))

    def fragment(self, query, **kwargs):
        if self.interceptor:
            kwargs = self.interceptor(**kwargs)
        return self.agora.fragment_generator(query=query, collector=self._scholar(**kwargs))

    def shutdown(self):
        for scholar in self.scholars.values():
            try:
                scholar.index.clear()
                scholar.shutdown()
            except Exception as e:
                print e.message

        self.scholars.clear()
