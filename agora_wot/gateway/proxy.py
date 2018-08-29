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
import base64
import copy
import logging
import traceback
import urlparse
from datetime import datetime
from email._parseaddr import mktime_tz
from email.utils import parsedate_tz
from urllib import quote

import isodate
import shortuuid
from agora.collector.http import extract_ttl
from agora.collector.wrapper import ResourceWrapper
from agora.engine.fountain import AbstractFountain
from agora.engine.plan.agp import extend_uri
from agora.engine.utils import Wrapper
from pyld import jsonld
from rdflib import ConjunctiveGraph
from rdflib import Graph
from rdflib import OWL
from rdflib import RDF
from rdflib import RDFS
from rdflib import XSD
from rdflib.term import Literal, URIRef, BNode

from agora_wot.blocks.td import TD
from agora_wot.blocks.ted import TED
from agora_wot.blocks.utils import encode_rdict
from agora_wot.gateway.path import parse

__author__ = 'Fernando Serena'

log = logging.getLogger('agora.wot')


def fltr(node, prefixes):
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


def get_ns(fountain):
    g = Graph()
    prefixes = fountain.prefixes
    for prefix, ns in prefixes.items():
        g.bind(prefix, ns)
    return g.namespace_manager


def path_data(path, data):
    if path:
        try:
            jsonpath_expr = parse(path)
            p_data = [match.value for match in jsonpath_expr.find(data)]
            if p_data:
                if len(p_data) == 1:
                    return p_data.pop()
                else:
                    return p_data
        except:
            pass

    return None


def iriToUri(iri):
    parts = urlparse.urlparse(iri)
    return urlparse.urlunparse(
        part.encode('utf-8') if parti == 1 else quote(part.encode('utf-8'))
        for parti, part in enumerate(parts)
    )


def apply_mappings(data, mappings, ns):
    def apply_mapping(md, mapping, p_n3):
        if isinstance(md, dict):
            data_keys = list(md.keys())
            if mapping.path:
                data_keys = filter(lambda x: x == mapping.key, data_keys)
            for k in data_keys:
                next_k = k
                if k == mapping.key:
                    mapped_k = p_n3
                    next_k = mapped_k
                    mapped_v = md[k]
                    if mapped_v is None:
                        continue
                    if isinstance(mapped_v, list) and mapping.limit:
                        mapped_v = mapped_v[:1]
                    if mapping.transform is not None:
                        mapped_v = mapping.transform.attach(md[k])
                    if mapped_k not in md:
                        md[mapped_k] = mapped_v
                    elif md[mapped_k] != mapped_v:
                        if not isinstance(md[mapped_k], list):
                            md[mapped_k] = [md[mapped_k]]
                        mapped_lv = md[mapped_k]
                        if isinstance(mapped_v, list):
                            for mv in mapped_v:
                                if mv not in mapped_lv:
                                    mapped_lv.append(mv)
                        elif mapped_v not in mapped_lv:
                            mapped_lv.append(mapped_v)
                if not mapping.path:
                    apply_mapping(md[next_k], mapping, p_n3)
        elif isinstance(md, list):
            [apply_mapping(elm, mapping, p_n3) for elm in md]

    container = any(filter(lambda x: x.key == '$container', mappings))
    if container and not isinstance(data, dict):
        data = {m.key: data for m in mappings if m.key == '$container'}

    for m in sorted(mappings, key=lambda x: x.path, reverse=True):
        p_n3 = m.predicate.n3(ns)
        m_data = data
        if m.path is not None:
            p_data = path_data(m.path, data)
            if not p_data:
                continue
            m_data = p_data

        apply_mapping(m_data, m, p_n3)
        if m.root:
            p_n3 = m.predicate.n3(ns)
            if isinstance(m_data, dict):
                m_data = [m_data]
            m_data = filter(lambda x: x, map(lambda x: x.get(p_n3, None) if isinstance(x, dict) else x, m_data))

            # CHECK THIS!!!
            if isinstance(data, list):
                data = {'$container': data}
            if p_n3 not in data:
                data[p_n3] = m_data
            else:
                if not isinstance(data[p_n3], list):
                    data[p_n3] = [data[p_n3]]
                for m in m_data:
                    if m not in data[p_n3]:
                        data[p_n3].append(m)

    data = fltr(data, dict(list(ns.namespaces())).keys())
    if data is None:
        data = {}
    return data


def ld_triples(ld, g=None):
    bid_map = {}

    def parse_term(term):
        try:
            term['value'] = term['value'].decode('unicode-escape')
        except UnicodeEncodeError:
            pass

        if term['type'] == 'IRI':
            return URIRef(u'{}'.format(term['value']))
        elif term['type'] == 'literal':
            datatype = URIRef(term.get('datatype', None))
            if datatype == XSD.dateTime:
                try:
                    term['value'] = float(term['value'])
                    term['value'] = datetime.utcfromtimestamp(term['value'])
                except:
                    try:
                        term['value'] = isodate.parse_datetime(term['value'])
                    except:
                        timestamp = mktime_tz(parsedate_tz(term['value']))
                        term['value'] = datetime.fromtimestamp(timestamp)
            if datatype == RDFS.Literal:
                datatype = None
                try:
                    term['value'] = float(term['value'])
                except:
                    pass
            return Literal(term['value'], datatype=datatype)
        else:
            bid = term['value'].split(':')[1]
            if bid not in bid_map:
                bid_map[bid] = shortuuid.uuid()
            return BNode(bid_map[bid])

    if g is None:
        g = Graph()
        if '@context' in ld:
            for ns in filter(
                    lambda k: ':' not in k and isinstance(ld['@context'][k], basestring) and ld['@context'][
                        k].startswith('http'), ld['@context']):
                g.bind(ns, URIRef(ld['@context'].get(ns)))

    if ld:
        norm = jsonld.normalize(ld)
        def_graph = norm.get('@default', [])
        for triple in def_graph:
            predicate = parse_term(triple['predicate'])
            if not predicate.startswith('http'):
                continue
            subject = parse_term(triple['subject'])
            object = parse_term(triple['object'])
            g.add((subject, predicate, object))
    else:
        print ld

    return g


class Proxy(object):
    def __init__(self, ted, fountain, server_name='proxy', url_scheme='http', server_port=None, path=''):
        # type: (TED) -> None
        self.__ted = ted
        self.__fountain = fountain
        self.__seeds = set([])
        self.__wrapper = ResourceWrapper(server_name=server_name, url_scheme=url_scheme, server_port=server_port,
                                         path=path)
        self.__rdict = {t.id: t for t in ted.ecosystem.tds}
        self.__ndict = {t.resource.node: t.id for t in ted.ecosystem.tds}

        self.__wrapper.intercept('{}/<tid>'.format(path))(self.describe_resource)
        self.__wrapper.intercept('{}/<tid>/<b64>'.format(path))(self.describe_resource)
        self.__network = self.__ted.ecosystem.network()
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

    def process_arguments(self, **kwargs):
        return {k: v.pop() if isinstance(v, list) else v for k, v in kwargs.items()}

    def __remove_type_redundancy(self, fountain, types, ns):
        known_types = fountain.types
        n3_types = {t.n3(ns): t for t in types}
        n3_filtered = filter(
            lambda t: t in known_types and not set.intersection(set(fountain.get_type(t)['sub']),
                                                                n3_types.keys()),
            n3_types.keys())
        return map(lambda t: n3_types.get(t), n3_filtered)

    def instantiate_seeds(self, **kwargs):
        seeds = {}
        kwargs = self.process_arguments(**kwargs)
        if self.interceptor:
            kwargs = self.interceptor(**kwargs)

        fountain = self.fountain
        ns = get_ns(fountain)

        root_types = reduce(lambda x, y: x.union(self.__remove_type_redundancy(fountain, y.types, ns)),
                            self.ecosystem.root_resources,
                            set())
        n3_root_types = {t.n3(ns): t for t in root_types}
        for ty in root_types:
            for td in self.ecosystem.tds_by_type(ty):
                try:
                    var_params = set([v.lstrip('$') for v in td.vars])
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
        return self.__wrapper.load(uri, **kwargs)

    def compose_endpoints(self, resource):
        id = resource.id
        for base_e in resource.base:
            if base_e.href is None:
                for pred in self.__network.predecessors(id):
                    pred_thing = self.__rdict[pred]
                    for pred_e in self.compose_endpoints(pred_thing):
                        yield pred_e + base_e
            else:
                yield base_e

    def describe_resource(self, tid, b64=None, **kwargs):
        td = self.__rdict[tid]

        fountain = self.fountain
        ns = get_ns(fountain)
        prefixes = fountain.prefixes
        g = ConjunctiveGraph()

        ttl = 100000
        try:
            if b64 is not None:
                b64 = b64.replace('%3D', '=')
                resource_args = eval(base64.b64decode(b64))
            else:
                resource_args = kwargs
            r_uri = self.url_for(tid=tid, b64=b64)
            if kwargs:
                r_uri = '{}?{}'.format(r_uri, '&'.join(['{}={}'.format(k, kwargs[k]) for k in kwargs]))
            r_uri = URIRef(r_uri)

            g = ConjunctiveGraph(identifier=r_uri)
            for prefix, uri in prefixes.items():
                g.bind(prefix, uri)

            bnode_map = {}

            for s, p, o in td.resource.graph:
                if o in self.__ndict:
                    o = URIRef(self.url_for(tid=self.__ndict[o], b64=b64, **resource_args))
                elif isinstance(o, BNode):
                    if o not in bnode_map:
                        bnode_map[o] = BNode()
                    o = bnode_map[o]
                elif isinstance(o, Literal):
                    if str(o) in resource_args:
                        o = Literal(resource_args[str(o)], datatype=o.datatype)

                if s == td.resource.node:
                    s = r_uri

                if isinstance(s, BNode):
                    if s not in self.__ndict:
                        if s not in bnode_map:
                            bnode_map[s] = BNode()

                        for t in td.resource.graph.objects(s, RDF.type):
                            for supt in fountain.get_type(t.n3(ns))['super']:
                                g.add((bnode_map[s], RDF.type, extend_uri(supt, prefixes)))

                        s = bnode_map[s]
                        g.add((s, p, o))
                else:
                    g.add((s, p, o))

            resource_props = set([])
            for t in td.resource.types:
                if isinstance(t, URIRef):
                    t_n3 = t.n3(ns)
                else:
                    t_n3 = t
                type_dict = fountain.get_type(t_n3)
                resource_props.update(type_dict['properties'])
                for st in type_dict['super']:
                    g.add((r_uri, RDF.type, extend_uri(st, prefixes)))

            if td.rdf_sources:
                for e in td.rdf_sources:
                    uri = URIRef(e.endpoint.href)
                    g.add((r_uri, OWL.sameAs, uri))
                    same_as_g = Graph()
                    same_as_g.load(source=uri)
                    for s, p, o in same_as_g:
                        if p.n3(ns) in resource_props:
                            if s == uri:
                                s = r_uri
                            elif not isinstance(s, BNode):
                                continue
                            g.add((s, p, o))

            if td.base:
                invoked_endpoints = {}
                endpoints = list(self.compose_endpoints(td))
                endpoints_order = {am.endpoint: am.order for am in td.access_mappings}
                for e in sorted(endpoints, key=lambda x: endpoints_order[x]):
                    if str(e.href) not in invoked_endpoints:
                        try:
                            invoked_endpoints[str(e.href)] = e.invoke(graph=g, subject=r_uri, **resource_args)
                        except AttributeError as e:
                            log.debug('Missing attributes:' + e.message)
                            continue

                    response = invoked_endpoints[str(e.href)]
                    if response.status_code == 200:
                        data = response.json()
                        e_mappings = td.endpoint_mappings(e)
                        mapped_data = apply_mappings(data, e_mappings, ns)
                        ld = self.enrich(r_uri, mapped_data, td.resource.types,
                                         fountain, ns=ns, vars=td.vars, **resource_args)
                        ld_triples(ld, g)
                        ttl = min(ttl, extract_ttl(response.headers) or ttl)
                    elif response.status_code < 500:
                        ttl = 10

        except Exception as e:
            # traceback.print_exc()
            ttl = 10
            log.warn(r_uri + ': {}'.format(e.message))
        return g, {'Cache-Control': 'max-age={}'.format(ttl)}

    def clear_seeds(self):
        self.__seeds.clear()
        for root in self.ecosystem.roots:
            resource = root.resource if isinstance(root, TD) else root
            for t in resource.types:
                t_n3 = t.n3(self.namespace_manager)
                self.fountain.delete_type_seeds(t_n3)

    def url_for(self, tid, b64=None, **kwargs):
        if tid in self.__ndict:
            tid = self.__ndict[tid]
        # if b64 is None and kwargs:
        if kwargs:
            var_dict = {}
            for var in filter(lambda x: x in kwargs, self.__rdict[tid].vars):
                var_dict[var] = kwargs[var]
            b64 = encode_rdict(var_dict)
        return self.__wrapper.url_for('describe_resource', tid=tid, b64=b64)

    def n3(self, t, ns):
        if isinstance(t, URIRef):
            t_n3 = t.n3(ns)
        else:
            t_n3 = t
        return t_n3

    def type_sub_tree(self, t, fountain, ns, t_dicts=None):
        if t_dicts is None:
            t_dicts = {}
        if t not in t_dicts:
            t_n3 = self.n3(t, ns)
            t_dicts[t_n3] = fountain.get_type(t_n3)
            for sub in t_dicts[t_n3]['sub']:
                self.type_sub_tree(sub, fountain, ns, t_dicts=t_dicts)
        return t_dicts

    def enrich(self, uri, data, types, fountain, ns=None, context=None, vars=None, t_dicts=None, p_dicts=None,
               **kwargs):
        # type: (URIRef, dict, list, AbstractFountain) -> any

        if context is None:
            context = {}

        if t_dicts is None:
            t_dicts = {}

        if p_dicts is None:
            p_dicts = {}

        if vars is None:
            vars = set([])

        if ns is None:
            ns = get_ns(fountain)

        for t in types:
            self.type_sub_tree(t, fountain, ns, t_dicts=t_dicts)

        t_matches = {t: reduce(lambda x, y: x + int(y in d['properties']), data, 0) for t, d in
                     t_dicts.items()}
        max_match = max(map(lambda x: t_matches[x], t_matches)) if t_matches else 0
        if not max_match and len(types) > 1:
            q_types = map(lambda t: t.n3(ns), types)
            types = filter(lambda x: not set.intersection(set(t_dicts[x]['sub']), q_types), q_types)
            if not types:
                return
            max_match = 1
        else:
            types = filter(lambda x: t_matches[x] == max_match, t_dicts.keys())

        target = kwargs.get('$target', None)
        if target in types:
            types = [target]
        else:
            common_types = filter(lambda x: not set.intersection(set(t_dicts[x]['super']), types), types)
            if len(common_types) > 1:
                return
            elif len(common_types) == 1:
                types = common_types

        j_types = []
        data['@id'] = uri
        data['@type'] = j_types
        prefixes = dict(ns.graph.namespaces())
        for t_n3 in types:
            props = t_dicts[t_n3]['properties']
            short_type = t_n3.split(':')[1]
            context[short_type] = {'@id': str(extend_uri(t_n3, prefixes)), '@type': '@id'}
            j_types.append(short_type)
            for p_n3 in data:
                if p_n3 in props:
                    p = extend_uri(p_n3, prefixes)
                    if p_n3 not in p_dicts:
                        p_dicts[p_n3] = fountain.get_property(p_n3)
                    pdict = p_dicts[p_n3]
                    if pdict['type'] == 'data':
                        # return default data type (string) when not defined
                        range = pdict['range'][0] if pdict['range'] else 'xsd:string'
                        if range == 'rdfs:Resource':
                            datatype = Literal(data[p_n3]).datatype
                        else:
                            datatype = extend_uri(range, prefixes)
                        jp = {'@type': datatype, '@id': p}
                    else:
                        jp = {'@type': '@id', '@id': p}

                    context[p_n3] = jp
                    p_n3_data = data[p_n3]
                    if isinstance(p_n3_data, dict):
                        sub = self.enrich(BNode(shortuuid.uuid()).n3(ns), p_n3_data, pdict['range'], fountain, ns,
                                          context, vars=vars, t_dicts=t_dicts, p_dicts=p_dicts, **kwargs)
                        data[p_n3] = sub['@graph']
                    elif hasattr(p_n3_data, '__call__'):
                        data[p_n3] = p_n3_data(key=p_n3, context=context, uri_provider=self.url_for, vars=vars,
                                               **kwargs)
                    elif isinstance(p_n3_data, list):
                        p_items_res = []
                        data[p_n3] = p_items_res
                        for p_item in p_n3_data:
                            if hasattr(p_item, '__call__'):
                                p_items_res.extend(
                                    p_item(key=p_n3, context=context, uri_provider=self.url_for, vars=vars,
                                           **kwargs))
                            elif pdict['type'] != 'data':
                                if isinstance(p_item, basestring):
                                    p_items_res.append(URIRef(iriToUri(p_item)))
                                else:
                                    sub = self.enrich(BNode(shortuuid.uuid()).n3(ns), p_item, pdict['range'],
                                                      fountain,
                                                      ns=ns, p_dicts=p_dicts,
                                                      context=context, vars=vars, **kwargs)
                                    if sub:
                                        p_items_res.append(sub['@graph'])
                            else:
                                p_items_res.append(p_item)
                    elif pdict['type'] == 'object':
                        try:
                            data[p_n3] = URIRef(iriToUri(p_n3_data))
                        except AttributeError:
                            if 'rdfs:Resource' in pdict['range']:
                                datatype = Literal(data[p_n3]).datatype
                            else:
                                range = pdict['range'][0] if pdict['range'] else 'xsd:string'
                                datatype = extend_uri(range, prefixes)
                            context[p_n3] = {'@type': datatype, '@id': p}

        return {'@context': context, '@graph': data}
