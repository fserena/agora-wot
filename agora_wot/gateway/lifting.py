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
import logging
import traceback
from datetime import datetime
from email._parseaddr import mktime_tz
from email.utils import parsedate_tz

import isodate
import shortuuid
from agora.collector.http import extract_ttl
from agora.engine.plan.agp import extend_uri
from pyld import jsonld
from rdflib import URIRef, XSD, RDFS, Literal, BNode, Graph, RDF, OWL

from agora_wot.gateway.path import path_data
from agora_wot.utils import n3, get_ns, fltr, iriToUri, bound_graph

__author__ = 'Fernando Serena'

log = logging.getLogger('agora.wot.lifting')


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


def apply_mappings(data, mappings, ns):
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


def type_sub_tree(t, fountain, ns, t_dicts=None):
    if t_dicts is None:
        t_dicts = {}
    if t not in t_dicts:
        t_n3 = n3(t, ns)
        t_dicts[t_n3] = fountain.get_type(t_n3)
        for sub in t_dicts[t_n3]['sub']:
            type_sub_tree(sub, fountain, ns, t_dicts=t_dicts)
    return t_dicts


def type_hints(data, types, fountain, t_dicts=None):
    if t_dicts is None:
        t_dicts = {}

    ns = get_ns(fountain)
    for t in types:
        type_sub_tree(t, fountain, ns, t_dicts=t_dicts)

    t_matches = {t: reduce(lambda x, y: x + int(y in d['properties']), data, 0) for t, d in
                 t_dicts.items()}
    max_match = max(map(lambda x: t_matches[x], t_matches)) if t_matches else 0
    if not max_match and len(types) > 1:
        q_types = map(lambda t: t.n3(ns), types)
        hints = filter(lambda x: not set.intersection(set(t_dicts[x]['sub']), q_types), q_types)
        if not hints:
            raise TypeError('No hints')
        max_match = 1
    else:
        hints = filter(lambda x: t_matches[x] == max_match, t_dicts.keys())
    return hints, t_dicts


def _resource_graph(uri, prefixes):
    g = bound_graph(identifier=uri)
    for prefix, uri in prefixes.items():
        g.bind(prefix, uri)
    return g


class BlueprintLoader(object):
    def __init__(self, fountain, nodes, url_provider):
        self.fountain = fountain
        self.nodes = nodes
        self.ns = get_ns(fountain)
        self.url_for = url_provider
        self.resource_props = set()

    def load(self, resource, graph, uri, b64, r_args):
        prefixes = self.fountain.prefixes
        bnode_map = {}

        for s, p, o in resource.graph:
            if o in self.nodes:
                o = URIRef(self.url_for(tid=self.nodes[o], b64=b64, **r_args))
            elif isinstance(o, BNode):
                if o not in bnode_map:
                    bnode_map[o] = BNode()
                o = bnode_map[o]
            elif isinstance(o, Literal):
                if str(o) in r_args:
                    o = Literal(r_args[str(o)], datatype=o.datatype)

            if s == resource.node:
                s = uri

            if isinstance(s, BNode):
                if s not in self.nodes:
                    if s not in bnode_map:
                        bnode_map[s] = BNode()

                    for t in resource.graph.objects(s, RDF.type):
                        for supt in self.fountain.get_type(t.n3(self.ns))['super']:
                            graph.add((bnode_map[s], RDF.type, extend_uri(supt, prefixes)))

                    s = bnode_map[s]
                    graph.add((s, p, o))
            else:
                graph.add((s, p, o))

        for t in resource.types:
            if isinstance(t, URIRef):
                t_n3 = t.n3(self.ns)
            else:
                t_n3 = t
            type_dict = self.fountain.get_type(t_n3)
            self.resource_props.update(type_dict['properties'])
            for st in type_dict['super']:
                graph.add((uri, RDF.type, extend_uri(st, prefixes)))


class RDFSourceLoader(object):
    def __init__(self, sources, properties, ns):
        self.sources = sources
        self.properties = properties
        self.ns = ns

    def load(self, graph, uri):
        for e in self.sources:
            source_uri = URIRef(e.endpoint.href)
            graph.add((uri, OWL.sameAs, source_uri))
            same_as_g = Graph()
            same_as_g.load(source=source_uri)
            for s, p, o in same_as_g:
                if p.n3(self.ns) in self.properties:
                    if s == source_uri:
                        s = uri
                    elif not isinstance(s, BNode):
                        continue
                    graph.add((s, p, o))


class EndpointLoader(object):
    def __init__(self, graph, r_args):
        self.graph = graph
        self.r_args = r_args or {}
        self.endpoints_data = {}
        self.max_def_ttl = 1000000

    def load(self, endpoint):
        href = str(endpoint.href)
        if href not in self.endpoints_data:
            try:
                response = endpoint.invoke(graph=self.graph, subject=self.graph.identifier, **self.r_args)
                if response.status_code == 200:
                    result = {'content': response.json(), 'ttl': extract_ttl(response.headers) or self.max_def_ttl}
                elif response.status_code < 500 and response.status_code != 404:
                    result = {'content': {}, 'ttl': 10}
                else:
                    result = {'content': {}, 'ttl': self.max_def_ttl}
                self.endpoints_data[href] = result
            except AttributeError as e:
                log.debug('Missing attributes:' + e.message)
                raise
        return self.endpoints_data[href]['content']

    @property
    def ttl(self):
        ttls = map(lambda r: r['ttl'], self.endpoints_data.values())
        return reduce(lambda x, y: min(x, y), ttls, self.max_def_ttl)


class EndpointDataMapper(object):
    def __init__(self, ns):
        self.ns = ns

    def map(self, data, mappings):
        return apply_mappings(data, mappings, self.ns)


class DataLifter(object):
    def __init__(self, graph, fountain, url_provider):
        self.graph = graph
        self.fountain = fountain
        self.prefixes = self.fountain.prefixes
        self.ns = get_ns(fountain)
        self.type_dicts = {}
        self.property_dicts = {}
        self.url_for = url_provider
        self.n3_uri_map = {}

    def type_hints(self, data, types, target=None):
        hints, _ = type_hints(data, types, self.fountain, t_dicts=self.type_dicts)

        if target in types:
            hints = [target]
        else:
            common_types = filter(lambda x: not set.intersection(set(self.type_dicts[x]['super']), hints), hints)
            if len(common_types) > 1:
                hints = []
            elif len(common_types) == 1:
                hints = common_types
        return hints

    def triples(self, r_node, data, td, **r_args):
        vars = td.vars
        types = self.type_hints(data, td.resource.types, target=r_args.get('$target', None))
        if not types:
            return

        ld = self.enrich(r_node, data, types, vars, **r_args)
        ld_triples(ld, self.graph)

    def get_property_dict(self, p_n3):
        if p_n3 not in self.property_dicts:
            self.property_dicts[p_n3] = self.fountain.get_property(p_n3)
        return self.property_dicts[p_n3]

    def get_type_dict(self, t_n3):
        if t_n3 not in self.type_dicts:
            self.type_dicts[t_n3] = self.fountain.get_type(t_n3)
        return self.type_dicts[t_n3]

    def __uri(self, n3):
        if n3 not in self.n3_uri_map:
            self.n3_uri_map[n3] = extend_uri(n3, self.prefixes)
        return self.n3_uri_map[n3]

    def hint_enrich(self, r_node, data, types, vars, context=None, **r_args):
        types, _ = self.type_hints(data, types)
        if not types:
            return

        return self.enrich(r_node, data, types, vars, context=context, **r_args)

    def enrich(self, r_node, data, types, vars, context=None, **r_args):
        if context is None:
            context = {}

        j_types = []
        data['@id'] = r_node
        data['@type'] = j_types
        data_props = data.keys()

        for t_n3 in types:
            tdict = self.get_type_dict(t_n3)
            type_props = tdict['properties']
            short_type = t_n3.split(':')[1]
            type_uri = str(self.__uri(t_n3))
            context[short_type] = {'@id': type_uri, '@type': '@id'}
            j_types.append(short_type)

            props = filter(lambda x: str(x) in type_props, data_props)
            for p_n3 in props:
                p = self.__uri(p_n3)
                pdict = self.get_property_dict(p_n3)
                if pdict['type'] == 'data':
                    # return default data type (string) when not defined
                    range = pdict['range'][0] if pdict['range'] else 'xsd:string'
                    if range == 'rdfs:Resource':
                        datatype = Literal(data[p_n3]).datatype
                    else:
                        datatype = self.__uri(range)
                    jp = {'@type': datatype, '@id': p}
                else:
                    jp = {'@type': '@id', '@id': p}

                context[p_n3] = jp
                p_n3_data = data[p_n3]
                if isinstance(p_n3_data, dict):
                    sub = self.hint_enrich(
                        r_node=BNode(shortuuid.uuid()).n3(self.ns),
                        data=p_n3_data,
                        types=pdict['range'],
                        vars=vars,
                        contex=context,
                        **r_args)

                    data[p_n3] = sub['@graph']
                elif hasattr(p_n3_data, '__call__'):
                    data[p_n3] = p_n3_data(key=p_n3, context=context,
                                           uri_provider=self.url_for,
                                           vars=vars,
                                           **r_args)
                elif isinstance(p_n3_data, list):
                    p_items_res = []
                    data[p_n3] = p_items_res
                    for p_item in p_n3_data:
                        if hasattr(p_item, '__call__'):
                            p_items_res.extend(
                                p_item(key=p_n3, context=context,
                                       uri_provider=self.url_for, vars=vars,
                                       **r_args))
                        elif pdict['type'] != 'data':
                            if isinstance(p_item, basestring):
                                p_items_res.append(URIRef(iriToUri(p_item)))
                            else:
                                sub = self.hint_enrich(
                                    r_node=BNode(shortuuid.uuid()).n3(self.ns),
                                    data=p_item,
                                    types=pdict['range'],
                                    vars=vars,
                                    context=context,
                                    **r_args)
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
                            datatype = self.__uri(range)
                        context[p_n3] = {'@type': datatype, '@id': p}
        return {'@context': context, '@graph': data}


class Lifter(object):
    def __init__(self, ted, wrapper, url_provider):
        self.ted = ted
        self.wrapper = wrapper
        self.url_for = url_provider

        self.__rdict = {t.id: t for t in ted.ecosystem.tds}
        self.__ndict = {t.resource.node: t.id for t in ted.ecosystem.tds}
        self.__network = ted.ecosystem.network()

    def __get_td(self, tid):
        return self.__rdict.get(tid, tid)

    def __compose_endpoints(self, resource):
        id = resource.id
        for base_e in resource.base:
            if base_e.href is None:
                for pred in self.__network.predecessors(id):
                    pred_thing = self.__rdict[pred]
                    for pred_e in self.__compose_endpoints(pred_thing):
                        yield pred_e + base_e
            else:
                yield base_e

    def lift(self, tid, fountain, r_uri, b64=None, r_args=None, base_graph=None, replace=False):
        prefixes = fountain.prefixes
        g = _resource_graph(r_uri, prefixes)
        if base_graph:
            g.__iadd__(base_graph)
        ttl = 10

        if r_args is None:
            r_args = {}

        try:
            td = self.__get_td(tid)
            bp_loader = BlueprintLoader(fountain, self.__ndict, self.url_for)
            bp_loader.load(td.resource, g, r_uri, b64, r_args)

            rdf_loader = RDFSourceLoader(td.rdf_sources, bp_loader.resource_props, bp_loader.ns)
            rdf_loader.load(g, r_uri)

            endpoint_loader = EndpointLoader(g, r_args)
            data_mapper = EndpointDataMapper(bp_loader.ns)
            data_lifter = DataLifter(g, fountain, self.url_for)

            if td.base:
                endpoints = list(self.__compose_endpoints(td))
                endpoints_order = {am.endpoint: am.order for am in td.access_mappings}
                for e in sorted(endpoints, key=lambda x: endpoints_order[x]):
                    try:
                        e_data = endpoint_loader.load(e)
                        mappings = td.endpoint_mappings(e)
                        mapped_data = data_mapper.map(e_data, mappings)
                        data_lifter.triples(r_uri, mapped_data, td, **r_args)
                    except AttributeError:
                        continue

            ttl = endpoint_loader.ttl
        except Exception as e:
            traceback.print_exc()
            log.warn(r_uri + ': {}'.format(e.message))

        if replace and base_graph:
            diff_g = g - base_graph
            for s, p, o in diff_g:
                base_rep_triples = base_graph.triples((s, p, None))
                for srem, prem, orem in base_rep_triples:
                    g.remove((srem, prem, orem))

        return g, ttl
