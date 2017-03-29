# coding=utf-8
import logging
import urllib
from StringIO import StringIO

from agora.examples.movies import load_films_from_dbpedia
from rdflib import Graph

import agora.examples
from agora import Agora, setup_logging
from agora import RedisCache
from agora.collector.scholar import Scholar
from agora.server.fountain import build as bn
from agora.server.fragment import build as bf
from agora.server.planner import build as bp
from agora.server.sparql import build as bs
from agora.ted import Proxy
from agora.ted import TED
from agora.ted.blocks import Endpoint, Mapping, AccessMapping, Resource, TD, RDFSource
from agora.ted.publish import build as bg

setup_logging(logging.DEBUG)

with open('ted.ttl') as f:
    ted_str = f.read()

g = Graph()
# g.parse(StringIO(ted_str), format='turtle')

cache = RedisCache(min_cache_time=10, persist_mode=True, path='cache', redis_file='store/cache/cache.db')

# Agora object
agora = Agora(persist_mode=True, redis_file='store/fountain/fountain.db', path='fountain')
with open('movies.ttl') as f:
    try:
        agora.fountain.add_vocabulary(f.read())
    except:
        pass

ted = TED()

# Build TED
e = Endpoint(href="http://www.omdbapi.com/?t={{object(foaf:name)}}&y=&plot=short&r=json",
             media="application/json")

am = AccessMapping(endpoint=e)
am.mappings.add(Mapping(key="imdbRating", predicate='http://agora.org/amovies#rating'))
am.mappings.add(Mapping(key="imdbVotes", predicate='http://agora.org/amovies#votes'))

# resource = Resource(types=['http://dbpedia.org/ontology/Film'])
# td = TD(resource, id='2046')
# td.add_access_mapping(am)
# e = Endpoint(href="http://dbpedia.org/resource/2046_(film)",
#              media="text/turtle")
# rdf_source = RDFSource(e)
# td.add_rdf_source(rdf_source)
# ted.ecosystem.add_component_from_td(td)
#
# td2 = TD.from_types(types=['http://dbpedia.org/ontology/Film'], id='rogue')
# rdf_source2 = RDFSource.from_uri('http://dbpedia.org/resource/Rogue_One')
# td2.add_rdf_source(rdf_source2)
# td2.add_access_mapping(am)
# ted.ecosystem.add_component_from_td(td2)


# Each film URI found in dbpedia is added to Agora as a seed
for film, name in load_films_from_dbpedia():
    try:
        td = TD.from_types(types=['http://dbpedia.org/ontology/Film'], id=urllib.quote_plus(name))
        rdf_source = RDFSource.from_uri(unicode(film))
        td.add_rdf_source(rdf_source)
        td.add_access_mapping(am)
        ted.ecosystem.add_component_from_td(td)
    except Exception:
        pass

proxy = Proxy(ted, agora.fountain, server_name='localhost', server_port=5000, path='/proxy')

scholar = Scholar(agora.planner, cache=cache, loader=proxy.load)


def query(query, **kwargs):
    return agora.query(query, collector=scholar, **kwargs)


def fragment(**kwargs):
    return agora.fragment_generator(collector=scholar, **kwargs)


server = bs(agora, query_function=query, import_name=__name__)
bf(agora, server=server, fragment_function=fragment)
bp(agora.planner, server=server)
bn(agora.fountain, server=server)
bg(proxy, server=server)

agora.fountain.delete_type_seeds('dbpedia-owl:Film')
for uri, type in proxy.seeds:
    try:
        agora.fountain.add_seed(uri, type)
    except:
        pass

if __name__ == '__main__':
    server.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False, threaded=True)
