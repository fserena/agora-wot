import logging
from StringIO import StringIO
from datetime import datetime
from time import sleep

from rdflib import Graph

from agora import Agora, setup_logging
import agora.examples
from agora import RedisCache
from agora.collector.scholar import Scholar
from agora.ted import TED, Gateway

setup_logging(logging.DEBUG)

with open('ted.ttl') as f:
    ted_str = f.read()

g = Graph()
g.parse(StringIO(ted_str), format='turtle')

cache = RedisCache(min_cache_time=10, persist_mode=True, path='cache', redis_file='store/cache/cache.db')

# Agora object
agora = Agora(persist_mode=True, redis_file='store/fountain/fountain.db', path='fountain')
with open('librairy.ttl') as f:
    try:
        agora.fountain.add_vocabulary(f.read())
    except:
        pass

ted = TED(g)
gw = Gateway(ted, agora.fountain)


agora.fountain.delete_type_seeds('librairy:DocumentService')
agora.fountain.delete_type_seeds('librairy:TopicService')
for uri, type in gw.seeds:
    try:
        agora.fountain.add_seed(uri, type)
    except:
        pass

q10 = """SELECT DISTINCT ?title ?created WHERE {
                                ?s librairy:title ?title ;
                                   librairy:creationTime ?created
                                FILTER(STR(?title)="High-order similarity relations in radiative transfer")
 } LIMIT 1"""


q11 = """SELECT DISTINCT ?topic ?content ?created WHERE {
                    ?topic librairy:content ?content ;
                           librairy:creationTime ?created
 }"""

q12 = """SELECT DISTINCT ?content ?title WHERE {
                    [] librairy:title ?title ;
                       librairy:containsTopic [
                         librairy:content ?content
                       ]
 }"""


scholar = Scholar(agora.planner, cache=cache, loader=gw.load)

elapsed = []

for query in [q12]:
    pre = datetime.now()
    n = 0
    for row in agora.query(query, collector=scholar, incremental=False):
        print '[', (datetime.now() - pre).total_seconds(), '] solution:',
        for label in row.labels:
            print label + '=' + str(row[label]),
        print
        n += 1
    print n, 'solutions'
    post = datetime.now()
    elapsed.append((post - pre).total_seconds())
    # sleep(0.5)

print elapsed

raw_input()
