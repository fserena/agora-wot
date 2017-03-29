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

from agora import Agora, setup_logging
from agora.collector.cache import RedisCache
from agora.collector.scholar import Scholar
from agora.engine.fountain.onto import DuplicateVocabulary
from agora.examples.movies import load_films_from_dbpedia
from datetime import datetime

from networkx import Graph
from rdflib import ConjunctiveGraph

__author__ = 'Fernando Serena'

# Setup logging level for Agora
setup_logging(logging.INFO)

# Agora object
agora = Agora()

# Open and add the vocabulary that we want to use to explore movies and associated data in dbpedia
with open('movies.ttl') as f:
    try:
        agora.fountain.add_vocabulary(f.read())
    except DuplicateVocabulary:
        pass

agora.fountain.add_seed('http://dbpedia.org/resource/Indiana_Jones_and_the_Last_Crusade', 'dbpedia-owl:Film')
agora.fountain.add_seed('http://dbpedia.org/resource/Braveheart', 'dbpedia-owl:Film')

query = """SELECT DISTINCT ?name ?actor WHERE { [] foaf:name ?name ;
                                            dbpedia-owl:starring [
                                            dbp:birthName ?actor ]
                                       }"""

g = ConjunctiveGraph()

for row in agora.query(query):
    print row

