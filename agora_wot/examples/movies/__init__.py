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
from SPARQLWrapper import JSON
from SPARQLWrapper import SPARQLWrapper

__author__ = 'Fernando Serena'


def load_films_from_dbpedia():
    """
    Get movie resources from dbpedia
    :return: movies generator
    """
    sparql = SPARQLWrapper("http://es.dbpedia.org/sparql")
    sparql.setReturnFormat(JSON)

    sparql.setQuery("""
           SELECT distinct ?film ?name
           WHERE {?film a dbpedia-owl:Film ; foaf:name ?name} LIMIT 100 OFFSET 0
       """)
    results = sparql.query().convert()

    for result in results["results"]["bindings"]:
        yield (result["film"]["value"], result["name"]["value"])
