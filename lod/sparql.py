# !/bin/env python #
# Ejemplo de consultas SPARQL sobre turtle #
# python consultas.py #
import rdflib
import sys

dataset = sys.argv[1] if len(sys.argv) > 1 else 'reviews.ttl'
g = rdflib.Graph()

schema = rdflib.Namespace("http://schema.org/")

# Read Turtle file #
g.parse(dataset, format='turtle')

results = g.query(
    """SELECT DISTINCT ?review ?p ?o
       WHERE {
          ?review a schema:Review.
          ?review ?p ?o.
       }""", initNs={'schema': schema})

for row in results:
    print("%s %s %s" % row)
