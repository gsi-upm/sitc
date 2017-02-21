import rdflib
import sys
g = rdflib.Graph()
dataset = sys.argv[1] if len(sys.argv) > 1 else 'reviews.ttl'
g.parse(dataset, format="n3")
print(g.serialize(format="n3").decode('utf-8'))
