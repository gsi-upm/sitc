
import sys
from future.standard_library import install_aliases
install_aliases()

from urllib import request, parse
from rdflib import Graph, term
from lxml import etree

if len(sys.argv) < 2:
    print('Usage: python {} <URL>'.format(sys.argv[0]))
    print('')
    print('Extract rdfa, microdata and json-ld annotations from a website')
    exit(1)

url = sys.argv[1]

g = Graph()
g.parse(url, format='rdfa')
g.parse(url, format='microdata')


def sanitize_triple(t):
    """Function to remove bad URIs from the graph that would otherwise
    make the serialization fail."""
    def sanitize_triple_item(item):
        if isinstance(item, term.URIRef) and '/' not in item:
            return term.URIRef(parse.quote(str(item)))
        return item

    return (sanitize_triple_item(t[0]),
            sanitize_triple_item(t[1]),
            sanitize_triple_item(t[2]))


with request.urlopen(url) as response:
    # Get all json-ld objects embedded in the html file
    html = response.read().decode('utf-8', errors='ignore')
    parser = etree.XMLParser(recover=True)
    root = etree.fromstring(html, parser=parser)
    if root:
        for jsonld in root.findall(".//script[@type='application/ld+json']"):
            g.parse(data=jsonld.text, publicID=url, format='json-ld')


fixedgraph = Graph()
fixedgraph += [sanitize_triple(s) for s in g]

print(g.serialize(format='turtle').decode('utf-8', errors='ignore'))
