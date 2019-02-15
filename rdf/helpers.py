import sys
from future.standard_library import install_aliases, print_function
install_aliases()

from urllib import request, parse
from rdflib import Graph, term, Namespace, BNode
from lxml import etree

import IPython
js = "IPython.CodeCell.options_default.highlight_modes['magic_turtle'] = {'reg':[/^%%ttl/]};"
IPython.core.display.display_javascript(js, raw=True)


from IPython.core.magic import (register_line_magic, register_cell_magic,
                                register_line_cell_magic)
from IPython.display import HTML, display, Image, Markdown


DEFINITIONS = {}

def solution(exercise='default'):
    if exercise not in DEFINITIONS:
        raise Exception('Solution for {} not found. Have you defined it?'.format(exercise))
    return DEFINITIONS[exercise]


@register_cell_magic
def ttl(line, cell):
    '''
    TTL magic command for ipython. It can be used in a cell like this:
    
    ```
    %%ttl
    
    ... Your TTL definition ...
    
    ```
    The definition will be loaded into a DEFINITION variable, using RDFlib.
    This definition can then be used for evaluation.
    '''
    g = Graph()
    msg = '''Error on line {line}

Reason: {reason}

If you don\'t know what this error means, try an online validator: http://ttl.summerofcode.be/
'''
    global DEFINITIONS
    key = line or 'default'
    DEFINITIONS[key] = None
    try:
        DEFINITIONS[key] = g.parse(data=cell,
                                  format="n3")
    except SyntaxError as ex:
        print(msg.format(line=ex.lines, reason=ex._why), file=sys.stderr)
        raise Exception('Bad Turtle syntax') from None
    except Exception as ex:
        print(msg.format(line='?', reason=ex), file=sys.stderr)
        raise Exception('Bad Turtle syntax') from None
    return Markdown('The Turtle syntax is correct.')


def extract_data(url):
    g = Graph()
    try:
        g.parse(url, format='rdfa')
    except Exception:
        print('Could not get rdfa data', file=sys.stderr)
    try:
        g.parse(url, format='microdata')
    except Exception:
        print('Could not get microdata', file=sys.stderr)


    def sanitize_triple(t):
        """Function to remove bad URIs from the graph that would otherwise
        make the serialization fail."""
        def sanitize_triple_item(item):
            if isinstance(item, term.URIRef) and ' ' in item:
                return term.URIRef(parse.quote(str(item)))
            return item

        return (sanitize_triple_item(t[0]),
                sanitize_triple_item(t[1]),
                sanitize_triple_item(t[2]))


    with request.urlopen(url) as response:
        # Get all json-ld objects embedded in the html file
        html = response.read().decode('utf-8', errors='ignore')
        parser = etree.XMLParser(recover=True)
        root = etree.fromstring(html.encode(), parser=parser)
        if root is not None and len(root):
            for jsonld in root.findall(".//script[@type='application/ld+json']"):
                g.parse(data=jsonld.text, publicID=BNode(), format='json-ld')


    fixedgraph = Graph()
    fixedgraph += [sanitize_triple(s) for s in g]

    return fixedgraph

def turtle(g):
    return Markdown('''
Results:

```turtle
{}
```
'''.format(g.serialize(format='turtle').decode('utf-8', errors='ignore')))

def print_data(url):
    g = extract_data(url)
    return turtle(g)

    

def check(testname):
    import tests
    
    test = getattr(tests, 'test_{}'.format(testname), None)
    if test is None:
        raise Exception('There are no tests for {}'.format(testname))
    definition = solution(testname)
    if definition is None:
        raise Exception('The definition for {} is empty or invalid.'.format(testname))
    return test(definition)
