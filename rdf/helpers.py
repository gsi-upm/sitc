import sys
import operator
import types
from future.standard_library import install_aliases
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


schema = Namespace('http://schema.org/')

DEFINITIONS = {}

def solution(exercise='default'):
    if exercise not in DEFINITIONS:
        raise Exception('Solution for {} not found'.format(exercise))
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
    try:
        DEFINITIONS[key] = g.parse(data=cell,
                                  format="n3")
    except SyntaxError as ex:
        return Markdown(msg.format(line=ex.lines, reason=ex._why))
    except Exception as ex:
        return Markdown(msg.format(line='?', reason=ex))
    return Markdown('File loaded!')
        
    return HTML('Loaded!') #HTML('<code>{}</code>'.format(cell))


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

#     print(g.serialize(format='turtle').decode('utf-8', errors='ignore'))
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

    

def test(description, got, expected=None, func=None):
    if isinstance(got, types.GeneratorType):
        got = set(got)
    try:
        if expected is None:
            func = func or operator.truth
            expected = True
            assert func(got)
        else:
            func = func or operator.eq
            assert func(got, expected)
    except AssertionError:
        print('Test failed: {}'.format(description), file=sys.stderr)
        print('\tExpected: {}'.format(expected), file=sys.stderr)
        print('\tGot:      {}'.format(got), file=sys.stderr)
        raise Exception('Test failed: {}'.format(description))

        
def atLeast(lst, number):
    return len(set(lst))>=number

def containsAll(lst, other):
    for i in other:
        if i not in lst:
            print('{} not found'.format(i), file=sys.stderr)
            return False
    return True