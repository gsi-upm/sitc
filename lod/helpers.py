'''
Helper functions and ipython magic for the SPARQL exercises.

The tests in the notebooks rely on the `LAST_QUERY` variable, which is updated by the `%%sparql` magic after every query.
This variable contains the full query used (`LAST_QUERY["query"]`), the endpoint it was sent to (`LAST_QUERY["endpoint"]`), and a dictionary with the response of the endpoint (`LAST_QUERY["results"]`).
For convenience, the results are also given as tuples (`LAST_QUERY["tuples"]`), and as a dictionary of of `{column:[values]}` (`LAST_QUERY["columns"]`).
'''
from IPython.core.magic import (register_line_magic, register_cell_magic,
                                register_line_cell_magic)
from IPython.display import HTML, display, Image, display_javascript
from urllib.request import Request, urlopen
from urllib.parse import quote_plus, urlencode
from urllib.error import HTTPError

import json
import sys

js = "IPython.CodeCell.options_default.highlight_modes['magic_sparql'] = {'reg':[/^%%sparql/]};"
display_javascript(js, raw=True)


def send_query(query, endpoint):
    FORMATS = ",".join(["application/sparql-results+json",
                        "text/javascript",
                        "application/json"])

    data = {'query': query}
    # b = quote_plus(query)

    r = Request(endpoint,
                data=urlencode(data).encode('utf-8'),
                headers={'content-type': 'application/x-www-form-urlencoded',
                         'accept': FORMATS},
                method='POST')
    res = urlopen(r)
    data = res.read().decode('utf-8')
    if res.getcode() == 200:
        try:
            return json.loads(data)
        except Exception:
            print('Got: ', data, file=sys.stderr)
            raise
    raise Exception('Error getting results: {}'.format(data))


def tabulate(tuples, header=None):
    if not header:
        header, tuples = tuples[0], tuples[1:]
    header = '<tr>{}<tr>'.format(''.join('<th>{}</th>'.format(h) for h in header))
    rows = []
    for row in tuples:
        inner = ''.join('<td>{}</td>'.format(c) for c in row)
        rows.append('<tr>{}</tr>'.format(inner))
    body = ''.join(rows)
    table = '<table><thead>{header}</thead><tbody>{body}</tbody></table>'.format(body=body,
                                                                                 header=header)
    return table


LAST_QUERY = {}

def solution():
    return LAST_QUERY


def query(query, endpoint=None, print_table=False):
    global LAST_QUERY

    endpoint = endpoint or "http://fuseki.gsi.upm.es/sitc/"
    results = send_query(query, endpoint)
    tuples = to_table(results)


    columns = {}
    header, values = tuples[0], tuples[1:]

    for ix, h in enumerate(header):
        columns[h] = []
        for row in values:
            columns[h].append(row[ix])

    LAST_QUERY.update({
        'query': query,
        'endpoint': query,
        'results': results,
        'tuples': values,
        'columns': columns
    })

    if not print_table:
        return tuples
    return HTML(tabulate(tuples))


def to_table(results):
    table = []
    header = results['head']['vars']
    table.append(header)
    for result in results["results"]["bindings"]:
        table.append(tuple(result.get(h, {}).get('value', "") for h in header))
    return table


@register_cell_magic
def sparql(line, cell):
    '''
    Sparql magic command for ipython. It can be used in a cell like this:
    
    ```
    %%sparql
    
    ... Your SPARQL query ...
    
    ```
    
    by default, it will use the DBpedia endpoint, but you can use a different endpoint like this:
    
    ```
    %%sparql http://my-sparql-endpoint...
    
    ... Your SPARQL query ...
    ```
    '''
    try:
        return query(cell, endpoint=line, print_table=True)
    except HTTPError as ex:
        error_message = ex.read().decode('utf-8')
        print('Error {}. Reason: {}'.format(ex.status, ex.reason))
        print(error_message, file=sys.stderr)


def show_photos(values):
    for value in values:
        if 'http://' in value:
            display(Image(url=value))
