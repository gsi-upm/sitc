from IPython.core.magic import (register_line_magic, register_cell_magic,
                                register_line_cell_magic)

from IPython.display import HTML, display, Image
from urllib.request import Request, urlopen
from urllib.parse import quote_plus, urlencode
from urllib.error import HTTPError

import json


def send_query(query, endpoint):
    FORMATS = ",".join(["application/sparql-results+json", "text/javascript", "application/json"])

    data = {'query': query}
    # b = quote_plus(query)

    r = Request(endpoint,
                data=urlencode(data).encode('utf-8'),
                headers={'content-type': 'application/x-www-form-urlencoded',
                         'accept': FORMATS},
                method='POST')
    return json.loads(urlopen(r).read().decode('utf-8'));


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


def query(query, endpoint=None, print_table=False):
    global LAST_QUERY

    endpoint = endpoint or "http://dbpedia.org/sparql"
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
    try:
        return query(cell, endpoint=line, print_table=True)
    except HTTPError as ex:
        error_message = ex.read().decode('utf-8')
        print('Error {}. Reason: {}'.format(ex.status, ex.reason))
        print(error_message)


def show_photos(values):
    for value in values:
        if 'http://' in value:
            display(Image(url=value))
