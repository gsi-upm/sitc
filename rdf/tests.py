from future.standard_library import print_function
import operator
import types
import sys
from rdflib import term, RDF, Namespace

schema = Namespace('http://schema.org/')


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


def test_hotel(g):
    test('Some triples are loaded',
         len(g))

    hotels = set(g.subjects(RDF.type, schema['Hotel']))
    test('At least 2 hotels are loaded',
         hotels,
         2,
         atLeast)

    for hotel in hotels:
        if 'GSIHOTEL' in hotel:  # Do not check the example hotel
            continue
        props = g.predicates(hotel)
        test('Each hotel has all required properties',
             props,
             list(schema[i] for i in ['description', 'email', 'logo', 'priceRange']),
             func=containsAll)

    reviews = set(g.subjects(RDF.type, schema['Review']))
    test('At least 3 reviews are loaded',
         reviews,
         3,
         atLeast)

    for review in reviews:
        props = g.predicates(review)
        test('Each review has all required properties',
             props,
             list(schema[i] for i in ['itemReviewed', 'reviewBody', 'reviewRating']),
             func=containsAll)
        ratings = list(g.objects(review, schema['reviewRating']))
        for rating in ratings:
            value = g.value(rating, schema['ratingValue'])
            test('The review should have ratings', value)

    authors = set(g.objects(None, schema['author']))
    for author in authors:
        for prop in g.predicates(author, None):
            if 'name' in str(prop).lower():
                break
    else:
        assert "At least a reviewer has a name (surname, givenName...)"

    print('All tests passed. Congratulations!')
    print()
    print('Now you can try to add the optional properties')


def test_example(g):
    test('Some triples have been loaded',
         len(g))
    test('A person has been defined',
         g.subjects(RDF.type, term.URIRef('http://xmlns.com/foaf/0.1/Person')))
    print('All tests passed. Well done!')
