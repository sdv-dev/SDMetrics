from sdmetrics.single_table.privacy.util import closest_neighbors


def test_closest_neighbors_exact():
    samples = [
        ('a', '1'),
        ('a', '2'),
        ('a', '3'),
        ('b', '1'),
        ('b', '2'),
        ('b', '3'),
    ]
    target = ('a', '2')
    results = closest_neighbors(samples, target)
    assert len(results) == 1
    assert results[0] == ('a', '2')


def test_closest_neighbors_non_exact():
    samples = [
        ('a', '1'),
        ('a', '3'),
        ('b', '1'),
        ('b', '2'),
        ('b', '3'),
    ]
    target = ('a', '2')
    results = closest_neighbors(samples, target)
    assert len(results) == 3
    assert ('a', '1') in results
    assert ('a', '3') in results
    assert ('b', '2') in results
