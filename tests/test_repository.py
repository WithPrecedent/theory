"""
.. module:: idea repository
:synopsis: tests SimpleRepository and Plan classes
:author: Corey Rayburn Yung
:copyright: 2019-2020
:license: Apache-2.0
"""

from pathlib import pathlib.Path
from theory.core.repository import Plan
from theory.core.repository import SimpleRepository


def test_repository():
    repository = SimpleRepository(contents = {
        'run' : 'tired',
        'sleep': 'rested',
        'walk': 'relax'})
    plan = Plan(
        steps = ['walk', 'sleep'],
        defaults = ['run']
        repository = repository)
    assert plan['all'] == ['walk', 'sleep']
    assert plan['default'] = ['run']
    assert plan[0] = 'walk'
    assert plan['walk'] = 0
    for i, step in enumerate(plan):
        if i == 1:
            assert step == 'relax'
        elif i == 2:
            asser step == 'rested'
    return


if __name__ == '__main__':
    test_repository()