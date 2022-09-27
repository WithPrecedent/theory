"""
.. module:: tests
:synopsis: tests of core theory entitys
:author: Corey Rayburn Yung
:copyright: 2019-2020
:license: Apache-2.0
"""

import os
import sys
sys.path.insert(0, os.path.join('..', 'theory'))
sys.path.insert(0, os.path.join('..', '..', 'theory'))

import theory.content as content

algorithm, parameters = content.create(
    configuration = {'general': {'gpu': True, 'seed': 4}},
    package = 'analyst',
    step = 'scale',
    step = 'normalize',
    parameters = {'copy': False})

print(algorithm, parameters)


