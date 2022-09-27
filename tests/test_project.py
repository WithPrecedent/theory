"""
.. module:: project test
:synopsis: tests Project class
:author: Corey Rayburn Yung
:copyright: 2019-2020
:license: Apache-2.0
"""

from pathlib import pathlib.Path

import pandas as pd
import pytest

from theory.core.idea import Idea
from theory.core.project import Project


def test_project():
    idea = Idea(
        configuration = pathlib.Path.cwd().joinpath('tests', 'idea_settings.ini'))
    project = Project(idea = idea)
    print('test', project.library)
    return

if __name__ == '__main__':
    test_project()