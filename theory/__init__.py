"""
theory: data science made simple
Corey Rayburn Yung <coreyrayburnyung@gmail.com>
Copyright 2020-2022, Corey Rayburn Yung
License: Apache-2.0

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

Contents:
    importables (dict[str, str]): dict of imports available directly from 
        'theory'. This dict is needed to lazily import' modules using the 
        '__getattr__' function of this module.  
        
theory uses amos's lazy import system so that subpackages, modules, and
specific objects are not imported until they are first accessed.
  
ToDo:
   

For Developers:

As with all of my packages, I use Google-style docstrings and follow the Google 
Python Style Guide (https://google.github.io/styleguide/pyguide.html) with two 
notable exceptions:
    1) I always add spaces around '='. This is because I find it more readable 
        and it is practically the norm with type annotations adding the spaces
        to function and method signatures. I realize that this will seem alien
        to many coders, but it is far easier on my eyes.
    2) I've expanded the Google exception for importing multiple items from one
        package from just 'typing' to also include 'collections.abc'. This is
        because, as of python 3.9, many of the type annotations in 'typing'
        are being depreciated and have already been combined with the similarly
        named types in 'collections.abc'. I except Google will make this change
        at some point in the near future.

My packages lean heavily toward over-documentation and verbosity. This is
designed to make them more accessible to beginnning coders and generally more 
usable. The one exception to that general rule is unit tests, which hopefully 
are clear enough to not require further explanation. If there is any area of the 
documentation that could be made clearer, please don't hesitate to email me or
any other package maintainer - I want to ensure the package is as accessible and 
useful as possible.
     
"""
from __future__ import annotations
from typing import Any

import amos

__version__ = '0.1.1'

__author__ = 'Corey Rayburn Yung'


""" 
The keys of 'importables' are the attribute names of how users should access
the modules and other items listed in values. 'importables' is necessary for
the lazy importation system used throughout theory.
"""
importables: dict[str, str] = {
    'core': 'core',
    'project': 'project',
    'utilities': 'utilities',
    'decorators': 'utilities.decorators',
    'memory': 'utilities.memory',
    'tools': 'utilities.tools',
    'quirks': 'core.quirks',
    'framework': 'core.framework',
    'base': 'core.base',
    'components': 'core.components',
    'externals': 'core.externals',
    'criteria': 'core.criteria',
    'stages': 'core.stages',
    'dataset': 'core.dataset',
    'analyst': 'analyst',
    'artist': 'artist',
    'critic': 'critic',
    'explorer': 'explorer',
    'wrangler': 'wrangler',
    'ProjectBase': 'core.base.ProjectBase',
    'Component': 'core.base.Component',
    'Outline': 'core.stages.Outline',
    'Workflow': 'core.stages.Workflow',
    'Summary': 'core.stages.Summary',
    'Parameters': 'core.components.Parameters',
    'Step': 'core.components.Step',
    'Technique': 'core.components.Technique',
    'Worker': 'core.components.Worker',
    'Pipeline': 'core.components.Pipeline',
    'Contest': 'core.components.Contest',
    'Study': 'core.components.Study',
    'Survey': 'core.components.Survey',
    'Dataset': 'core.dataset.Dataset',
    'Project': 'core.interface.Project'}


def __getattr__(name: str) -> Any:
    """Lazily imports modules and items within them.
    
    Args:
        name (str): name of amos module or item.

    Raises:
        AttributeError: if there is no module or item matching 'name'.

    Returns:
        Any: a module or item stored within a module.
        
    """
    return amos.from_importables(
        name = name,
        importables= importables,
        package = __name__)
