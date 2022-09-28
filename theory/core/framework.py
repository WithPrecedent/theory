"""
core.framework: 
Corey Rayburn Yung <coreyrayburnyung@gmail.com>
Copyright 2020, Corey Rayburn Yung
License: Apache-2.0 (https://www.apache.org/licenses/LICENSE-2.0)
Contents:
    
    
"""
from __future__ import annotations
import dataclasses
import pathlib
import random
from typing import (Any, Callable, ClassVar, Dict, Iterable, List, Mapping, 
                    Optional, Sequence, Tuple, Type, Union)

import amos


@dataclasses.dataclass
class Settings(amos.Configuration):
    """Loads and stores configuration settings for a theory project.
    Args:
        contents (Union[str, pathlib.Path, Mapping[str, Mapping[str, Any]]]): a 
            dict, a str file path to a file with settings, or a pathlib Path to
            a file with settings. Defaults to en empty dict.
        infer_types (bool): whether values in 'contents' are converted to other 
            datatypes (True) or left alone (False). If 'contents' was imported 
            from an .ini file, a False value will leave all values as strings. 
            Defaults to True.
        defaults (Mapping[str, Mapping[str]]): any default options that should
            be used when a user does not provide the corresponding options in 
            their configuration settings. Defaults to a dict with 'general', 
            'files', and 'theory' sections listed.
        skip (Sequence[str]): names of suffixes to skip when constructing nodes
            for a theory project. Defaults to a list with 'general', 'files',
            'theory', and 'parameters'. 
                          
    """
    contents: Union[str, pathlib.Path, Mapping[str, Mapping[str, Any]]] = (
        dataclasses.field(default_factory = dict))
    infer_types: bool = True
    defaults: Mapping[str, Mapping[str, Any]] = dataclasses.field(
        default_factory = lambda: {'general': {'verbose': False,
                                               'parallelize': False,
                                               'conserve_memory': False,
                                               'gpu': False,
                                               'seed': random.randrange(1000)},
                                   'files': {'source_format': 'csv',
                                             'interim_format': 'csv',
                                             'final_format': 'csv',
                                             'file_encoding': 'windows-1252'},
                                   'theory': {'default_design': 'pipeline',
                                                'default_workflow': 'graph'}})
    skip: Sequence[str] = dataclasses.field(
        default_factory = lambda: ['general', 
                                   'files', 
                                   'theory', 
                                   'parameters'])

 
@dataclasses.dataclass
class Clerk(amos.Clerk):
    pass  

