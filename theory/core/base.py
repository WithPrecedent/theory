"""
core.base: core classes for a theory data science project
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
    Theory (framework.ProjectBase, chrisjen.Project)
    Idea (framework.ProjectBase, chrisjen.Settings):
    Clerk (framework.ProjectBase, chrisjen.Filer):
    Component (chrisjen.Component):
    Stage
    
    
    
"""
from __future__ import annotations
import abc
import copy
import dataclasses
import inspect
import random
import pathlib
from typing import (
    TYPE_CHECKING, Any, Callable, ClassVar, Dict, Hashable, Iterable, List, MutableMapping, 
    Optional, Sequence, Tuple, Type, Union)

import amos
import chrisjen
import more_itertools

if TYPE_CHECKING:
    from . import framework
    import numpy as np
    import pandas as pd


@dataclasses.dataclass
class Theory(framework.ProjectBase, chrisjen.Project):
    """Directs construction and execution of a theory data science project.
    
    Args:
        name (str): designates the name of a class instance that is used for 
            internal referencing throughout amos. For example, if a 
            amos instance needs settings from a Idea instance, 
            'name' should match the appropriate section name in a Idea 
            instance. Defaults to None. 
        idea (Union[Idea, Type[Idea], 
            pathlib.Path, str, Mapping[str, Mapping[str, Any]]]): a 
            Settings-compatible subclass or instance, a str or pathlib.Path 
            containing the file path where a file of a supported file type with
            settings for a Idea instance is located, or a 2-level 
            mapping containing settings. Defaults to the default Idea 
            instance.
        clerk (Union[Clerk, Type[Clerk], pathlib.Path, 
            str]): a Clerk-compatible class or a str or pathlib.Path 
            containing the full path of where the root folder should be located 
            for file input and output. A 'clerk' must contain all file path and 
            import/export methods for use throughout amos. Defaults to the 
            default Clerk instance. 
        identification (str): a unique identification name for a amos
            Project. The name is used for creating file folders related to the 
            project. If it is None, a str will be created from 'name' and the 
            date and time. Defaults to None.   
        outline (project.Stage): an outline of a project workflow derived from 
            'idea'. Defaults to None.
        workflow (project.Stage): a workflow of a project derived from 
            'outline'. Defaults to None.
        summary (project.Stage): a summary of a project execution derived from 
            'workflow'. Defaults to None.
        automatic (bool): whether to automatically advance 'worker' (True) or 
            whether the worker must be advanced manually (False). Defaults to 
            True.
        data (Any): any data object for the project to be applied. If it is
            None, an instance will still execute its workflow, but it won't
            apply it to any external data. Defaults to None.  
        states (ClassVar[Sequence[Union[str, project.Stage]]]): a list of Stages 
            or strings corresponding to keys in 'bases.stage.library'. Defaults 
            to a list containing 'outline', 'workflow', and 'summary'.
        validations (ClassVar[Sequence[str]]): a list of attributes that need 
            validating. Defaults to a list of attributes in the dataclass field.
    
    Attributes:
        bases (ClassVar[amos.types.Lexicon]): a class attribute containing
            a dictionary of base classes with libraries of subclasses of those 
            bases classes. Changing this attribute will entirely replace the 
            existing links between this instance and all other base classes.
        
    """
    name: str = None
    idea: Union[
        Idea, 
        Type[Idea], 
        MutableMapping[str, MutableMapping[str, Any]],
        pathlib.Path, 
        str] = None
    clerk: Union[
        Clerk, 
        Type[Clerk],
        pathlib.Path, 
        str] = None
    identification: str = None
    outline: stages.ProjectOutline = None
    workflow: stages.ProjectWorkflow = None
    summary: stages.Stage = None
    automatic: bool = True
    data: Union[str, np.ndarray, pd.DataFrame, pd.Series] = None
    stages: ClassVar[Sequence[Union[str, Stage]]] = [
        'outline', 'workflow', 'summary']
    validations: ClassVar[Sequence[str]] = [
        'idea', 'name', 'identification', 'clerk']
    
    
@dataclasses.dataclass
class Idea(framework.ProjectBase, amos.Settings):
    """Loads and stores configuration settings for a theory project.
    
    To create settings instance, a user can pass as the 'contents' parameter a:
        1) pathlib file path of a compatible file type;
        2) string containing a a file path to a compatible file type;
                                or,
        3) 2-level nested dict.

    If 'contents' is imported from a file, settings creates a dict and can 
    convert the dict values to appropriate datatypes. Currently, supported file 
    types are: ini, json, toml, yaml, and python. If you want to use toml, yaml, 
    or json, the identically named packages must be available in your python
    environment.

    If 'infer_types' is set to True (the default option), str dict values are 
    automatically converted to appropriate datatypes (str, list, float, bool, 
    and int are currently supported). Type conversion is automatically disabled
    if the source file is a python module (assuming the user has properly set
    the types of the stored python dict).

    Because settings uses ConfigParser for .ini files, by default it stores 
    a 2-level dict. The desire for accessibility and simplicity dictated this 
    limitation. A greater number of levels can be achieved by having separate
    sections with names corresponding to the strings in the values of items in 
    other sections. 

    Args:
        contents (MutableMapping[Hashable, Any]): a dict for storing 
            configuration options. Defaults to en empty dict.
        default_factory (Optional[Any]): default value to return when the 'get' 
            method is used. Defaults to an empty dict.
        default (Mapping[str, Mapping[str]]): any default options that should
            be used when a user does not provide the corresponding options in 
            their configuration settings. Defaults to an empty dict.
        infer_types (bool): whether values in 'contents' are converted to other 
            datatypes (True) or left alone (False). If 'contents' was imported 
            from an .ini file, all values will be strings. Defaults to True.
        skip (Sequence[str]): names of suffixes to skip when constructing nodes
            for a theory project. Defaults to a list with 'general', 'files',
            'theory', and 'parameters'. 
                          
    """
    contents: MutableMapping[Hashable, Any] = dataclasses.field(
        default_factory = dict)
    default_factory: Optional[Any] = dict
    infer_types: bool = True
    default: MutableMapping[str, MutableMapping[str, Any]] = dataclasses.field(
        default_factory = lambda: {
            'general': {
                'verbose': False,
                'parallelize': False,
                'conserve_memory': False,
                'gpu': False,
                'seed': random.randrange(1000)},
            'files': {
                'source_format': 'csv',
                'interim_format': 'csv',
                'final_format': 'csv',
                'file_encoding': 'windows-1252'},
            'theory': {
                'default_design': 'pipeline',
                'default_workflow': 'graph'}})
    infer_types: bool = True
    skip: Sequence[str] = dataclasses.field(
        default_factory = lambda: ['general', 'files', 'theory', 'parameters'])

 
@dataclasses.dataclass
class Clerk(framework.ProjectBase, chrisjen.Filer):
    """File and folder management for theory projects.

    Creates and stores dynamic and static file paths, properly formats files
    for import and export, and provides methods for loading and saving
    amos, pandas, and numpy objects.

    Args:
        root_folder (Union[str, pathlib.Path]): the complete path from which the 
            other paths and folders used by Clerk are ordinarily derived 
            (unless you decide to use full paths for all other options). 
            Defaults to None. If not passed, the parent folder of the current 
            working workery is used.
        input_folder (Union[str, pathlib.Path]]): the input_folder subfolder 
            name or a complete path if the 'input_folder' is not off of
            'root_folder'. Defaults to 'input'.
        output_folder (Union[str, pathlib.Path]]): the output_folder subfolder
            name or a complete path if the 'output_folder' is not off of
            'root_folder'. Defaults to 'output'.
        parameters (MutableMapping[str, str]): keys are the amos names of 
            parameters and values are the values which should be passed to the
            Distributor instances when loading or savings files. Defaults to the
            global 'default_parameters' variable.

    """
    root_folder: Union[str, pathlib.Path] = pathlib.Path('..')
    input_folder: Union[str, pathlib.Path] = 'input'
    output_folder: Union[str, pathlib.Path] = 'output'
    parameters: MutableMapping[str, str] = dataclasses.field(
        default_factory = lambda: {
            'file_encoding': 'windows-1252',
            'index_column': True,
            'include_header': True,
            'conserve_memory': False,
            'test_size': 1000,
            'threads': -1,
            'visual_tightness': 'tight', 
            'visual_format': 'png'}) 
 


@dataclasses.dataclass
class Component(framework.ProjectBase, amos.quirks.Element, abc.ABC):
    """Base class for parts of a amos Workflow.
    
    Args:
        name (str): designates the name of a class instance that is used for 
            internal referencing throughout theory. For example, if a theory 
            instance needs options from a Settings instance, 'name' should match 
            the appropriate section name in a Settings instance. Defaults to 
            None. 
                
    Attributes:
        bases (ClassVar[ProjectBases]): library that stores theory base classes 
            and allows runtime access and instancing of those stored subclasses.
        subclasses (ClassVar[amos.types.Catalog]): library that stores 
            concrete subclasses and allows runtime access and instancing of 
            those stored subclasses. 
        instances (ClassVar[amos.types.Catalog]): library that stores
            subclass instances and allows runtime access of those stored 
            subclass instances.
                
    """
    name: str = None

    """ Required Subclass Methods """
    
    @abc.abstractmethod
    def execute(self, project: Theory, 
                **kwargs) -> Theory:
        """[summary]
        Args:
            project (Theory): [description]
        Returns:
            Theory: [description]
            
        """ 
        return project

    @abc.abstractmethod
    def implement(self, project: Theory, 
                  **kwargs) -> Theory:
        """[summary]
        Args:
            project (Theory): [description]
        Returns:
            Theory: [description]
            
        """  
        return project
        
    """ Public Class Methods """
    
    @classmethod
    def create(cls, name: Union[str, Sequence[str]], **kwargs) -> Component:
        """[summary]
        Args:
            name (Union[str, Sequence[str]]): [description]
        Raises:
            KeyError: [description]
        Returns:
            Component: [description]
            
        """        
        keys = more_itertools.always_iterable(name)
        for key in keys:
            for library in ['instances', 'subclasses']:
                item = None
                try:
                    item = getattr(cls, library)[key]
                    break
                except KeyError:
                    pass
            if item is not None:
                break
        if item is None:
            raise KeyError(f'No matching item for {str(name)} was found') 
        elif inspect.isclass(item):
            return cls(name = name, **kwargs)
        else:
            instance = copy.deepcopy(item)
            for key, value in kwargs.items():
                setattr(instance, key, value)
            return instance


@dataclasses.dataclass
class Stage(framework.ProjectBase, amos.quirks.Needy, abc.ABC):
    """Creates a amos object.
    
    Args:
        needs (ClassVar[Union[Sequence[str], str]]): attributes needed from 
            another instance for some method within a subclass. Defaults to an
            empty list.     
                
    Attributes:
        bases (ClassVar[ProjectBases]): library that stores theory base classes 
            and allows runtime access and instancing of those stored subclasses.
        subclasses (ClassVar[amos.types.Catalog]): library that stores 
            concrete subclasses and allows runtime access and instancing of 
            those stored subclasses. 
        instances (ClassVar[amos.types.Catalog]): library that stores
            subclass instances and allows runtime access of those stored 
            subclass instances.
                       
    """
    needs: ClassVar[Union[Sequence[str], str]] = []
