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
    Process (framework.ProjectBase, chrisjen.Component, abc.ABC):
    Phase
    
    
    
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
    from . import stages
    import numpy as np
    import pandas as pd


@dataclasses.dataclass
class Theory(framework.ProjectBase, chrisjen.Project):
    """Directs construction and execution of a theory data science project.
    
    Args:
        name (Optional[str]): designates the name that is used for internal 
            referencing throughout theory. For example, if an instance needs 
            settings from an Idea instance, 'name' should match the appropriate 
            section name in an Idea instance. Defaults to None.  
        idea (Union[Idea, Type[Idea], pathlib.Path, str, Mapping[str, 
            Mapping[str, Any]]]): an Idea-compatible subclass or instance, a str 
            or pathlib.Path containing the file path where a file of a supported 
            file type with settings for an Idea instance is located, or a 
            2-level mapping containing settings. Defaults to the default Idea 
            instance.
        clerk (Union[Clerk, Type[Clerk], pathlib.Path, str]): a Clerk-compatible 
            class or a str or pathlib.Path containing the full path of where the 
            root folder should be located for file input and output. A 'clerk' 
            must contain all file path and import/export methods for use 
            throughout theory. Defaults to the default Clerk instance. 
        tag (str): a unique tag name for a theory project. The name is used for 
            creating file folders related to the project. If it is None, a str 
            will be created from 'name' and the date and time. Defaults to None.   
        outline (project.Phase): an outline of a project workflow derived from 
            'idea'. Defaults to None.
        workflow (project.Phase): a workflow of a project derived from 
            'outline'. Defaults to None.
        summary (project.Phase): a summary of a project execution derived from 
            'workflow'. Defaults to None.
        automatic (bool): whether to automatically advance through the phases of
            the project (True) or whether the phases must be advanced through 
            manually (False). Defaults to True.
        data (Any): any data object for the project to be applied. If it is
            None, an instance will still execute its workflow, but it won't
            apply it to any external data. Defaults to None.  
        validations (ClassVar[Sequence[str]]): a list of attributes that need 
            validating. Defaults to a list of names of attributes in the 
            dataclass field.
    
    Attributes:
        bases (ClassVar[amos.Dictionary]): a class attribute containing a 
            dictionary of base classes with libraries of subclasses of those 
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
    tag: str = None
    outline: stages.Outline = None
    workflow: stages.Workflow = None
    summary: stages.Summary = None
    automatic: bool = True
    data: Union[str, np.ndarray, pd.DataFrame, pd.Series] = None
    validations: ClassVar[Sequence[str]] = ['idea', 'name', 'tag', 'clerk']
    
    
@dataclasses.dataclass
class Idea(framework.ProjectBase, chrisjen.Settings):
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
    for import and export, and provides methods for loading and saving theory, 
    pandas, and numpy objects.

    Args:
        root_folder (Union[str, pathlib.Path]): the complete path from which the 
            other paths and folders used by Clerk are ordinarily derived 
            (unless you decide to use full paths for all other options). 
            Defaults to None. If not passed, the parent folder of the current 
            working folder is used.
        input_folder (Union[str, pathlib.Path]]): the input_folder subfolder 
            name or a complete path if the 'input_folder' is not off of
            'root_folder'. Defaults to 'input'.
        output_folder (Union[str, pathlib.Path]]): the output_folder subfolder
            name or a complete path if the 'output_folder' is not off of
            'root_folder'. Defaults to 'output'.
        parameters (MutableMapping[str, str]): keys are the names of parameters 
            and values are the values which should be passed when loading or 
            savings files. Defaults to the parameters included in the dataclass
            field.

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
class Process(framework.ProjectBase, chrisjen.Component, abc.ABC):
    """Base class for parts of a theory Workflow.
    
    Args:
        name (Optional[str]): designates the name that is used for internal 
            referencing throughout theory. For example, if an instance needs 
            settings from an Idea instance, 'name' should match the appropriate 
            section name in an Idea instance. Defaults to None. 
        contents (Optional[Any]): stored item(s) to be applied to 'project'
            passed to the 'execute' method. Defaults to None.
        parameters (MutableMapping[Hashable, Any]): parameters to be attached to 
            'contents' when the 'implement' method is called. Defaults to an
            empty Parameters instance.
                
    """
    name: Optional[str] = None
    contents: Optional[Any] = None
    parameters: MutableMapping[Hashable, Any] = dataclasses.field(
        default_factory = chrisjen.Parameters)  

    """ Required Subclass Methods """
    
    @abc.abstractmethod
    def execute(
        self, 
        item: Theory, 
        *args: Optional[Any], 
        **kwargs: Optional[Any]) -> Theory:
        """Calls the 'implement' method after finalizing parameters.

        The 'execute' method can also be used to call the 'implement' method
        multiple times.
        
        Args:
            item (Theory): a Theory instance which contains any item or data to 
                which 'contents' should be applied.

        Returns:
            Theory: a Theory instance with results added after applying 
                'contents'.
            
        """
        return item

    @abc.abstractmethod
    def implement(
        self, 
        item: Theory, 
        *args: Optional[Any], 
        **kwargs: Optional[Any]) -> Theory:
        """Applies 'contents' to 'item'.

        Subclasses must provide their own methods.

        Args:
            item (Theory): a Theory instance which contains any item or data to 
                which 'contents' should be applied.

        Returns:
            Theory: a Theory instance with results added after applying 
                'contents'.
            
        """ 
        return item
        
    # """ Public Class Methods """
    
    # @classmethod
    # def create(cls, name: Union[str, Sequence[str]], **kwargs) -> Process:
    #     """[summary]
    #     Args:
    #         name (Union[str, Sequence[str]]): [description]
    #     Raises:
    #         KeyError: [description]
    #     Returns:
    #         Process: [description]
            
    #     """        
    #     keys = more_itertools.always_iterable(name)
    #     for key in keys:
    #         for library in ['instances', 'subclasses']:
    #             item = None
    #             try:
    #                 item = getattr(cls, library)[key]
    #                 break
    #             except KeyError:
    #                 pass
    #         if item is not None:
    #             break
    #     if item is None:
    #         raise KeyError(f'No matching item for {str(name)} was found') 
    #     elif inspect.isclass(item):
    #         return cls(name = name, **kwargs)
    #     else:
    #         instance = copy.deepcopy(item)
    #         for key, value in kwargs.items():
    #             setattr(instance, key, value)
    #         return instance


@dataclasses.dataclass
class Phase(framework.ProjectBase, amos.quirks.Needy, abc.ABC):
    """Creates a theory object.
    
    Args:
        needs (ClassVar[Union[Sequence[str], str]]): attributes needed from 
            another instance for some method within a subclass. Defaults to an
            empty list.     
                
    Attributes:
        bases (ClassVar[ProjectBases]): library that stores theory base classes 
            and allows runtime access and instancing of those stored subclasses.
        subclasses (ClassVar[amos.Catalog]): library that stores 
            concrete subclasses and allows runtime access and instancing of 
            those stored subclasses. 
        instances (ClassVar[amos.Catalog]): library that stores
            subclass instances and allows runtime access of those stored 
            subclass instances.
                       
    """
    needs: ClassVar[Union[Sequence[str], str]] = []
