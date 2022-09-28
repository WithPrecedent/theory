"""
base: core classes for a theory data science project
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
    Idea (amos.project.Settings):
    Clerk (amos.project.Clerk):
    ProjectManager (amos.project.Manager):
    ProjectComponent (amos.project.Component):
    ProjectAlgorithm
    ProjectCriteria
    
    
    
"""
from __future__ import annotations
import abc
import copy
import dataclasses
import inspect
import random
import pathlib
from typing import (
    Any, Callable, ClassVar, Dict, Hashable, Iterable, List, MutableMapping, 
    Optional, Sequence, Tuple, Type, Union)

import amos
import chrisjen
import more_itertools


@dataclasses.dataclass
class Idea(amos.Settings):
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
class Clerk(chrisjen.Filer):
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
class ProjectBases(object):
    """Stores base classes in theory.
     
    """
    def register(self, name: str, item: Union[Type, object]) -> None:
        """[summary]
        Args:
            name (str): [description]
            item (Union[Type, object]): [description]
        Raises:
            ValueError: [description]
            TypeError: [description]
        Returns:
            [type]: [description]
            
        """
        if name in dir(self):
            raise ValueError(f'{name} is already registered')
        elif inspect.isclass(item) and issubclass(item, ProjectBase):
            setattr(self, name, item)
        elif isinstance(item, ProjectBase):
            setattr(self, name, item.__class__)
        else:
            raise TypeError(f'item must be a ProjectBase')
        return self

    def remove(self, name: str) -> None:
        """[summary]
        Args:
            name (str): [description]
        Raises:
            AttributeError: [description]
            
        """
        try:
            delattr(self, name)
        except AttributeError:
            raise AttributeError(f'{name} does not exist in {self.__name__}')


@dataclasses.dataclass
class ProjectBase(abc.ABC):
    """Base mixin for automatic registration of subclasses and instances. 
    
    Any concrete (non-abstract) subclass will automatically store itself in the 
    class attribute 'subclasses' using the snakecase name of the class as the 
    key.
    
    Any direct subclass will automatically store itself in the class attribute 
    'bases' using the snakecase name of the class as the key.
    
    Any instance of a subclass will be stored in the class attribute 'instances'
    as long as '__post_init__' is called (either by a 'super()' call or if the
    instance is a dataclass and '__post_init__' is not overridden).
    
    Args:
        bases (ClassVar[ProjectBases]): library that stores direct subclasses 
            (those with Base in their '__bases__' attribute) and allows runtime 
            access and instancing of those stored subclasses.
    
    Attributes:
        subclasses (ClassVar[amos.types.Catalog]): library that stores 
            concrete subclasses and allows runtime access and instancing of 
            those stored subclasses. 'subclasses' is automatically created when 
            a direct ProjectBase subclass (ProjectBase is in its '__bases__') is 
            instanced.
        instances (ClassVar[amos.types.Catalog]): library that stores
            subclass instances and allows runtime access of those stored 
            subclass instances. 'instances' is automatically created when a 
            direct ProjectBase subclass (ProjectBase is in its '__bases__') is 
            instanced. 
                      
    Namespaces: 
        bases, subclasses, instances, borrow, instance, and __init_subclass__.
    
    """
    bases: ClassVar[ProjectBases] = ProjectBases()
    
    """ Initialization Methods """
    
    def __init_subclass__(cls, **kwargs):
        """Adds 'cls' to appropriate class libraries."""
        super().__init_subclass__(**kwargs)
        # Creates a snakecase key of the class name.
        key = amos.tools.snakify(cls.__name__)
        # Adds class to 'bases' if it is a base class.
        if ProjectBase in cls.__bases__:
            # Creates libraries on this class base for storing subclasses.
            cls.subclasses = amos.types.Catalog()
            cls.instances = amos.types.Catalog()
            # Adds this class to 'bases' using 'key'.
            cls.bases.register(name = key, item = cls)
        # Adds concrete subclasses to 'library' using 'key'.
        if not abc.ABC in cls.__bases__:
            cls.subclasses[key] = cls

    def __post_init__(self) -> None:
        """Initializes class instance attributes."""
        # Calls parent and/or mixin initialization method(s).
        try:
            super().__post_init__()
        except AttributeError:
            pass
        try:
            key = self.name
        except AttributeError:
            key = amos.tools.snakify(self.__class__.__name__)
        self.instances[key] = self
 
    """ Public Class Methods """
    
    @classmethod
    def borrow(cls, name: Union[str, Sequence[str]]) -> Type[ProjectBase]:
        """[summary]
        Args:
            name (Union[str, Sequence[str]]): [description]
        Raises:
            KeyError: [description]
        Returns:
            ProjectBase: [description]
            
        """
        item = None
        for key in more_itertools.always_iterable(name):
            try:
                item = cls.subclasses[key]
                break
            except KeyError:
                pass
        if item is None:
            raise KeyError(f'No matching item for {str(name)} was found') 
        else:
            return item
           
    @classmethod
    def instance(cls, name: Union[str, Sequence[str]], **kwargs) -> ProjectBase:
        """[summary]
        Args:
            name (Union[str, Sequence[str]]): [description]
        Raises:
            KeyError: [description]
        Returns:
            ProjectBase: [description]
            
        """
        item = None
        for key in more_itertools.always_iterable(name):
            for library in ['instances', 'subclasses']:
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
class Component(ProjectBase, amos.quirks.Element, abc.ABC):
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
    def execute(self, project: amos.Project, 
                **kwargs) -> amos.Project:
        """[summary]
        Args:
            project (amos.Project): [description]
        Returns:
            amos.Project: [description]
            
        """ 
        return project

    @abc.abstractmethod
    def implement(self, project: amos.Project, 
                  **kwargs) -> amos.Project:
        """[summary]
        Args:
            project (amos.Project): [description]
        Returns:
            amos.Project: [description]
            
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
class Stage(ProjectBase, amos.quirks.Needy, abc.ABC):
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
