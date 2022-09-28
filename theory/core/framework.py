"""
core.framework: 
Corey Rayburn Yung <coreyrayburnyung@gmail.com>
Copyright 2020, Corey Rayburn Yung
License: Apache-2.0 (https://www.apache.org/licenses/LICENSE-2.0)
Contents:
    
    
"""
from __future__ import annotations
import abc
import dataclasses
import pathlib
import random
from typing import (Any, Callable, ClassVar, Dict, Iterable, List, Mapping, 
                    Optional, Sequence, Tuple, Type, Union)

import amos


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
