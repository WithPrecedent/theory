"""
components: core components of a data science workflow
Corey Rayburn Yung <coreyrayburnyung@gmail.com>
Copyright 2020, Corey Rayburn Yung
License: Apache-2.0 (https://www.apache.org/licenses/LICENSE-2.0)

Contents:
    
"""
from __future__ import annotations
import abc
import dataclasses
import multiprocessing
from typing import (Any, Callable, ClassVar, Dict, Iterable, List, Mapping, 
                    Optional, Sequence, Tuple, Type, Union)

import more_itertools
import amos

from . import base
from . import Phases


@dataclasses.dataclass    
class Parameters(amos.Dictionary):
    """Creates and stores parameters for a theory component.
    
    Parameters allows parameters to be drawn from several different sources, 
    including those which only become apparent during execution of a theory
    project.
    
    Parameters can be unpacked with '**', which will turn the 'contents' 
    attribute an ordinary set of kwargs. In this way, it can serve as a drop-in
    replacement for a dict that would ordinarily be used for accumulating 
    keyword arguments.
    
    If a theory class uses a Parameters instance, the 'finalize' method should
    be called before that instance's 'implement' method in order for each of the
    parameter types to be incorporated.
    
    Args:
        name (str): designates the name of a class instance that is used for 
            internal referencing throughout theory. For example, if a theory 
            instance needs options from a Settings instance, 'name' should match 
            the appropriate section name in a Settings instance. Defaults to 
            None. 
        contents (Mapping[str, Any]): keyword parameters for use by a theory
            classes' 'implement' method. The 'finalize' method should be called
            for 'contents' to be fully populated from all sources. Defaults to
            an empty dict.
        default (Mapping[str, Any]): default parameters to use if none are 
            provided through an argument or settings. 'default' will also be
            used if any parameters are listed in 'required', in which case the
            parameters will be drawn from 'default' if they are not otherwise
            provided. Defaults to an empty dict.
        runtime (Mapping[str, str]): parameters that can only be determined at
            runtime due to dynamic action of theory. The keys should be the
            names of the parameters and the values should be attributes or items
            in 'contents' of 'project' passed to the 'finalize' method. Defaults
            to an emtpy dict.
        required (Sequence[str]): parameters that must be passed when the 
            'implement' method of a theory class is called.
        selected (Sequence[str]): an exclusive list of parameters that are 
            allowed. If 'selected' is empty, all possible parameters are 
            allowed. However, if any are listed, all other parameters that are
            included are removed. This is can be useful when including 
            parameters in a Settings instance for an entire step, only some of
            which might apply to certain techniques. Defaults to an empty dict.

    """
    name: str = None
    contents: Mapping[str, Any] = dataclasses.field(default_factory = dict)
    default: Mapping[str, Any] = dataclasses.field(default_factory = dict)
    runtime: Mapping[str, str] = dataclasses.field(default_factory = dict)
    required: Sequence[str] = dataclasses.field(default_factory = list)
    selected: Sequence[str] = dataclasses.field(default_factory = list)
      
    """ Public Methods """

    def finalize(self, project: base.Theory, **kwargs) -> None:
        """[summary]

        Args:
            name (str):
            project (base.Theory):
            
        """
        # Uses kwargs or 'default' parameters as a starting base.
        self.contents = kwargs if kwargs else self.default
        # Adds any parameters from 'settings'.
        try:
            self.contents.update(self._get_from_settings(
                settings = project.settings))
        except AttributeError:
            pass
        # Adds any required parameters.
        for item in self.required:
            if item not in self.contents:
                self.contents[item] = self.default[item]
        # Adds any runtime parameters.
        if self.runtime:
            self.add_runtime(theory =project) 
            # Limits parameters to those selected.
            if self.selected:
                self.contents = {k: self.contents[k] for k in self.selected}
        return self

    """ Private Methods """
    
    def _add_runtime(self, project: base.Theory, **kwargs) -> None:
        """[summary]

        Args:
            project (base.Theory):
            
        """    
        for parameter, attribute in self.runtime.items():
            try:
                self.contents[parameter] = getattr(project, attribute)
            except AttributeError:
                try:
                    self.contents[parameter] = project.contents[attribute]
                except (KeyError, AttributeError):
                    pass
        if self.selected:
            self.contents = {k: self.contents[k] for k in self.selected}
        return self
     
    def _get_from_settings(self, settings: Mapping[str, Any]) -> Dict[str, Any]: 
        """[summary]

        Args:
            name (str): [description]
            settings (Mapping[str, Any]): [description]

        Returns:
            Dict[str, Any]: [description]
            
        """
        try:
            parameters = settings[f'{self.name}_parameters']
        except KeyError:
            suffix = self.name.split('_')[-1]
            prefix = self.name[:-len(suffix) - 1]
            try:
                parameters = settings[f'{prefix}_parameters']
            except KeyError:
                try:
                    parameters = settings[f'{suffix}_parameters']
                except KeyError:
                    parameters = {}
        return parameters


@dataclasses.dataclass
class TheoryProcess(base.Process, abc.ABC):
    """Base class for parts of a amos Workflow.

    Args:
        name (str): designates the name of a class instance that is used for 
            internal referencing throughout theory. For example, if a theory 
            instance needs options from a Settings instance, 'name' should match 
            the appropriate section name in a Settings instance. Defaults to 
            None. 
        contents (Union[Callable, Type, object, str]): stored item(s) for use by 
            a Process subclass instance. If it is Type or str, an instance 
            will be created. If it is a str, that instance will be drawn from 
            the 'instances' or 'subclasses' attributes.
        parameters (Union[Mapping[str, Any], base.Parameters]): parameters, in 
            the form of an ordinary dict or a Parameters instance, to be 
            attached to 'contents'  when the 'implement' method is called.
            Defaults to an empty Parameters instance.
        iterations (Union[int, str]): number of times the 'implement' method 
            should  be called. If 'iterations' is 'infinite', the 'implement' 
            method will continue indefinitely unless the method stops further 
            iteration. Defaults to 1.
        parallel (ClassVar[bool]): indicates whether this Process design is
            meant to be part of a parallel workflow structure. Defaults to 
            False.

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
    name: str = None
    contents: Union[Callable, Type, object, str] = None
    parameters: Union[Mapping[str, Any], base.Parameters] = base.Parameters()
    iterations: Union[int, str] = 1
    parallel: ClassVar[bool] = False
    
    """ Public Methods """
    
    def execute(self, project: base.Theory, 
                **kwargs) -> base.Theory:
        """[summary]

        Args:
            project (base.Theory): [description]

        Returns:
            base.Theory: [description]
            
        """ 
        if self.iterations in ['infinite']:
            while True:
                theory =self.implement(theory =project, **kwargs)
        else:
            for iteration in range(self.iterations):
                theory =self.implement(theory =project, **kwargs)
        return theory

    def implement(self, project: base.Theory, 
                  **kwargs) -> base.Theory:
        """[summary]

        Args:
            project (base.Theory): [description]

        Returns:
            base.Theory: [description]
            
        """  
        if self.parameters:
            parameters = self.parameters
            parameters.update(kwargs)
        else:
            parameters = kwargs
        if self.contents not in [None, 'None', 'none']:
            theory =self.contents.execute(theory =project, **parameters)
        return theory


@dataclasses.dataclass
class Step(TheoryProcess):
    """Wrapper for a Technique.

    Subclasses of Step can store additional methods and attributes to implement
    all possible technique instances that could be used. This is often useful 
    when using parallel Worklow instances which test a variety of strategies 
    with similar or identical parameters and/or methods.

    A Step instance will try to return attributes from Technique if the
    attribute is not found in the Step instance. 

    Args:
        name (str): designates the name of a class instance that is used for 
            internal referencing throughout theory. For example, if a theory 
            instance needs options from a Settings instance, 'name' should match 
            the appropriate section name in a Settings instance. Defaults to 
            None. 
        contents (Union[Callable, Type, object, str]): stored item(s) for use by 
            a Process subclass instance. If it is Type or str, an instance 
            will be created. If it is a str, that instance will be drawn from 
            the 'instances' or 'subclasses' attributes.
        parameters (Union[Mapping[str, Any], base.Parameters]): parameters, in 
            the form of an ordinary dict or a Parameters instance, to be 
            attached to 'contents'  when the 'implement' method is called.
            Defaults to an empty Parameters instance.
        iterations (Union[int, str]): number of times the 'implement' method 
            should  be called. If 'iterations' is 'infinite', the 'implement' 
            method will continue indefinitely unless the method stops further 
            iteration. Defaults to 1.
        parallel (ClassVar[bool]): indicates whether this Process design is
            meant to be part of a parallel workflow structure. Because Steps
            are generally part of a parallel-structured workflow, the attribute
            defaults to True.
                                                
    """
    name: str = None
    contents: Union[Callable, Type, object, str] = None
    parameters: Union[Mapping[str, Any], base.Parameters] = base.Parameters()
    iterations: Union[int, str] = 1
    parallel: ClassVar[bool] = True

    
@dataclasses.dataclass
class Technique(amos.quirks.Loader, TheoryProcess):
    """Primitive object for executing algorithms in a theory workflow.

    Args:
        name (str): designates the name of a class instance that is used for 
            internal referencing throughout theory. For example, if a theory 
            instance needs options from a Settings instance, 'name' should match 
            the appropriate section name in a Settings instance. Defaults to 
            None. 
        contents (Union[Callable, Type, object, str]): stored item(s) for use by 
            a Process subclass instance. If it is Type or str, an instance 
            will be created. If it is a str, that instance will be found in 
            'module'. Defaults to None.
        parameters (Union[Mapping[str, Any], base.Parameters]): parameters, in 
            the form of an ordinary dict or a Parameters instance, to be 
            attached to 'contents'  when the 'implement' method is called.
            Defaults to an empty Parameters instance.
        iterations (Union[int, str]): number of times the 'implement' method 
            should  be called. If 'iterations' is 'infinite', the 'implement' 
            method will continue indefinitely unless the method stops further 
            iteration. Defaults to 1.
        module (str): name of module where 'contents' is located if 'contents'
            is a string. It can either be a theory or external module, as
            long as it is available to the python environment. Defaults to None.
        parallel (ClassVar[bool]): indicates whether this Process design is
            meant to be part of a parallel workflow structure. Defaults to 
            False.
                                                
    """  
    name: str = None
    contents: Union[Callable, Type, object, str] = None
    parameters: Union[Mapping[str, Any], base.Parameters] = base.Parameters()
    iterations: Union[int, str] = 1
    module: str = None
    parallel: ClassVar[bool] = False   

              
@dataclasses.dataclass
class Worker(TheoryProcess):
    """An iterable in a amos workflow that maintains its own workflow.

    Args:
        name (str): designates the name of a class instance that is used for 
            internal referencing throughout theory. For example, if a theory 
            instance needs options from a Settings instance, 'name' should match 
            the appropriate section name in a Settings instance. Defaults to 
            None. 
        contents (Union[Callable, Type, object, str]): stored item(s) for use by 
            a Process subclass instance. If it is Type or str, an instance 
            will be created. If it is a str, that instance will be drawn from 
            the 'instances' or 'subclasses' attributes.
        parameters (Union[Mapping[str, Any], base.Parameters]): parameters, in 
            the form of an ordinary dict or a Parameters instance, to be 
            attached to 'contents'  when the 'implement' method is called.
            Defaults to an empty Parameters instance.
        iterations (Union[int, str]): number of times the 'implement' method 
            should  be called. If 'iterations' is 'infinite', the 'implement' 
            method will continue indefinitely unless the method stops further 
            iteration. Defaults to 1.
        workflow (Phases.Workflow): a workflow of other theory Processs.
            Defaults to an empty Workflow.
        parallel (ClassVar[bool]): indicates whether this Process design is
            meant to be part of a parallel workflow structure. Because Steps
            are generally part of a parallel-structured workflow, the attribute
            defaults to True.
                                                
    """
    name: str = None
    contents: Union[Callable, Type, object, str] = None
    parameters: Union[Mapping[str, Any], base.Parameters] = base.Parameters()
    iterations: Union[int, str] = 1
    workflow: Phases.Workflow = Phases.Workflow()
    parallel: ClassVar[bool] = True
    
    """ Public Class Methods """

    @classmethod
    def from_outline(cls, name: str,
                     outline: amos.project.Outline, **kwargs) -> Worker:
        """[summary]

        Args:
            name (str): [description]
            outline (amos.project.Outline): [description]

        Returns:
            Worker: [description]
            
        """        
        worker = super().from_outline(name = name, outline = outline, **kwargs)
        if hasattr(worker, 'workflow'):
            worker.workflow = cls.bases.Phase.library.borrow(
                names = 'workflow')()
            if worker.parallel:
                method = cls._create_parallel
            else:
                method = cls._create_serial
            worker = method(worker = worker, outline = outline)
        return worker
                  
    """ Private Class Methods """ 

    @classmethod                
    def _create_parallel(cls, worker: Worker,
                         outline: amos.project.Outline) -> Worker:
        """[summary]

        Args:
            worker (Worker): [description]
            outline (amos.project.Outline): [description]

        Returns:
        
        """
        name = worker.name
        step_names = outline.components[name]
        possible = [outline.components[s] for s in step_names]
        worker.workflow.branchify(nodes = possible)
        for i, step_options in enumerate(possible):
            for option in step_options:
                technique = cls.from_outline(name = option, outline = outline)
                wrapper = cls.from_outline(name = step_names[i],
                                           outline = outline,
                                           contents = technique)
                worker.workflow.components[option] = wrapper
        return worker 
    
    @classmethod
    def _create_serial(cls, worker: Worker,
                       outline: amos.project.Outline) -> Worker:                     
        """[summary]

        Args:
            worker (Worker): [description]
            outline (amos.project.Outline): [description]

        Returns:
        
        """
        print('test serial', worker.name)
        name = worker.name
        components = cls._depth_first(name = name, outline = outline)
        collapsed = list(more_itertools.collapse(components))
        worker.workflow.extend(nodes = collapsed)
        for item in collapsed:
            component = cls.from_outline(name = item, outline = outline)
            worker.workflow.components[item] = component
        return worker

    @classmethod
    def _depth_first(cls, name: str, outline: base.Phase) -> List:
        """

        Args:
            name (str):
            details (Blueprint): [description]

        Returns:
            List[List[str]]: [description]
            
        """
        organized = []
        components = outline.components[name]
        for item in components:
            organized.append(item)
            if item in outline.components:
                organized_subcomponents = []
                subcomponents = cls._depth_first(name = item, outline = outline)
                organized_subcomponents.append(subcomponents)
                if len(organized_subcomponents) == 1:
                    organized.append(organized_subcomponents[0])
                else:
                    organized.append(organized_subcomponents)
        return organized
     

@dataclasses.dataclass
class Pipeline(Worker):
    """
        
    Args:
        name (str): designates the name of a class instance that is used for 
            internal referencing throughout theory. For example, if a theory 
            instance needs options from a Settings instance, 'name' should match 
            the appropriate section name in a Settings instance. Defaults to 
            None. 
        contents (Union[Callable, Type, object, str]): stored item(s) for use by 
            a Process subclass instance. If it is Type or str, an instance 
            will be created. If it is a str, that instance will be drawn from 
            the 'instances' or 'subclasses' attributes.
        parameters (Union[Mapping[str, Any], base.Parameters]): parameters, in 
            the form of an ordinary dict or a Parameters instance, to be 
            attached to 'contents'  when the 'implement' method is called.
            Defaults to an empty Parameters instance.
        iterations (Union[int, str]): number of times the 'implement' method 
            should  be called. If 'iterations' is 'infinite', the 'implement' 
            method will continue indefinitely unless the method stops further 
            iteration. Defaults to 1.
        workflow (Phases.Workflow): a workflow of other theory Processs.
            Defaults to an empty Workflow.
        parallel (ClassVar[bool]): indicates whether this Process design is
            meant to be part of a parallel workflow structure. Because Steps
            are generally part of a parallel-structured workflow, the attribute
            defaults to True.
                                                
    """
    name: str = None
    contents: Union[Callable, Type, object, str] = None
    parameters: Union[Mapping[str, Any], base.Parameters] = base.Parameters()
    iterations: Union[int, str] = 1
    workflow: Phases.Workflow = Phases.Workflow()
    parallel: ClassVar[bool] = True
    

@dataclasses.dataclass
class ParallelWorker(Worker, abc.ABC):
    """Resolves a parallel workflow by selecting the best option.

    It resolves a parallel workflow based upon the criteria in 'contents'.
        
    Args:
        name (str): designates the name of a class instance that is used for 
            internal referencing throughout theory. For example, if a theory 
            instance needs options from a Settings instance, 'name' should match 
            the appropriate section name in a Settings instance. Defaults to 
            None. 
        contents (Union[Callable, Type, object, str]): stored item(s) for use by 
            a Process subclass instance. If it is Type or str, an instance 
            will be created. If it is a str, that instance will be drawn from 
            the 'instances' or 'subclasses' attributes.
        parameters (Union[Mapping[str, Any], base.Parameters]): parameters, in 
            the form of an ordinary dict or a Parameters instance, to be 
            attached to 'contents'  when the 'implement' method is called.
            Defaults to an empty Parameters instance.
        iterations (Union[int, str]): number of times the 'implement' method 
            should  be called. If 'iterations' is 'infinite', the 'implement' 
            method will continue indefinitely unless the method stops further 
            iteration. Defaults to 1.
        workflow (Phases.Workflow): a workflow of other theory Processs.
            Defaults to an empty Workflow.
        parallel (ClassVar[bool]): indicates whether this Process design is
            meant to be part of a parallel workflow structure. Because Steps
            are generally part of a parallel-structured workflow, the attribute
            defaults to True.
                                                
    """
    name: str = None
    contents: Union[Callable, Type, object, str] = None
    parameters: Union[Mapping[str, Any], base.Parameters] = base.Parameters()
    iterations: Union[int, str] = 1
    workflow: Phases.Workflow = Phases.Workflow()
    parallel: ClassVar[bool] = True

    """ Public Methods """
    
    def implement(self, data: Any, **kwargs) -> Any:
        """[summary]

        Args:
            data (Any): [description]

        Returns:
            Any: [description]
            
        """        
        if hasattr(data, 'parallelize') and data.parallelize:
            method = self._implement_in_parallel
        else:
            method = self._implement_in_serial
        return method(data = data, **kwargs)

    """ Private Methods """
   
    def _implement_in_parallel(self, data: Any, **kwargs) -> Any:
        """Applies 'implementation' to 'project' using multiple cores.

        Args:
            project (Theory): amos project to apply changes to and/or
                gather needed data from.
                
        Returns:
            Theory: with possible alterations made.       
        
        """
        multiprocessing.set_start_method('spawn')
        with multiprocessing.Pool() as pool:
            data = pool.starmap(self._implement_in_serial, data, **kwargs)
        return data 

    def _implement_in_serial(self, data: Any, **kwargs) -> Any:
        """Applies 'implementation' to 'project' using multiple cores.

        Args:
            project (Theory): amos project to apply changes to and/or
                gather needed data from.
                
        Returns:
            Theory: with possible alterations made.       
        
        """
        for path in self.workflow.permutations:
            data = self._implement_path(data = data, path = path, **kwargs)
        return data
    
    def _implement_path(self, data: Any, path: List[str], **kwargs) -> Any:  
        for node in path:
            component = self.workflow.components[node]
            data = component.execute(data = data, **kwargs)
        return data
    
       
@dataclasses.dataclass
class Contest(ParallelWorker):
    """Resolves a parallel workflow by selecting the best option.

    It resolves a parallel workflow based upon criteria in 'contents'
        
    Args:
        name (str): designates the name of a class instance that is used for 
            internal referencing throughout theory. For example, if a theory 
            instance needs options from a Settings instance, 'name' should match 
            the appropriate section name in a Settings instance. Defaults to 
            None. 
        contents (Union[Callable, Type, object, str]): stored item(s) for use by 
            a Process subclass instance. If it is Type or str, an instance 
            will be created. If it is a str, that instance will be drawn from 
            the 'instances' or 'subclasses' attributes.
        parameters (Union[Mapping[str, Any], base.Parameters]): parameters, in 
            the form of an ordinary dict or a Parameters instance, to be 
            attached to 'contents'  when the 'implement' method is called.
            Defaults to an empty Parameters instance.
        iterations (Union[int, str]): number of times the 'implement' method 
            should  be called. If 'iterations' is 'infinite', the 'implement' 
            method will continue indefinitely unless the method stops further 
            iteration. Defaults to 1.
        workflow (Phases.Workflow): a workflow of other theory Processs.
            Defaults to an empty Workflow.
        parallel (ClassVar[bool]): indicates whether this Process design is
            meant to be part of a parallel workflow structure. Because Steps
            are generally part of a parallel-structured workflow, the attribute
            defaults to True.
                                                
    """
    name: str = None
    contents: Union[Callable, Type, object, str] = None
    parameters: Union[Mapping[str, Any], base.Parameters] = base.Parameters()
    iterations: Union[int, str] = 1
    workflow: Phases.Workflow = Phases.Workflow()
    parallel: ClassVar[bool] = True

    """ Public Methods """
    
    def implement(self, data: Any, **kwargs) -> Any:
        """[summary]

        Args:
            data (Any): [description]

        Returns:
            Any: [description]
        """
                
        return data   
 
    
@dataclasses.dataclass
class Study(ParallelWorker):
    """Allows parallel workflow to continue

    A Study might be wholly passive or implement some reporting or alterations
    to all parallel workflows.
        
    Args:
        name (str): designates the name of a class instance that is used for 
            internal referencing throughout theory. For example, if a theory 
            instance needs options from a Settings instance, 'name' should match 
            the appropriate section name in a Settings instance. Defaults to 
            None. 
        contents (Union[Callable, Type, object, str]): stored item(s) for use by 
            a Process subclass instance. If it is Type or str, an instance 
            will be created. If it is a str, that instance will be drawn from 
            the 'instances' or 'subclasses' attributes.
        parameters (Union[Mapping[str, Any], base.Parameters]): parameters, in 
            the form of an ordinary dict or a Parameters instance, to be 
            attached to 'contents'  when the 'implement' method is called.
            Defaults to an empty Parameters instance.
        iterations (Union[int, str]): number of times the 'implement' method 
            should  be called. If 'iterations' is 'infinite', the 'implement' 
            method will continue indefinitely unless the method stops further 
            iteration. Defaults to 1.
        workflow (Phases.Workflow): a workflow of other theory Processs.
            Defaults to an empty Workflow.
        parallel (ClassVar[bool]): indicates whether this Process design is
            meant to be part of a parallel workflow structure. Because Steps
            are generally part of a parallel-structured workflow, the attribute
            defaults to True.
                                                
    """
    name: str = None
    contents: Union[Callable, Type, object, str] = None
    parameters: Union[Mapping[str, Any], base.Parameters] = base.Parameters()
    iterations: Union[int, str] = 1
    workflow: Phases.Workflow = Phases.Workflow()
    parallel: ClassVar[bool] = True

    """ Public Methods """
    
    def implement(self, data: Any, **kwargs) -> Any:
        """[summary]

        Args:
            data (Any): [description]

        Returns:
            Any: [description]
        """           
        return data    

    
@dataclasses.dataclass
class Survey(ParallelWorker):
    """Resolves a parallel workflow by averaging.

    It resolves a parallel workflow based upon the averaging criteria in 
    'contents'
        
    Args:
        name (str): designates the name of a class instance that is used for 
            internal referencing throughout theory. For example, if a theory 
            instance needs options from a Settings instance, 'name' should match 
            the appropriate section name in a Settings instance. Defaults to 
            None. 
        contents (Union[Callable, Type, object, str]): stored item(s) for use by 
            a Process subclass instance. If it is Type or str, an instance 
            will be created. If it is a str, that instance will be drawn from 
            the 'instances' or 'subclasses' attributes.
        parameters (Union[Mapping[str, Any], base.Parameters]): parameters, in 
            the form of an ordinary dict or a Parameters instance, to be 
            attached to 'contents'  when the 'implement' method is called.
            Defaults to an empty Parameters instance.
        iterations (Union[int, str]): number of times the 'implement' method 
            should  be called. If 'iterations' is 'infinite', the 'implement' 
            method will continue indefinitely unless the method stops further 
            iteration. Defaults to 1.
        workflow (Phases.Workflow): a workflow of other theory Processs.
            Defaults to an empty Workflow.
        parallel (ClassVar[bool]): indicates whether this Process design is
            meant to be part of a parallel workflow structure. Because Steps
            are generally part of a parallel-structured workflow, the attribute
            defaults to True.
                                                
    """
    name: str = None
    contents: Union[Callable, Type, object, str] = None
    parameters: Union[Mapping[str, Any], base.Parameters] = base.Parameters()
    iterations: Union[int, str] = 1
    workflow: Phases.Workflow = Phases.Workflow()
    parallel: ClassVar[bool] = True

    """ Public Methods """
    
    def implement(self, data: Any, **kwargs) -> Any:
        """[summary]

        Args:
            data (Any): [description]

        Returns:
            Any: [description]
        """           
        return data   
    