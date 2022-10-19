# Contribution Guidelines

To contribute to theory, please follow these basic rules:

## Style

1. The project generally follows the Google Style Guide for python (https://google.github.io/styleguide/pyguide.html) with two 
   notable exceptions:
   * I always add spaces around '='. This is because I find it more readable  and it is practically the norm with type annotations adding the spaces to function and method signatures. I realize that this will seem alien to many coders, but it is far easier on my eyes. 
   * I've expanded the Google exception for importing multiple items from one package from just 'typing' to also include 'collections.abc'. This is because, as of python 3.9, many of the type annotations in 'typing' are being depreciated and have already been combined with the similarly named types in 'collections.abc'. I except Google will make this change at some point in the near future. 
  It is particularly important for contributions to follow the Google style for docstrings so that sphinx napoleon can automatically incorporate the docstrings into online documentation.

2. Explicitness preferences are heightened beyond PEP8 guidelines. Varible names should be verbose enough so that their meaning is clear and consistent. Type annotations should always be used in arguments and docstrings. As theory is intended to be used by all levels of coders (and by non-coders as well), it is important to make everything as clear as possible to someone seeing the code for the first time. 

3. theory generally follows an object-oriented approach because that makes integration with scikit-learn and general modularity easier. Contributions are not precluded from using other programming styles, but class wrappers might be needed to interface properly with the overall theory structure. 

## Structure

1. Any new subpackages should follow a similar template to existing ones. All classes within theory should use the @dataclasses.dataclass decorator (introduced in python 3.7) to minimize boilerplate code and make classes easier to read.

2. theory lazily (runtime) loads most external and internal modules. This is done to lower overhead and incorporate "soft" dependencies. 
   
3. theory favors coomposition over inheritance and makes extensive use of the composite and builder design patterns. Inheritance is used, and only allowed from the abstract base classes that define a particular grouping of classes. For example, the Book, Chapter, and Technique classes inherit from Manuscript to allow for sharing of common methods.

4. When composing objects through a loosely coupled hierarchy, it is often important to provide connections in both directions. 

5. All file management should be perfomed throught the shared Clerk instance.

6. All external settings should be imported and constructed using the shared Idea instance. 

7. All external data should be contained in instances of Dataset. Before beginning the processes in Analyst, ideally, there should be a single, combined pandas DataFrame stored in the Dataset instance.

## General

1. When in doubt, copy. All of the core subpackages follow these rules. If you are starting a new object in theory, the safest route is just to copy an analagous class (and related import statements) into a new module and go from there.

2. Add any new soft or hard dependencies to the pyproject.toml file in the root folder of the package. Even though there is a risk to the approach, theory favors importation over integration of open-source code. This allows updates to those external dependencies to be seamlessly added into a theory workflow. This can create problems when constructing virtual python environments, but, absent special circumstances, importatiion is preferred.

3. If you have a great idea that is inconsistent with these guidelines, email the maintainer directly. We are always looking for ways to improve theory and are open to amending or discarding various contribution guidelines if they are stifling innovation.