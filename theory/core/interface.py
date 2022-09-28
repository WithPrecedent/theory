"""
interface: theory project interface
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
    Project (base.Theory): main access point and interface for
        creating and implementing data science projects.
    
"""
from __future__ import annotations
import dataclasses
import pathlib
from typing import (
    Any, Callable, ClassVar, Dict, Iterable, List, Mapping, 
    Optional, Sequence, Tuple, Type, Union)

import amos
import chrisjen
import numpy as np
import pandas as pd

from . import base
from . import stages

