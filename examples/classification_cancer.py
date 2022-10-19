"""
wisconsin breast cancer classification example
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


To Do:

            
"""
import pathlib

import pandas as pd
import numpy as np
import sklearn.datasets

from theory import Theory


# Loads cancer data and converts from numpy arrays to a pandas DataFrame.
cancer = sklearn.datasets.load_breast_cancer()
df = pd.DataFrame(
    data = np.c_[cancer['data'], cancer['target']],
    columns = np.append(cancer['feature_names'], ['target']))
# Sets root_folder for data and results exports.
# root_folder = os.path.join('..', '..')
# Sets location of configuration settings for the project. Depending upon your
# OS and python configuration, one of these might work better.
idea = pathlib.Path.cwd().joinpath('examples', 'cancer_settings.ini')
#idea = os.path.join(os.getcwd(), 'cancer_settings.ini')

# Creates theory project, automatically configuring the process based upon
# settings in the 'idea_file'.
cancer_theory = Theory(
    idea = idea,
    # clerk = root_folder,
    dataset = df)
# Converts label to boolean type to correct numpy default above.
cancer_theory.dataset.change_datatype(
    columns = 'target',
    datatype = 'boolean')
# Fills missing data with appropriate default values based on column datatype.
# cancer_project.dataset.smart_fill()
# Iterates through every recipe and exports plots, explainers, and other
# metrics from each recipe.
cancer_theory.apply()
# Outputs information about the best recipe to the terminal.
# cancer_theory['critic'].print_best()
# Saves dataset file with predictions or predicted probabilities added
# (based on options in idea).
# cancer_theory.dataset.save(file_name = 'cancer_df')