"""
criteria: algorithms for assessing models and other techniques
Corey Rayburn Yung <coreyrayburnyung@gmail.com>
Copyright 2020, Corey Rayburn Yung
License: Apache-2.0 (https://www.apache.org/licenses/LICENSE-2.0)

Contents:

    
"""
from __future__ import annotations
import dataclasses
from typing import (Any, Callable, ClassVar, Dict, Iterable, List, Mapping, 
                    Optional, Sequence, Tuple, Type, Union)

from . import base 


@dataclasses.dataclass
class Accuracy(base.TheoryCriteria):
    
    name: str = 'accuracy'
    module: str = 'sklearn.metrics'
    contents: str = 'accuracy_score'
        
        
@dataclasses.dataclass
class AdjustedMutualInfo(base.TheoryCriteria):
    
    name: str = 'adjusted_mutual_info_score'
    module: str = 'sklearn.metrics'
    contents: str = 'adjusted_mutual_info'
    

@dataclasses.dataclass
class AdjustedRand(base.TheoryCriteria):
    
    name: str = 'adjusted_rand'
    module: str = 'sklearn.metrics'
    contents: str = 'adjusted_rand_score'
    
    
@dataclasses.dataclass
class BalancedAccuracy(base.TheoryCriteria):
    
    name: str = 'balanced_accuracy'
    module: str = 'sklearn.metrics'
    contents: str = 'balanced_accuracy_score'
    
    
@dataclasses.dataclass
class BrierScoreLoss(base.TheoryCriteria):
    
    name: str = 'brier_score_loss'
    module: str = 'sklearn.metrics'
    contents: str = 'brier_score_loss'
    
    
@dataclasses.dataclass
class Calinski(base.TheoryCriteria):

    name: str = 'calinski_harabasz'
    module: str = 'sklearn.metrics'
    contents: str = 'calinski_harabasz_score'
    
    
@dataclasses.dataclass
class DaviesBouldin(base.TheoryCriteria):

    name: str = 'davies_bouldin'
    module: str = 'sklearn.metrics'
    contents: str = 'davies_bouldin_score'
    
    
@dataclasses.dataclass
class Completeness(base.TheoryCriteria):

    name: str = 'completeness'
    module: str = 'sklearn.metrics'
    contents: str = 'completeness_score'
    
    
@dataclasses.dataclass
class ContingencyMatrix(base.TheoryCriteria):

    name: str = 'contingency_matrix'
    module: str = 'sklearn.metrics'
    contents: str = 'cluster.contingency_matrix'
    
    
@dataclasses.dataclass
class ExplainedVariance(base.TheoryCriteria):

    name: str = 'explained_variance'
    module: str = 'sklearn.metrics'
    contents: str = 'explained_variance_score'
    
    
@dataclasses.dataclass
class F1(base.TheoryCriteria):

    name: str = 'f1'
    module: str = 'sklearn.metrics'
    contents: str = 'f1_score'
    
    
@dataclasses.dataclass
class F1Weighted(base.TheoryCriteria):

    name: str = 'f1_weighted'
    module: str = 'sklearn.metrics'
    contents: str = 'f1_score'
    required = {'average': 'weighted'}
    
    
@dataclasses.dataclass
class Fbeta(base.TheoryCriteria):

    name: str = 'fbeta'
    module: str = 'sklearn.metrics'
    contents: str = 'fbeta_score'
    required = {'beta': 1}),
    
    
@dataclasses.dataclass
class FowlkesMallows(base.TheoryCriteria):

    name: str = 'fowlkes_mallows'
    module: str = 'sklearn.metrics'
    contents: str = 'fowlkes_mallows_score'
    
    
@dataclasses.dataclass
class Hamming(base.TheoryCriteria):

    name: str = 'hamming_loss'
    module: str = 'sklearn.metrics'
    contents: str = 'hamming_loss'
    
    
@dataclasses.dataclass
class HomogeneityCompleteness(base.TheoryCriteria):

    name: str = 'homogeneity_completeness'
    module: str = 'sklearn.metrics'
    contents: str = 'homogeneity_completeness_v_measure'
    
    
@dataclasses.dataclass
class Homogeneity(base.TheoryCriteria):

    name: str = 'homogeneity'
    module: str = 'sklearn.metrics'
    contents: str = 'homogeneity_score'
    
    
@dataclasses.dataclass
class Jaccard(base.TheoryCriteria):

    name: str = 'jaccard_similarity'
    module: str = 'sklearn.metrics'
    contents: str = 'jaccard_similarity_score'
    
    
@dataclasses.dataclass
class MedianAbsoluteError(base.TheoryCriteria):

    name: str = 'median_absolute_error'
    module: str = 'sklearn.metrics'
    contents: str = 'median_absolute_error'
    
    
@dataclasses.dataclass
class MatthewsCoefficient(base.TheoryCriteria):

    name: str = 'matthews_correlation_coefficient'
    module: str = 'sklearn.metrics'
    contents: str = 'matthews_corrcoef'
    
    
@dataclasses.dataclass
class MaxError(base.TheoryCriteria):

    name: str = 'max_error'
    module: str = 'sklearn.metrics'
    contents: str = 'max_error'
    
    
@dataclasses.dataclass
class MeanAbsoluteError(base.TheoryCriteria):

    name: str = 'mean_absolute_error'
    module: str = 'sklearn.metrics'
    contents: str = 'mean_absolute_error'
    
      
@dataclasses.dataclass
class MeanSquaredError(base.TheoryCriteria):

    name: str = 'mean_squared_error'
    module: str = 'sklearn.metrics'
    contents: str = 'mean_squared_error'
    
    
@dataclasses.dataclass
class MeanSquaredLogError(base.TheoryCriteria):

    name: str = 'mean_squared_log_error'
    module: str = 'sklearn.metrics'
    contents: str = 'mean_squared_log_error'
    
    
@dataclasses.dataclass
class MutualInfoScore(base.TheoryCriteria):

    name: str = 'mutual_info_score'
    module: str = 'sklearn.metrics'
    contents: str = 'mutual_info_score'
    
    
@dataclasses.dataclass
class LogLoss(base.TheoryCriteria):

    name: str = 'log_loss'
    module: str = 'sklearn.metrics'
    contents: str = 'log_loss'
    
    
@dataclasses.dataclass
class NormalizedMutualInfo(base.TheoryCriteria):

    name: str = 'normalized_mutual_info'
    module: str = 'sklearn.metrics'
    contents: str = 'normalized_mutual_info_score'
    
    
@dataclasses.dataclass
class Precision(base.TheoryCriteria):

    name: str = 'precision'
    module: str = 'sklearn.metrics'
    contents: str = 'precision_score'
    
    
@dataclasses.dataclass
class PrecisionWeighted(base.TheoryCriteria):

    name: str = 'precision_weighted'
    module: str = 'sklearn.metrics'
    contents: str = 'precision_score'
    required = {'average': 'weighted'}
    
    
@dataclasses.dataclass
class R2(base.TheoryCriteria):

    name: str = 'r2'
    module: str = 'sklearn.metrics'
    contents: str = 'r2_score'
    
    
@dataclasses.dataclass
class Recall(base.TheoryCriteria):

    name: str = 'recall'
    module: str = 'sklearn.metrics'
    contents: str = 'recall_score'
    
    
@dataclasses.dataclass
class RecallWeighted(base.TheoryCriteria):

    name: str = 'recall_weighted'
    module: str = 'sklearn.metrics'
    contents: str = 'recall_score'
    required = {'average': 'weighted'}
    
    
@dataclasses.dataclass
class ROCAUC(base.TheoryCriteria):

    name: str = 'roc_auc'
    module: str = 'sklearn.metrics'
    contents: str = 'roc_auc_score'
    
    
@dataclasses.dataclass
class Silhouette(base.TheoryCriteria):

    name: str = 'silhouette'
    module: str = 'sklearn.metrics'
    contents: str = 'silhouette_score'
    
    
@dataclasses.dataclass
class VMeasure(base.TheoryCriteria):

    name: str = 'v_measure'
    module: str = 'sklearn.metrics'
    contents: str = 'v_measure_score'
    
    
@dataclasses.dataclass
class ZeroOne(base.TheoryCriteria):

    name: str = 'zero_one'
    module: str = 'sklearn.metrics'
    contents: str = 'zero_one_loss'
