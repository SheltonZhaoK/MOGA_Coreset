import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from .snnls import SparseNNLS


class ImportanceSampling(SparseNNLS, BaseEstimator, TransformerMixin):
  def __init__(self, A, b, iterations):
    super().__init__(A, b)
    self.cts = np.zeros(self.w.shape[0]) 
    self.ps = np.sqrt((self.A**2).sum(axis=0))
    if np.any(self.ps > 0):
      self.ps /= self.ps.sum()
    else:
      self.ps = np.ones(self.w.shape[0]) / float(self.w.shape[0])
    self.check_error_monotone = False
    self.iterations = iterations

  def reset(self):
    super().reset()
    self.cts = np.zeros(self.w.shape[0]) 

  def _compute_sampling_probabilities(self):
    self.ps = np.sqrt((self.A**2).sum(axis=0))
    if np.any(self.ps > 0):
      self.ps / self.ps.sum()

  def _select(self):
    return np.random.choice(self.ps.shape[0], p = self.ps)

  def _reweight(self, f):
    self.cts[f] += 1
    self.w = (self.cts / self.cts.sum()) / self.ps 

  def fit(self, X, y=None):
    self.X = X
    self.y = y
    return self
    
  def transform(self, X, y=None):
    self.build(self.iterations)
    coreset_indices = np.nonzero(self.w > 0)[0]
    return self.X[coreset_indices], self.y[coreset_indices], coreset_indices

class UniformSampling(BaseEstimator, TransformerMixin):
    def __init__(self, sample_size=None):
        self.sample_size = sample_size

    def fit(self, X, y=None):
        self.X = X
        self.y = y
        return self

    def transform(self, X):
        # Generate random indices
        indices = np.random.choice(X.shape[0], self.sample_size, replace=False)
        return self.X[indices], self.y[indices], indices

class BalancedUniformSampling(BaseEstimator, TransformerMixin):
    def __init__(self, sample_size=None):
        self.sample_size = sample_size

    def fit(self, X, y=None):
        self.X = X
        self.y = y
        return self

    def transform(self, X):
        unique_classes, class_counts = np.unique(self.y, return_counts=True)
        samples_per_class = max(1, self.sample_size // len(unique_classes))

        sampled_indices = []
        for cls in unique_classes:
            class_indices = np.where(self.y == cls)[0]
            sampled_class_indices = np.random.choice(class_indices, min(samples_per_class, len(class_indices)), replace=False)
            sampled_indices.extend(sampled_class_indices)

        # Adjust in case total sample size is less than requested due to class size limitations
        sampled_indices = np.random.choice(sampled_indices, self.sample_size, replace=False)
        return self.X[sampled_indices], self.y[sampled_indices], sampled_indices