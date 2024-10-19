import torch
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
import numpy as np


class KMeansSampling(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters, seed):
        self.seed = seed
        self.n_clusters = n_clusters

    def fit(self, X, y=None):
        self.X = X
        self.y = y
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.seed).fit(X)
        return self

    def transform(self, X):
        # Get the indices of the nearest center for each sample
        indices = self.kmeans.predict(X)
        centers = self.kmeans.cluster_centers_
        
        # Select the unique nearest centers
        unique_indices = np.unique(indices)
        selected_samples = centers[unique_indices]
        
        # If y is not None, select corresponding labels
        if self.y is not None:
            selected_labels = self.y[unique_indices]
            return selected_samples, selected_labels, unique_indices
        return selected_samples

# class KCenterGreedySampling(BaseEstimator, TransformerMixin):
#     def __init__(self, budget, metric, random_seed=None, already_selected=None):
#         self.budget = budget
#         self.metric = metric
#         self.random_seed = random_seed
#         self.already_selected = already_selected

#     def fit(self, X, y=None):
#         self.X = X
#         self.y = y
#         # self.X = torch.cat(X, dim=0)
#         self.indices = self.k_center_greedy(self.X, self.budget, self.metric, self.random_seed, already_selected=self.already_selected)
#         return self

#     def transform(self, X, y=None):
#         # Select samples based on the indices from k_center_greedy
#         selected_samples = X[self.indices]

#         # If y is not None, select corresponding labels
#         if self.y is not None:
#             selected_labels = self.y[self.indices]
#             return selected_samples, selected_labels
#         return selected_samples

#     def k_center_greedy(self, matrix, budget: int, metric, random_seed=None, index=None, already_selected=None,
#                         print_freq: int = 20):

#         if type(matrix) == torch.Tensor:
#             assert matrix.dim() == 2
#         elif type(matrix) == np.ndarray:
#             assert matrix.ndim == 2
#             matrix = torch.from_numpy(matrix).requires_grad_(False)

#         sample_num = matrix.shape[0]
#         assert sample_num >= 1
#         if budget < 0:
#             raise ValueError("Illegal budget size.")
#         elif budget > sample_num:
#             budget = sample_num

#         if index is not None:
#             assert matrix.shape[0] == len(index)
#         else:
#             index = np.arange(sample_num)

#         assert callable(metric)

#         already_selected = np.array(already_selected)

#         with torch.no_grad():
#             np.random.seed(random_seed)
#             if already_selected.size == 1:
#                 select_result = np.zeros(sample_num, dtype=bool)
#                 # Randomly select one initial point.
#                 already_selected = [np.random.randint(0, sample_num)]
#                 budget -= 1
#                 select_result[already_selected] = True
#             else:
#                 select_result = np.in1d(index, already_selected)

#             num_of_already_selected = np.sum(select_result)

#             # Initialize a (num_of_already_selected+budget-1)*sample_num matrix storing distances of pool points from
#             # each clustering center.
#             dis_matrix = -1 * torch.ones([num_of_already_selected + budget - 1, sample_num], requires_grad=False)

#             print(matrix[select_result].shape, matrix[~select_result].shape)

#             dis_matrix[:num_of_already_selected, ~select_result] = metric(matrix[select_result], matrix[~select_result])

#             mins = torch.min(dis_matrix[:num_of_already_selected, :], dim=0).values

#             for i in range(budget):
#                 if i % print_freq == 0:
#                     print("| Selecting [%3d/%3d]" % (i + 1, budget))
#                 p = torch.argmax(mins).item()
#                 select_result[p] = True

#                 if i == budget - 1:
#                     break
#                 mins[p] = -1
#                 dis_matrix[num_of_already_selected + i, ~select_result] = metric(matrix[[p]], matrix[~select_result])
#                 mins = torch.min(mins, dis_matrix[num_of_already_selected + i])
#         return index[select_result]

class KCenterGreedySampling(BaseEstimator, TransformerMixin):
    def __init__(self, budget, metric=None, random_seed=None, already_selected=None, verbose=False):
        self.budget = budget
        self.metric = metric
        if random_seed is not None:
            np.random.seed(random_seed)
            torch.manual_seed(random_seed)
        self.already_selected = already_selected if already_selected is not None else []
        self.verbose = verbose

    def fit(self, X, y=None):
        if not callable(self.metric):
            raise ValueError("Metric is not callable")
        if self.budget < 1 or self.budget > len(X):
            raise ValueError("Budget is out of range")
        self.X = X
        self.y = y
        self.indices = self.k_center_greedy(self.X, self.budget, self.metric, self.already_selected, self.verbose)
        return self

    def transform(self, X, y=None):
        selected_samples = X[self.indices]
        return self.X[self.indices], self.y[self.indices], self.indices

    def k_center_greedy(self, matrix, budget, metric, already_selected, verbose):
        if isinstance(matrix, np.ndarray):
            matrix = torch.from_numpy(matrix).float()

        n_samples = matrix.size(0)
        selected = np.zeros(n_samples, dtype=bool)
        selected[already_selected] = True

        if len(already_selected) < 1:
            selected[np.random.randint(0, n_samples)] = True

        for _ in range(min(budget, n_samples - len(already_selected))-1):
            if not selected.any():
                new_index = np.random.randint(0, n_samples)
            else:
                distances = metric(matrix[selected], matrix[~selected])
                # new_index = distances.max(dim=1)[1].item()
                max_values, max_indices = distances.max(dim=1)
                new_index = max_indices[0].item()
            selected[new_index] = True
            if verbose:
                print(f"Selected {np.sum(selected)} / {budget}")

        return np.where(selected)[0]

    