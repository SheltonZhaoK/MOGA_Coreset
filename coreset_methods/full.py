from sklearn.base import BaseEstimator, TransformerMixin
class Full(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        self.X = X
        self.y = y
        return self
    
    def transform(self, X):
        return self.X, self.y, list(range(len(X)))
