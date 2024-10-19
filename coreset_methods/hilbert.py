from sklearn.base import BaseEstimator, TransformerMixin
import bayesiancoresets as bc

class IDProjector(bc.Projector):
    def update(self, wts, pts):
        pass
    def project(self, pts, grad=False):
        return pts

class Hilbert(BaseEstimator, TransformerMixin):
    def __init__(self, X, iterations):
        self.iterations = iterations
        self.alg = bc.HilbertCoreset(X, IDProjector(), snnls = bc.snnls.OrthoPursuit)

    def fit(self, X, y=None):
        self.X = X
        self.y = y
        return self
    
    def transform(self, X):
        self.alg.build(self.iterations) 
        indices = vars(self.alg)["idcs"]
        return self.X[indices], self.y[indices], indices