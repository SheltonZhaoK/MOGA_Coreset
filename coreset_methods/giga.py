import numpy as np
from .util.errors import NumericalPrecisionError
from sklearn.base import BaseEstimator, TransformerMixin
from . import util
from .snnls import SparseNNLS

class GIGA(SparseNNLS, BaseEstimator, TransformerMixin):

  def __init__(self, A, b, iterations):
    self.iterations = iterations
    super().__init__(A, b)
    Anorms = np.sqrt((self.A**2).sum(axis=0))
    if np.any( Anorms == 0):
      raise ValueError(self.alg_name+'.__init__(): A must not have any 0 columns')
    self.An = self.A / Anorms

    self.bnorm = np.sqrt(((self.b)**2).sum())
    if self.bnorm == 0.:
      raise NumericalPrecisionError('norm of b must be > 0')
    self.bn = self.b / self.bnorm

  def _select(self):
    xw = self.A.dot(self.w)
    nw = np.sqrt(((xw)**2).sum())
    nw = 1. if nw == 0. else nw
    xw /= nw

    
    cdir = self.bn.flatten() - self.bn.flatten().dot(xw)*xw
    cdirnrm =np.sqrt((cdir**2).sum()) 
    if cdirnrm < util.TOL:
      raise NumericalPrecisionError('cdirnrm < TOL: cdirnrm = ' + str(cdirnrm))
    cdir /= cdirnrm
    scorends = self.An.T.dot(np.hstack((cdir[:,np.newaxis], xw[:,np.newaxis]))) 
    #extract points for which the geodesic direction is stable (1st condition) and well defined (2nd)
    idcs = np.logical_and(scorends[:,1] > -1.+1e-14,  1.-scorends[:,1]**2 > 0.)
    #compute the norm 
    scorends[idcs, 1] = np.sqrt(1.-scorends[idcs,1]**2)
    scorends[np.logical_not(idcs),1] = np.inf
    #compute the scores and argmax
    return (scorends[:,0]/scorends[:,1]).argmax()
 
  def _reweight(self, f):
    xw = self.A.dot(self.w)
    nw = np.sqrt((xw**2).sum())
    nw = 1. if nw == 0. else nw
    xf = self.A[:, f]
    nf = np.sqrt((xf**2).sum())

    gA = self.bn.dot((xf/nf)) - self.bn.dot((xw/nw)) * (xw/nw).dot((xf/nf))
    gB = self.bn.dot((xw/nw)) - self.bn.dot((xf/nf)) * (xw/nw).dot((xf/nf))
    if gA <= 0. or gB < 0:
      raise NumericalPrecisionError

    a = gB/(gA+gB)/nw
    b = gA/(gA+gB)/nf
    
    x = a*xw + b*xf
    nx = np.sqrt((x**2).sum())
    scale = self.bnorm/nx*(x/nx).dot(self.bn)
    
    alpha = a*scale
    beta = b*scale
     
    self.w = alpha*self.w
    self.w[f] = max(0., self.w[f]+beta)

  def fit(self, X, y=None):
    self.X = X
    self.y = y
    return self
    
  def transform(self, X, y=None):
    self.build(self.iterations)
    coreset_indices = np.nonzero(self.w > 0)[0]
    return self.X[coreset_indices], self.y[coreset_indices], coreset_indices