# -*- coding: utf-8 -*-
import itertools, random

import numpy as np
from scipy import linalg
import pylab as pl
import matplotlib as mpl

class Gaussian:
    def __init__(self, n_var=1):
        self.n_points = 0
        self.n_var = n_var
        self.mean = np.array([.0 for i in range(self.n_var)])
        self.covar = np.matrix(self.mean).transpose() * np.matrix(self.mean) + 100

    def fit(self, X):
        self.n_points = X.shape[0]
        self.mean = X.mean(0)
        self.n_var = self.mean.shape[0]
        self.X_ = X - self.mean
        self.covar = (np.matrix(self.X_).transpose() * np.matrix(self.X_) 
                     / (X.shape[0]-1)) # unbiased

    
    def add_point(self, x):
        oldmean = self.mean
        self.mean = (self.n_points * self.mean + x) / (self.n_points + 1)
        self.covar = ((self.n_points - 1) * self.covar 
            + (x - self.mean).transpose() * (x - oldmean)) / self.n_points
        self.n_points += 1


    def rm_point(self, x):



"""
Dirichlet process mixture model (for N observations y_1, ..., y_N)
    1) generate a distribution G ~ DP(G_0, α)
    2) generate parameters θ_1, ..., θ_N ~ G
    [1+2) <=> (with B_1, ..., B_N a measurable partition of the set for which 
        G_0 is a finite measure, G(B_i) = θ_i:)
       generate G(B_1), ..., G(B_N) ~ Dirichlet(αG_0(B_1), ..., αG_0(B_N)]
    3) generate each datapoint y_i ~ F(θ_i)
Now, an alternative is:
    1) generate a vector β ~ Stick(1, α) (<=> GEM(1, α))
    2) generate cluster assignments c_i ~ Categorical(β) (gives K clusters)
    3) generate parameters Φ_1, ...,Φ_K ~ G_0
    4) generate each datapoint y_i ~ F(Φ_{c_i})
    for instance F is a Gaussian and Φ_c = (mean_c, var_c)
Another one is:
    1) generate cluster assignments c_1, ..., c_N ~ CRP(N, α) (K clusters)
    2) generate parameters Φ_1, ...,Φ_K ~ G_0
    3) generate each datapoint y_i ~ F(Φ_{c_i})

So we have P(y | Φ_{1:K}, β_{1:K}) = \sum_{j=1}^K β_j Norm(y | μ_j, S_j)
"""
class DPMM:
    def __init__(self, n_components=10):
        self.n_components = n_components
        self.params = [(0.0, 100.0) for i in range(self.n_components)]

    def fit_Gibbs(self, X):
        self.z = [random.randint(0, self.n_components - 1) for i in range(len(X))] # or init with k-means
        for i in range(len(X)): # replace with a while which stops when parameters stop updating
            # remove X[i]'s sufficient statistics from z[i]
            for k in range(self.n_components):
                # compute P_k(X[i]) = P(X[i] | {[X[j] : z[j] = k, j!=i})
                # compute P(z[i] = k | z[-i], Data) ∝ (N_{k,-i}+α/K)P_k(X[i])
                pass
            # sample z[i] ~ P(z[i])
            # add X[i]'s sufficient statistics to cluster z[i]
            pass

    def fit_collapsed_Gibbs(self, X):
        self.z = [random.randint(1, self.n_components) for i in range(len(X))] # or init with k-means
        for i in range(len(X)): # replace with a while which stops when parameters stop updating
            # remove X[i]'s sufficient statistics from z[i]
            for k in range(self.n_components):
                # compute P_k(X[i]) = P(X[i] | X[-i] = k)
                # set N_{k,-i} = dim({X[-i] = k})
                # compute P(z[i] = k | z[-i], Data) = N_{k,-i}/(α+N-1)
                pass
            # compute P*(X[i]) = P(X[i]|λ)
            # compute P(z[i] = * | z[-i], Data) = α/(α+N-1)
            # normalize P(z[i]) (above)
            # sample z[i] ~ P(z[i])
            # add X[i]'s sufficient statistics to cluster z[i]
            # if any cluster is empty, remove it and decrease K


# Number of samples per component
n_samples = 500

# Generate random sample, two components
np.random.seed(0)
C = np.array([[0., -0.1], [1.7, .4]])
X = np.r_[np.dot(np.random.randn(n_samples, 2), C),
          .7 * np.random.randn(n_samples, 2) + np.array([-6, 3])]

from sklearn import mixture

# Fit a mixture of gaussians with EM using five components
gmm = mixture.GMM(n_components=5, covariance_type='full')
gmm.fit(X)

# Fit a dirichlet process mixture of gaussians using five components
dpgmm = mixture.DPGMM(n_components=5, covariance_type='full')
dpgmm.fit(X)

dpmm = DPMM(n_components=5)
dpmm.fit_collapsed_Gibbs(X)

color_iter = itertools.cycle(['r', 'g', 'b', 'c', 'm'])

for i, (clf, title) in enumerate([(gmm, 'GMM'),
                                  (dpgmm, 'Dirichlet Process GMM (sklearn, Variational)'),
                                  (dpmm, 'Dirichlet Process GMM (ours, Gibbs)')]):
    splot = pl.subplot(3, 1, 1 + i)
    Y_ = clf.predict(X)
    for i, (mean, covar, color) in enumerate(zip(
            clf.means_, clf._get_covars(), color_iter)):
        v, w = linalg.eigh(covar)
        u = w[0] / linalg.norm(w[0])
        # as the DP will not use every component it has access to
        # unless it needs it, we shouldn't plot the redundant
        # components.
        if not np.any(Y_ == i):
            continue
        pl.scatter(X[Y_ == i, 0], X[Y_ == i, 1], .8, color=color)

        # Plot an ellipse to show the Gaussian component
        angle = np.arctan(u[1] / u[0])
        angle = 180 * angle / np.pi  # convert to degrees
        ell = mpl.patches.Ellipse(mean, v[0], v[1], 180 + angle, color=color)
        ell.set_clip_box(splot.bbox)
        ell.set_alpha(0.5)
        splot.add_artist(ell)

    pl.xlim(-10, 10)
    pl.ylim(-3, 6)
    pl.xticks(())
    pl.yticks(())
    pl.title(title)

pl.savefig('dpgmm.png')



#def normal_distrib(

#def dirichlet_process_mixture_model(base_distrib, alpha):
#chinese_restaurant_process(N, alpha)

