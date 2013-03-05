# -*- coding: utf-8 -*-
import itertools, random

import numpy as np
from scipy import linalg
import pylab as pl
import matplotlib as mpl
import math

epsilon = 10e-8
max_iter = 10000

class Gaussian:
    def __init__(self, X=np.zeros((0,1)), kappa_0=1.0, nu_0=1.0001, mu_0=None, 
            Psi_0=None): # Psi is also called Lambda or T
        # See http://en.wikipedia.org/wiki/Conjugate_prior 
        # Normal-inverse-Wishart conjugate of the Multivariate Normal
        # or see p.18 of Kevin P. Murphy's 2007 paper:"Conjugate Bayesian 
        # analysis of the Gaussian distribution.", in which Psi = T
        self.n_points = X.shape[0]
        self.n_var = X.shape[1]

        if mu_0 == None: # initial mean for the cluster
            self._mu_0 = np.zeros((1, self.n_var))
        else:
            self._mu_0 = mu_0
        assert(self._mu_0.shape == (1, self.n_var))

        self._kappa_0 = kappa_0 # mean fraction

        self._nu_0 = nu_0 # degrees of freedom
        if self._nu_0 < self.n_var:
            self._nu_0 = self.n_var

        if Psi_0 == None:
            self._Psi_0 = np.eye(self.n_var)
        else:
            self._Psi_0 = Psi_0
        assert(self._Psi_0.shape == (self.n_var, self.n_var))

        if X.shape[0] > 0:
            self.fit(X)
        else:
            self.default()


    def default(self):
        self.mean = np.matrix(np.zeros((1, self.n_var))) # TODO init to mean of the dataset
        self.covar = 100.0 * np.matrix(np.eye(self.n_var)) # TODO change 100


    def recompute_ss(self):
        """ need to have actualized _X, _sum, and _square_sum """ 
        self.n_points = self._X.shape[0]
        self.n_var = self._X.shape[1]
        if self.n_points <= 0:
            self.default()
            return

        #print "recomputing suff. stats."
        kappa_n = self._kappa_0 + self.n_points
        nu = self._nu_0 + self.n_points - self.n_var + 1
        mu = np.matrix(self._sum) / self.n_points
        mu_mu_0 = mu - self._mu_0

        C = self._square_sum - self.n_points * (mu.transpose() * mu)
        Psi = self._Psi_0 + C + self._kappa_0 * self.n_points * mu_mu_0.transpose() * mu_mu_0 / (self._kappa_0 + self.n_points)

        self.mean = (self._kappa_0 * self._mu_0 + self.n_points * mu) / (self._kappa_0 + self.n_points)
        self.covar = (Psi * (kappa_n + 1)) / (kappa_n * nu)
        #self.covar = self._square_sum - self.n_points * (
        #        self.mean.transpose() * self.mean) / (self.n_points - 1)
                # unbiased
        #self.covar = (np.matrix(self._X - self.mean).transpose() * np.matrix(self._X - self.mean) / (X.shape[0]-1)) # unbiased
        assert(np.linalg.det(self.covar) != 0)


    def fit(self, X):
        self._X = X
        self._sum = X.sum(0)
        self._square_sum = np.matrix(X).transpose() * np.matrix(X)
        self.recompute_ss()

    
    def add_point(self, x):
        if self.n_points <= 0:
            self._X = np.array([x])
            self._sum = self._X.sum(0)
            self._square_sum = np.matrix(self._X).transpose() * np.matrix(self._X)
        else:
            self._X = np.append(self._X, [x], axis=0)
            self._sum += x
            self._square_sum += np.matrix(x).transpose() * np.matrix(x)
        self.recompute_ss()


    def rm_point(self, x):
        assert(self._X.shape[0] > 0)
        indices = (abs(self._X - x)).argmin(axis=0)
        ind = indices[0,0]
        for ii in indices:
            if (ii-ii[0] == np.zeros(len(ii))).all(): # ensure that all coordinates match
                ind = ii[0,0]
                break
        tmp = np.matrix(self._X[ind])
        self._sum -= self._X[ind]
        self._X = np.delete(self._X, ind, axis=0)
        self._square_sum -= tmp.transpose() * tmp
        self.recompute_ss()


    def pdf(self, x):
        size = len(x)
        #print x
        #print self.mean
        assert(size == self.mean.shape[1])
        assert((size, size) == self.covar.shape)
        det = np.linalg.det(self.covar)
        assert(det != 0)
        norm_const = 1.0 / (math.pow((2*np.pi), float(size)/2) 
                * math.pow(det, 1.0/2))
        x_mu = x - self.mean
        inv = self.covar.I        
        result = math.pow(math.e, -0.5 * (x_mu * inv * x_mu.transpose()))
        return norm_const * result



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
    def _get_means(self):
        return np.array([g.mean for g in self.params])


    def _get_covars(self):
        return np.array([g.covar for g in self.params])


    def __init__(self, n_components=-1, alpha=1.0):
        self.params = [Gaussian()]
        self.n_components = n_components
        self.means_ = self._get_means()
        self.alpha = alpha


#    def fit_Gibbs(self, X):
#        self.z = [random.randint(0, self.n_components - 1) for i in range(len(X))] # or init with k-means
#        for i in range(len(X)): # replace with a while which stops when parameters stop updating
#            # remove X[i]'s sufficient statistics from z[i]
#            for k in range(self.n_components):
#                # compute P_k(X[i]) = P(X[i] | {[X[j] : z[j] = k, j!=i})
#                # compute P(z[i] = k | z[-i], Data) ∝ (N_{k,-i}+α/K)P_k(X[i])
#                pass
#            # sample z[i] ~ P(z[i])
#            # add X[i]'s sufficient statistics to cluster z[i]
#            pass


    def fit_collapsed_Gibbs(self, X):
        mean_data = X.mean(axis=0)
        self.n_points = X.shape[0]
        self.n_var = X.shape[1]
        self._X = X
        if self.n_components == -1:
            # initialize with 1 cluster for each datapoint
            self.params = [Gaussian(X=np.matrix(X[i], mu_0=mean_data)) for i in range(X.shape[0])]
            self.z = [i for i in range(X.shape[0])]
            self.n_components = X.shape[0]
            previous_means = 2 * self._get_means()
            previous_components = self.n_components
        else:
            # init randomly (or with k-means)
            self.params = [Gaussian(X=np.zeros((0, X.shape[1]), mu_0=mean_data)) for i in range(self.n_components)]
            self.z = [random.randint(0, self.n_components - 1) 
                      for i in range(X.shape[0])]
            previous_means = 2 * self._get_means()
            previous_components = self.n_components
            for i in range(X.shape[0]):
                self.params[self.z[i]].add_point(X[i])

        print "Initialized collapsed Gibbs sampling with %i cluster" % (self.n_components)

        n_iter = 0
        # while the clusters did not converge and we still have iter credit
        while (n_iter < max_iter 
                and (previous_components != self.n_components
                or abs((previous_means - self._get_means()).sum()) > epsilon)):
            n_iter += 1
            previous_means = self._get_means()
            previous_components = self.n_components

            for i in range(X.shape[0]):
                # remove X[i]'s sufficient statistics from z[i]
                self.params[self.z[i]].rm_point(X[i])
                # if it empties the cluster, remove it and decrease K
                if self.params[self.z[i]].n_points <= 0:
                    self.params.pop(self.z[i])
                    self.z = map(lambda e: e-1 if e > self.z[i] else e, self.z)
                    self.n_components -= 1

                marginal_likelihood_Xi = {}
                mixing_Xi = {}
                tmp = []
                for k in range(self.n_components):
                    # compute P_k(X[i]) = P(X[i] | X[-i] = k)
                    #print "marginal likelihood for", k, "is"
                    marginal_likelihood_Xi[k] = self.params[k].pdf(X[i])
                    #print marginal_likelihood_Xi[k]
                    # set N_{k,-i} = dim({X[-i] = k})
                    # compute P(z[i] = k | z[-i], Data) = N_{k,-i}/(α+N-1)
                    mixing_Xi[k] = self.params[k].n_points / (self.alpha + self.n_points - 1)
                    tmp.append(marginal_likelihood_Xi[k] * mixing_Xi[k])
                    
                # compute P*(X[i]) = P(X[i]|λ)
                base_distrib = Gaussian(X=np.zeros((0, X.shape[1])))
                #print "prior predictive is"
                prior_predictive = base_distrib.pdf(X[i])
                #print prior_predictive
                # compute P(z[i] = * | z[-i], Data) = α/(α+N-1)
                prob_new_cluster = self.alpha / (self.alpha + self.n_points - 1)
                tmp.append(prior_predictive * prob_new_cluster)

                # normalize P(z[i]) (tmp above)
                s = sum(tmp)
                tmp = map(lambda e: e/s, tmp)

                # sample z[i] ~ P(z[i])
                rdm = np.random.rand()
                total = tmp[0]
                k = 0
                while (rdm > total):
                    k += 1
                    total += tmp[k]
                # add X[i]'s sufficient statistics to cluster z[i]
                if k == self.n_components: # create a new cluster
                    self.params.append(base_distrib)
                    self.n_components += 1
                else:
                    assert(k < self.n_components)
                self.params[k].add_point(X[i])

            print "still sampling, %i clusters currently, with log-likelihood %f" % (self.n_components, self.log_likelihood())

        self.means_ = self._get_means()


    def predict(self, X):
        if (X != self._X).any():
            self.fit_collapsed_Gibbs(X)
        #Y = np.zero(X.shape[0], dtype='uint8')
        Y = np.array([self.z[i] for i in range(X.shape[0])])
        return Y


    def log_likelihood(self):
        log_likelihood = 0.
        for n in xrange(self.n_points):
            print self.z
            print n, len(self.z)
            print self.z[n]
            print len(self.params)
            log_likelihood -= 0.5 * self.n_var * np.log(2.0 * np.pi) + 0.5 * np.log(np.linalg.det(
                self.params[self.z[n]].covar))
            #log_likelihood -= 0.5 * 
        return log_likelihood




# Number of samples per component
n_samples = 50

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

dpmm = DPMM(n_components=-1)
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

