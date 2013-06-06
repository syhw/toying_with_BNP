# -*- coding: utf-8 -*-
import copy
from itertools import izip, cycle
import numpy as np
from scipy import linalg
import pylab as pl
import matplotlib as mpl
from scipy.special import gamma, digamma

max_iter = 500
epsilon = 10e-8


class VB_GMM:
    def _get_means(self):
        return self.m


    def _get_covars(self):
        return np.array([linalg.inv(
            ((self.nu[k] + 1.0 - self.n_var)*self.beta[k]
            /(1.0 + self.beta[k])) * self.W[k])
            for k in xrange(self.n_components)])


    def __init__(self, n_components=-1):
        self.n_components = n_components


    def fit_variational(self, X, alpha_0=0.01, beta_0=1.0, nu_0=1.0, W_0=10.0):
        self.n_points = X.shape[0]
        self.n_var = X.shape[1]
        self._X = X
        self.alpha_0 = alpha_0
        self.beta_0 = beta_0
        self.nu_0 = nu_0
        self.W_0 = W_0 * np.eye(self.n_var)
        if self.n_components == -1:
            # initialize with 1 cluster for each datapoint
            self.n_components = X.shape[0]
            # TODO
        else:
            # init randomly (or with k-means)
            from sklearn.cluster import k_means
            self.m_0 = k_means(self._X, self.n_components)[0] # float ndarray with shape (k, n_features)
        self.r = np.array([[-1.0 for i in xrange(self.n_components)] for j in xrange(self.n_points)])

        #self.alpha = np.array([self.alpha_0 for i in xrange(self.n_components)])
        self.alpha = np.array([(i+1)*self.alpha_0 for i in xrange(self.n_components)]) # non symmetric prior
        #self.alpha = np.array([self.alpha_0, self.alpha_0] + [0.01*self.alpha_0 for i in xrange(self.n_components-2)]) # non symmetric prior
        self.beta = np.array([self.beta_0 for i in xrange(self.n_components)])
        self.nu = np.array([self.nu_0 for i in xrange(self.n_components)])
        self.W = np.array([self.W_0 for i in xrange(self.n_components)])
        self.C_alpha_0 = gamma(self.alpha.sum()) / reduce(lambda x,y: x*y, 
                map(gamma, self.alpha))

        tmp_pi = self.alpha.sum()*1.0/self.alpha.shape[0] # check that TODO
        self.B_W0_nu0 = (linalg.det(self.W_0)**(-self.nu_0/2)*
                (2**(self.nu_0*self.n_var/2) * tmp_pi**(
                    self.n_var*(self.n_var-1)/4.) * reduce(lambda x,y: x*y, 
                        map(gamma, [(self.nu_0 + 2 - i )/2. 
                            for i in xrange(self.n_var)])))**(-1))


        self.m = copy.deepcopy(self.m_0)
        self.pi = np.array([0.0 for i in xrange(self.n_components)])
        self.Lambda = np.array([0.0 for i in xrange(self.n_components)])

        self.S = copy.deepcopy(self.W)
        self.N = np.array([0.0 for i in xrange(self.n_components)])
        self.x_ = copy.deepcopy(self.m_0)

        print "Initialized variational GMM with %i cluster" % (self.n_components)

        n_iter = 0 # with max_iter hard limit, in case of cluster oscillations
        # while the clusters did not converge (i.e. the number of components or
        # the means of the components changed) and we still have iter credit
        previous_log_likelihood = 10e15 # arbitrary and useless
        current_log_likelihood = 10e14 # arbitrary and useless
        while (previous_log_likelihood - current_log_likelihood > epsilon 
                and n_iter < max_iter):
            previous_log_likelihood = current_log_likelihood
            n_iter += 1

            # E-step
            for j in xrange(self.n_components):
                self.pi[j] = np.exp(digamma(self.alpha[j]) - digamma(
                    self.alpha.sum()))
                self.Lambda[j] = (2**self.n_var) * linalg.det(self.W[j])
                for d in xrange(self.n_var):
                    self.Lambda[j] *= np.exp(digamma(
                        (self.nu[j] + 1.0 - d)/2.0))
            for i in xrange(X.shape[0]):
                for j in xrange(self.n_components):
                    X_minus_m = X[i]-self.m[j]
                    self.r[i,j] = self.pi[j] * (self.Lambda[j]**0.5) * np.exp(
                        -self.n_var/(2*self.beta[j]) - (self.nu[j]/2)*np.dot(
                                X_minus_m.T, np.dot(self.W[j], X_minus_m)))
                self.r[i,:] /= self.r[i,:].sum() + epsilon

            # M-step
            for j in xrange(self.n_components):
                self.N[j] = self.r[:,j].sum()
                self.x_[j] = 1.0/(epsilon + self.N[j]) * np.dot(self.r[:,j], X)
                tmp_1 = (X - self.x_[j])
                tmp_2 = (self.r[:,j] * tmp_1.T).T
                self.S[j] = 1.0/(epsilon + self.N[j]) * np.dot(tmp_1.T, tmp_2)
                #self.alpha[j] = self.alpha_0 + self.N[j] # TODO
                self.beta[j] = self.beta_0 + self.N[j] 
                self.m[j] = 1.0/self.beta[j] * (self.beta_0 * self.m_0[j]
                        + self.N[j] * self.x_[j])
                tmp_ = (self.x_[j] - self.m_0[j])
                self.W[j] = (linalg.inv(self.W_0) + self.N[j]*self.S[j]
                        + (self.beta_0 * self.N[j])/(self.beta_0 + self.N[j])
                        * np.dot(tmp_.T, tmp_))
                self.nu[j] = self.nu_0 + self.N[j]
            current_log_likelihood = self.log_likelihood()
            print ("still learning, %i clusters currently, log-likelihood %f" 
                    % (self.n_components, current_log_likelihood))

        # check for empty components
        for k in xrange(self.n_components):
            print self.W[k]
            tmp = ((self.nu[k] + 1.0 - self.n_var)*self.beta[k]
            /(1.0 + self.beta[k])) * self.W[k]
            print tmp
            if linalg.det(tmp) < epsilon: # not invertible, i.e. singular 
                self.m = np.delete(self.m, k)
                self.W = np.delete(self.W, k)

        self.means_ = self._get_means()



    def predict(self, X):
        """ produces and returns the clustering of the X data """
        if (X != self._X).any():
            self.fit_variational(X)
        #Y = np.array([mapper.index(self.z[i]) for i in range(X.shape[0])])
        Y = self.r.argmax(axis=1) # TODO as commented above (order in indices)
        return Y


    def log_likelihood(self):
        # L = \sum_Z \int \int \int q(Z,pi,mu,Lambda)ln{p(X,Z,pi,my,Lambda)/
        #                           q(Z,pi,my,Lambda)} dpi dmu dLambda
        #  = E[ln p(X|Z,mu,Lambda)] + E[ln p(Z|pi)] + E[ln p(pi)]
        #  + E[ln p(mu,Lambda)] - E[ln q(Z)] - E[ln q(pi)] - E[ln q(mu,Lambda)]
        # C.f. Bishop's PRML p.481-482 or Thomas' handout p.3
        d = self.n_var
        C_alpha = gamma(self.alpha.sum()) / reduce(lambda x,y: x*y, 
                map(gamma, self.alpha))
        s = self.C_alpha_0 / C_alpha
        for k, (Wk, nuk, pik) in enumerate(izip(self.W, self.nu, self.pi)):
            print nuk
            print map(gamma, [(nuk + 2 - i )/2. 
                        for i in xrange(d)])
            B_W_nu = linalg.det(Wk)**(-nuk/2)*(2**(nuk*d/2) * pik**(d*(d-1)/4.)
                    * reduce(lambda x,y: x*y, map(gamma, [(nuk + 2 - i )/2. 
                        for i in xrange(d)])) + epsilon)**(-1)
            tmp_v = self.x_[k] - self.m[k]
            s += 0.5 * self.N[k] * (np.log(self.Lambda[k]) 
                    - self.n_var * self.beta[k]**(-1) - nuk * np.trace(np.dot(
                        self.S[k], self.W[k])) - nuk * np.dot(tmp_v, np.dot(
                            self.W[k], tmp_v)) - d * np.log(2*np.pi))
            s += np.log(self.pi[k]**(self.alpha_0 - self.alpha[k])) + np.log(
                    self.B_W0_nu0*1.0/B_W_nu) + sum(map(lambda x: x*np.log(
                        (self.pi[k]+epsilon)/(x+epsilon)), self.r[:,k])) + 0.5*np.log(
                    self.Lambda[k]**(self.nu_0 - nuk)) + d/2. * (np.log(
                    self.beta_0 / self.beta[k]) + (1 - 
                        self.beta_0/self.beta[k]) + nuk)
            tmp_v = self.m[k] - self.m_0[k]
            s += -0.5 * (self.beta_0 * nuk * np.dot(tmp_v, np.dot(self.W[k],
                tmp_v)) + nuk * np.trace(np.dot(linalg.inv(self.W_0), 
                    self.W[k])))
        return s



# Number of samples per component
n_samples = 100

# Generate random sample, two components
np.random.seed(0)

# 2, 2-dimensional Gaussians
C = np.array([[0., -0.1], [1.7, .4]])
X = np.r_[np.dot(np.random.randn(n_samples, 2), C),
          .7 * np.random.randn(n_samples, 2) + np.array([-6, 3])]

from sklearn import mixture
# Fit a mixture of gaussians with EM using five components
gmm = mixture.GMM(n_components=5, covariance_type='full')
gmm.fit(X)

vbgmm = mixture.VBGMM(n_components=5, covariance_type='full')
vbgmm.fit(X)

vb_gmm = VB_GMM(n_components=5)
vb_gmm.fit_variational(X)

color_iter = cycle(['r', 'g', 'b', 'c', 'm'])

X_repr = X
if X.shape[1] > 2:
    from sklearn import manifold
    X_repr = manifold.Isomap(n_samples/10, n_components=2).fit_transform(X)

for i, (clf, title) in enumerate([(gmm, 'GMM'),
                                  (vbgmm, 'VB DPGMM (sklearn))'),
                                  (vb_gmm, 'VB GMM (ours)')]):
    splot = pl.subplot(3, 1, 1 + i)
    Y_ = clf.predict(X)
    print Y_
    for j, (mean, covar, color) in enumerate(zip(
            clf.means_, clf._get_covars(), color_iter)):
        # We shouldn't plot the redundant components.
        if not np.any(Y_ == j):
            continue

        pl.scatter(X_repr[Y_ == j, 0], X_repr[Y_ == j, 1], .8, color=color)

        if clf.means_.shape[len(clf.means_.shape) - 1] == 2: # hack TODO remove
            # Plot an ellipse to show the Gaussian component
            v, w = linalg.eigh(covar)
            u = w[0] / linalg.norm(w[0])
            angle = np.arctan(u[1] / u[0])
            angle = 180 * angle / np.pi  # convert to degrees
            #print mean
            #print v[0]
            #print v[1]
            #print angle
            ell = mpl.patches.Ellipse(mean, v[0], v[1], 180 + angle, color='k')
            ell.set_clip_box(splot.bbox)
            ell.set_alpha(0.5)
            splot.add_artist(ell)

    pl.xlim(-10, 10)
    pl.ylim(-3, 6)
    pl.xticks(())
    pl.yticks(())
    pl.title(title)

pl.savefig('vbgmm.png')
