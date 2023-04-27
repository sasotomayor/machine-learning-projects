"""Mixture model for matrix completion"""
from typing import Tuple
import numpy as np
from scipy.special import logsumexp
from common import GaussianMixture


def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment

    """
    from scipy.stats import multivariate_normal

    prob = np.array([])
    ps = np.array([])
    for x in X:
        index_0s = np.where(x != 0)[0]
        p = np.array([])
        for i in range(len(mixture.mu)):
            if len(x[index_0s]) > 0:
                p = np.append(p, np.array([[multivariate_normal.pdf(x[index_0s], mean=mixture.mu[i][index_0s], cov=mixture.var[i])]]))
            else:
                p = np.append(p, np.array([[sum(mixture.p)]]))
        if len(ps) == 0: ps = np.array([p])
        else: ps = np.append(ps, [p], axis=0)
        P = np.array([])
        for j in range(len(p)):
            if mixture.p.dot(p).sum() != 0: P = np.append(P, mixture.p[j]*p[j]/(mixture.p.dot(p).sum()))
            else: P = np.append(P, 0)
        if len(prob) == 0: prob = np.array([P])
        else: prob = np.append(prob, [P], axis=0)
    likelihood = 0
    for i in range(len(ps)):
        for j in range(len(ps[0])):
            likelihood += prob[i,j]*np.log(np.exp(-16) + ps[i,j]*mixture.p[j]/(prob[i,j]))
    
    return prob, likelihood



def mstep(X: np.ndarray, post: np.ndarray, mixture: GaussianMixture,
          min_variance: float = .25) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        post: (n, K) array holding the soft counts
            for all components for all examples
        mixture: the current gaussian mixture
        min_variance: the minimum variance for each gaussian

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    updated_mu = np.array([])
    updated_var = np.array([])
    updated_p = np.array([])
    for i in range(post.shape[1]):
        updated_p = np.append(updated_p, sum(post[:, i])/len(X))
        mu_i = np.array([])
        for l in range(len(X[0])):
            mu_ik_num = 0
            mu_ik_den = 0
            for j in range(len(X)):
                gamma_jl = X[j, l] > 0
                mu_ik_num += post[j, i]*gamma_jl*X[j,l]
                mu_ik_den += post[j, i]*gamma_jl
            if mu_ik_den > 1: mu_i = np.append(mu_i, mu_ik_num/mu_ik_den)
            else:
                mu_i = np.append(mu_i, mixture.mu[i, l])
        if len(updated_mu) == 0: updated_mu = np.matrix([mu_i])
        else: updated_mu = np.append(updated_mu, np.matrix([mu_i]), axis=0)

        var_i_num = 0
        var_i_den = 0
        for j in range(len(X)):
            c_j = np.where(X[j] > 0)[0]
            var_i_num += post[j, i]*np.power(np.linalg.norm(X[j, c_j] - updated_mu[i, c_j]), 2)
            var_i_den += post[j, i]*len(c_j)
        updated_var = np.append(updated_var, max(var_i_num/var_i_den, min_variance))

    return GaussianMixture(updated_mu, updated_var, updated_p)


def run(X: np.ndarray, mixture: GaussianMixture,
        post: np.ndarray) -> Tuple[GaussianMixture, np.ndarray, float]:
    """Runs the mixture model

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the current assignment
    """
    old_likelihood = -np.inf
    new_likelihood = -999999999999999
    diff = np.inf

    while diff > 0.000001*np.abs(new_likelihood):
        old_likelihood = new_likelihood
        prob, new_likelihood = estep(X, mixture)
        mixture = mstep(X, prob)
        #mixture = GaussianMixture(np.array(mixture.mu), mixture.var, mixture.p)
        diff = new_likelihood - old_likelihood

    return mixture, prob, new_likelihood


def fill_matrix(X: np.ndarray, mixture: GaussianMixture) -> np.ndarray:
    """Fills an incomplete matrix according to a mixture model

    Args:
        X: (n, d) array of incomplete data (incomplete entries =0)
        mixture: a mixture of gaussians

    Returns
        np.ndarray: a (n, d) array with completed data
    """
    
    from scipy.stats import multivariate_normal

    prob = np.array([])
    ps = np.array([])
    for x in X:
        index_0s = np.where(x != 0)[0]
        p = np.array([])
        for i in range(len(mixture.mu)):
            if len(x[index_0s]) > 0:
                p = np.append(p, np.array([[multivariate_normal.pdf(x[index_0s], mean=mixture.mu[i][index_0s], cov=mixture.var[i])]]))
            else:
                p = np.append(p, np.array([[sum(mixture.p)]]))
        if len(ps) == 0: ps = np.array([p])
        else: ps = np.append(ps, [p], axis=0)
        P = np.array([])
        for j in range(len(p)):
            P = np.append(P, mixture.p[j]*p[j]/(mixture.p.dot(p).sum()))
        if len(prob) == 0: prob = np.array([P])
        else: prob = np.append(prob, [P], axis=0)

    pred = np.array([])
    for i in range(len(X)):
        row_i = np.array([])
        for j in range(len(X[0])):
            if X[i, j] == 0:
                row_i = np.append(row_i, np.matmul(prob[i], mixture.mu[:, j]))
            else:
                row_i = np.append(row_i, X[i, j])
        if len(pred) == 0: pred = np.array([row_i])
        else: pred = np.append(pred, [row_i], axis=0)
    
    return pred