"""Mixture model using EM"""
from typing import Tuple
import numpy as np
from common import GaussianMixture



def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data
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
        p = np.array([])
        for i in range(len(mixture.mu)):
            p = np.append(p, np.array([[multivariate_normal.pdf(x, mean=mixture.mu[i], cov=mixture.var[i])]]))
        if len(ps) == 0: ps = np.array([p])
        else: ps = np.append(ps, [p], axis=0)
        P = np.array([])
        for j in range(len(p)):
            P = np.append(P, mixture.p[j]*p[j]/mixture.p.dot(p).sum())
        if len(prob) == 0: prob = np.array([P])
        else: prob = np.append(prob, [P], axis=0)
    
    likelihood = 0
    for i in range(len(ps)):
        for j in range(len(ps[0])):
            likelihood += prob[i,j]*np.log(ps[i,j]*mixture.p[j]/prob[i,j])
    
    return prob, likelihood


def mstep(X: np.ndarray, post: np.ndarray) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    updated_mu = np.array([])
    updated_var = np.array([])
    updated_p = np.array([])
    for i in range(post.shape[1]):
        updated_p = np.append(updated_p, sum(post[:, i])/len(X))
        if len(updated_mu) == 0: updated_mu = (np.matrix([post[:, i]])*X)/post[:, i].sum()
        else: updated_mu = np.append(updated_mu, (np.matrix([post[:, i]])*X)/post[:, i].sum(), axis=0)
        updated_var = np.append(updated_var, (np.matrix([post[:, i]])*np.power(X - updated_mu[i], 2)).sum()/((np.matrix([post[:, i]])*np.power(X - updated_mu[i], 2)).shape[1]*post[:, i].sum()))

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
        mixture = GaussianMixture(np.array(mixture.mu), mixture.var, mixture.p)
        diff = new_likelihood - old_likelihood

    return mixture, prob, new_likelihood