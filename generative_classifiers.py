import numpy as np
from scipy.stats import multivariate_normal

def run_naive_bayes(train_x, train_y, test_x, test_y):
    mu1 = np.mean(train_x[train_y == 1], axis=0)
    sig1 = np.diag(np.std(train_x[train_y == 1], axis=0))
    mvn_rr = multivariate_normal(mean=mu1, cov=sig1)

    mu2 = np.mean(train_x[train_y == 0], axis=0)
    sig2 = np.diag(np.std(train_x[train_y == 0], axis=0))
    mvn_star = multivariate_normal(mean=mu2, cov=sig2)

    rr_probs = mvn_rr.pdf(test_x)
    star_probs = mvn_star.pdf(test_x)

    re = np.exp(rr_probs)
    se = np.exp(star_probs)
    np.sum([re, se], axis=1)
    ss = np.sum(np.vstack([re, se]).T, axis=1)

    return np.vstack([(re/ss), (se/ss)]).T[:, 0]


def run_GMM(train_x, train_y, test_x, test_y):
    mu1 = np.mean(train_x[train_y == 1], axis=0)
    cov1 = np.cov(train_x[train_y == 1].T)
    mvn_rr = multivariate_normal(mean=mu1, cov=cov1)

    mu2 = np.mean(train_x[train_y == 0], axis=0)
    cov2 = np.cov(train_x[train_y == 0].T)
    mvn_star = multivariate_normal(mean=mu2, cov=cov2)

    rr_probs = mvn_rr.pdf(test_x)
    star_probs = mvn_star.pdf(test_x)

    re = np.exp(rr_probs)
    se = np.exp(star_probs)
    np.sum([re, se], axis=1)
    ss = np.sum(np.vstack([re, se]).T, axis=1)

    return np.vstack([(re/ss), (se/ss)]).T[:, 0]
