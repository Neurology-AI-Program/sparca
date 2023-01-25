import numpy as np

def horn_selection(X: np.ndarray, n_samples: int = 1) -> int:

    """
    Function for performing Horn's parallel analysis [1] method for estimating a good choice for the number of
    components to use for PCA decomposition of X.

    [1] J. L. Horn, "A rationale and test for the number of factors in factor analysis", 
    Psychometrika 1965 Vol. 30 Issue 2 Pages 179-185

    Parameters
    ----------
    X: numpy.ndarray
        2d array representing the design matrix
    n_samples: int
        integer number of resamples used to estimate threshold (default 1)
    
    Returns
    -------
    h: int
        Estimated number of components
    """
    
    n_s, n_f = X.shape

    if n_s > n_f:
        E = X.T
    else:
        E = X

    cov_E = np.cov(E)
    _eig_E, _ = np.linalg.eig(cov_E)
    eig_E = -np.sort(-np.real(_eig_E))

    eig_S = np.zeros((n_samples, eig_E.size))

    for i in range(n_samples):

        S = np.random.randn(*E.shape)
        cov_S = np.cov(S)
        _eig_S, _ = np.linalg.eig(cov_S)
        eig_S[i] = -np.sort(-np.real(_eig_S))

    e_gt_s = (eig_E > eig_S.mean(axis = 0)).astype('int')

    return np.max([np.argmax(np.roll(e_gt_s, 1) - e_gt_s), 1]).astype('int')