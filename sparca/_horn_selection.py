import numpy as np

def horn_selection(X, n_samples = 1):
    
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

    return np.max([np.argmax(np.roll(e_gt_s, 1) - e_gt_s), 1])