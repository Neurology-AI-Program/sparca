import numpy as np
import pandas as pd
from collections import defaultdict
from joblib import Parallel, delayed

from sklearn.linear_model import orthogonal_mp
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import FeatureAgglomeration

from ._horn_selection import horn_selection

def _fit_sparse_coefs(X, y, r2_threshold = 0.95):
    
    n_targets = y.shape[1]
    n_features = X.shape[1]
    sparse_coefs = np.zeros((n_features, n_targets))
    remaining_inds = np.arange(n_targets)
    n_nonzero_coefs = 1

    while remaining_inds.size > 0 and n_nonzero_coefs <= n_features:
        _y = y[:, remaining_inds]
        coefs = orthogonal_mp(X, _y, n_nonzero_coefs = n_nonzero_coefs).reshape(X.shape[1], -1)
        y_fit = X@coefs
        r2 = r2_score(_y, y_fit, multioutput = 'raw_values')
        stop_inds = np.argwhere(r2 >= r2_threshold).reshape(-1)
        cont_inds = np.argwhere(r2 < r2_threshold).reshape(-1)
        sparse_coefs[:, remaining_inds[stop_inds]] = coefs[:, stop_inds]
        remaining_inds = remaining_inds[cont_inds]
        n_nonzero_coefs += 1

    return sparse_coefs


def _fit_cluster(c, clusters, X, r2_threshold):

    header = clusters[c]['header']
    n_features = len(header)
    n_compressed_features = 0
    X_cluster = X[header]

    if n_features > 1:
        h = horn_selection(X_cluster)
        n_compressed_features += h
        pca_cluster = PCA(n_components = h)
        X_pca = pca_cluster.fit_transform(X_cluster)
    else:
        n_compressed_features += 1
        X_pca = X_cluster.values

    sparse_coefs = _fit_sparse_coefs(X_cluster, X_pca, r2_threshold)
    components = {col : sorted([(l, f) for l, f in zip(header, sparse_coefs[:, col]) if f != 0], key = lambda t: -t[1]) for col in range(sparse_coefs.shape[1])}

    return n_compressed_features, sparse_coefs, components


class SparCA(BaseEstimator, TransformerMixin):

    def __init__(self, n_clusters, cluster_model = None, n_jobs = None, r2_threshold = 0.95):

        self.pre_scale = StandardScaler()

        if cluster_model is None:
            self.agg = FeatureAgglomeration(n_clusters = n_clusters)
            self.n_clusters = n_clusters
            self.cluster_model_is_fitted = False
        else:
            self.agg = cluster_model
            self.n_clusters = cluster_model.n_clusters
            self.cluster_model_is_fitted = True
        
        self.clusters_ = defaultdict(dict)
        self.n_jobs = n_jobs
        self.r2_threshold = r2_threshold

    
    def _scale_dataframe(self, df):
        
        X_scale_data = self.pre_scale.transform(df)
        
        return pd.DataFrame(X_scale_data, columns = df.columns, index = df.index)


    @staticmethod
    def _cluster_pretty_string(cluster):

        pretty_string = ''
        pretty_string += 'Cluster features:\n'

        for h in cluster['header']:
            pretty_string += f'\t* {h}\n'

        max_abs_coef = np.abs(cluster['sparse_coefs']).max()
        symbol = lambda f: '+' if f > 0 else '-'

        pretty_string += 'Sparse components:\n'

        for c in cluster['components']:
            pretty_string += f'\t* component {c}\n'
            int_rep = [(l, np.ceil(3*np.abs(f)/max_abs_coef)*np.sign(f)) for l, f in cluster['components'][c]]
            for l, f in int_rep:
                pretty_string += f'\t\t* {l} {symbol(f)*np.abs(int(f))}\n'

        return pretty_string

    
    def fit(self, X):
        
        self.n_raw_features_ = X.shape[1]
        self.n_compressed_features_ = 0
        self.pre_scale.fit(X)
        X_scale = self._scale_dataframe(X.copy())

        if not self.cluster_model_is_fitted:
            self.agg.fit(X_scale)
        
        for i, l in enumerate(self.agg.labels_):
            if l in self.clusters_:
                self.clusters_[l]['header'].append(X_scale.columns[i])
            else:
                self.clusters_[l]['header'] = [X_scale.columns[i]]
        
        cluster_fit_fn = lambda c: (c, _fit_cluster(c, self.clusters_, X_scale, self.r2_threshold))
        
        if self.n_jobs is not None and self.n_jobs > 1:
            cluster_fit_results = Parallel(n_jobs = self.n_jobs)(delayed(cluster_fit_fn)(c) for c in self.clusters_)
        else:
            cluster_fit_results = [cluster_fit_fn(c) for c in self.clusters_]
            
        self.n_compressed_features_ = 0
        for c, (n_compressed_features, sparse_coefs, components) in cluster_fit_results:
            
            self.clusters_[c]['sparse_coefs'] = sparse_coefs
            self.clusters_[c]['components'] = components
            self.n_compressed_features_ += n_compressed_features


    def transform(self, X, cluster_ids = None):
        
        X_scale = self._scale_dataframe(X)
        
        col_stack = []
        columns = []
        
        if cluster_ids is None:
            cluster_ids = range(len(self.clusters_))
        
        for c in cluster_ids:
            
            X_cluster = X_scale[self.clusters_[c]['header']]
            X_compressed = X_cluster@self.clusters_[c]['sparse_coefs']
            col_stack.append(X_compressed)
            columns.extend([f'cluster_{c}_component_{i}' for i in range(X_compressed.shape[1])])
            
        return pd.DataFrame(np.column_stack(col_stack), columns = columns, index = X.index)


    def cluster_reports(self, cluster_ids = None):
        
        if cluster_ids is None:
            cluster_ids = range(len(self.clusters_))

        cluster_reports = []
            
        for c in cluster_ids:
            
            pretty_string = self._cluster_pretty_string(self.clusters_[c])
            cluster_text = f'Cluster {c} report:\n' + pretty_string
            cluster_reports.append(cluster_text)

        return cluster_reports