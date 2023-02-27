# sparca
sparse compressed agglomeration for dimensionality reduction and feature selection

# Requirements
* scikit-learn
* pandas
* numpy

# Installation
```
pip install git+https://github.com/Neurology-AI-Program/sparca.git
```

# Usage
For testing purposes,
a small dataset sourced from the UCI repository is provided with this library.  It can be loaded with<br>
 ```
 from sparca import load_test_data, SparCA
 df, info = load_test_data()
 ```
 When initializing a SparCA model, the only required parameter is the number of clusters used for grouping features<br>
 ```
 model = SparCA(n_clusters = 9)
 ```
SparCA provides a scikit-learn style estimator API with familiar fit and transform methods, but these methods must be applied to a pandas DataFrame.  Application to numpy ndarray representations of the design matrix are not currently supported.<br>
```
X = df.drop(columns = ['target'])
model.fit(X)
X_reduced = model.transform(X)
```
Information useful for reasoning about the model can be gleaned from the summaries provided by `model.cluster_reports()`, which produces a list of the features grouped into each cluster and the subset of features used in the derived components.

