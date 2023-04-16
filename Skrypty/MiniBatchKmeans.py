from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.cluster import MiniBatchKMeans

class MiniBatchKMeansx(BaseEstimator, ClassifierMixin):
    def __init__(self, n_clusters, random_state=None):
        self.cluster_centers_ = None
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=random_state)


    def partial_fit(self, X, y, classes=None):
        self.kmeans.partial_fit(X)
        self.cluster_centers_ = self.kmeans.cluster_centers_

    def predict(self, X):
        return self.kmeans.predict(X)

