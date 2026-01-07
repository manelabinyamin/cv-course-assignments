from builtins import object, range

import numpy as np


class KNearestNeighbor(object):
    """a kNN classifier with L2 distance"""

    def __init__(self):
        pass

    def train(self, X, y):
        """
        Train the classifier. For k-nearest neighbors this is just
        memorizing the training data.

        Inputs:
        - X: A numpy array of shape (num_train, D) containing the training data
          consisting of num_train samples each of dimension D.
        - y: A numpy array of shape (N,) containing the training labels, where
             y[i] is the label for X[i].
        """
        self.X_train = X
        self.y_train = y

    def predict(self, X, k=1, num_loops=0):
        """
        Predict labels for test data using this classifier.

        Inputs:
        - X: A numpy array of shape (num_test, D) containing test data consisting
             of num_test samples each of dimension D.
        - k: The number of nearest neighbors that vote for the predicted labels.
        - num_loops: Determines which implementation to use to compute distances
          between training points and testing points.

        Returns:
        - y: A numpy array of shape (num_test,) containing predicted labels for the
          test data, where y[i] is the predicted label for the test point X[i].
        """
        if num_loops == 0:
            dists = self.compute_distances_no_loops(X)
        elif num_loops == 1:
            dists = self.compute_distances_one_loop(X)
        elif num_loops == 2:
            dists = self.compute_distances_two_loops(X)
        else:
            raise ValueError("Invalid value %d for num_loops" % num_loops)

        return self.predict_labels(dists, k=k)

    def compute_distances_two_loops(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using a nested loop over both the training data and the
        test data.

        Inputs:
        - X: A numpy array of shape (num_test, D) containing test data.

        Returns:
        - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
          is the Euclidean distance between the ith test point and the jth training
          point.
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            for j in range(num_train):
                dist = np.sqrt(np.sum((X[i] - self.X_train[j]) ** 2))
                dists[i, j] = dist
        return dists

    def compute_distances_one_loop(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using a single loop over the test data.

        Input / Output: Same as compute_distances_two_loops
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            dist = np.sqrt(np.sum((X[i] - self.X_train) ** 2, axis=1))
            dists[i, :] = dist
        return dists

    def compute_distances_no_loops(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using no explicit loops.

        Input / Output: Same as compute_distances_two_loops
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        """
        # first attempt - simple broadcasting. crushes the run due to oom
        
        dists = X[None, ...] - self.X_train[:, None, ...]
        dists = np.sqrt(np.sum(dists**2, axis=2))

        # second attempt - Use the fact that (a - b)**2 == a*a.T - 2*a*b + b*b.T
        D[i,j]^2 = ||A[i] - B[j]||^2
        = (A[i] - B[j]).T * (A[i] - B[j])
        = A[i].T * A[i] - 2 * A[i] * B[j].T + B[j].T * B[j]
        
        D^2 = sum(A^2, axis=1, keepdims=True) - 2 * A * B.T + sum(B^2, axis=1, keepdims=True).T

        """

        dists = (
            np.sum(X**2, axis=1, keepdims=True)
            - 2 * X @ self.X_train.T
            + np.sum(self.X_train**2, axis=1).T
        )
        dists = np.sqrt(dists)
        return dists

    def predict_labels(self, dists, k=1):
        """
        Given a matrix of distances between test points and training points,
        predict a label for each test point.

        Inputs:
        - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
          gives the distance betwen the ith test point and the jth training point.

        Returns:
        - y: A numpy array of shape (num_test,) containing predicted labels for the
          test data, where y[i] is the predicted label for the test point X[i].
        """
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)
        for i in range(num_test):
            # A list of length k storing the labels of the k nearest neighbors to
            # the ith test point.

            # get k min values
            closest_y = self.y_train[np.argsort(dists[i])[:k]]

            values, counts = np.unique(np.sort(closest_y), return_counts=True)

            y_pred[i] = values[np.argmax(counts)]

        return y_pred
