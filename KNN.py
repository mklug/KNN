import heapq
import numpy as np
import pandas as pd


class KNN:
    '''
    K-nearest neighbors classifier that allows the nearest 
    neighbors to be either uniformly weighted or weighted
    by the inverse of the distance to the test points. 
    Largely follows the sklearn API.
    '''

    def __init__(self, n_neighbors=5, weights='uniform'):
        '''
        Initialize with the number of neighbors as ``n_neighbors``
        and with the ``weights`` parameter in {'uniform', 'distance'}.
        With 'uniform', each of the nearest neighbors votes with even
        weight.  With 'distance', each nearest neighbor votes with 
        weight being the inverse of the distance to the test point.  
        '''
        if weights not in {'uniform', 'distance'}:
            raise Exception("Invalid ``weights`` parameter.")
        self.weights = weights
        self.n_neighbors = n_neighbors

    def fit(self, X_train, Y_train):
        '''
        Takes ``X_train`` in a form convertible to a DataFrame and 
        ``Y_train`` in a form convertible to a Series, both with 
        the same index. Stores the classes in the ``Y_train`` data.  
        '''
        X_train = pd.DataFrame(X_train)
        Y_train = pd.Series(Y_train)

        if len(X_train) != len(Y_train):
            raise Exception("Lengths of X and Y must be the same.")

        if not all(X_train.index == Y_train.index):
            raise Exception("Indices of X and Y must be the same.")

        self.X_train = X_train  # DataFrame
        self.Y_train = Y_train  # Series
        self.classes_ = Y_train.unique()  # np.array

    def _predict_proba_row(self, x):
        '''
        Takes a Series ``x`` with the same columns as ``X_train`` and 
        computes the distance to each point in ``X_train``.  Stores
        the ``n_neighbors`` closest points and uses those to return
        probability vector with index given by the classes in 
        ``Y_train``, as determined by ``weights``.  If ``weights``
        is 'uniform', then each of the nearest neighbors votes evenly
        with their class and if ``weights`` is 'distance', then each 
        nearest neighbor votes with weighting the inverse of the 
        distance to ``x``.  
        '''
        if not all(x.index == self.X_train.columns):
            raise Exception("Input must have index the same \
                            as the columns of the training data.")

        h = []
        distances = self.X_train.apply(lambda x0: np.linalg.norm(x0-x),
                                       axis=1)

        for d, y in zip(distances, self.Y_train):
            heapq.heappush(h, (-d, y))
            if len(h) > self.n_neighbors:
                heapq.heappop(h)

        res = pd.Series(0.0, index=self.classes_)

        if self.weights == 'uniform':
            while len(h) > 0:
                _, current_label = heapq.heappop(h)
                res[current_label] += 1

        elif self.weights == 'distance':
            while len(h) > 0:
                dist, current_label = heapq.heappop(h)
                if np.isclose(dist, 0):
                    res = pd.Series(0, index=self.classes_)
                    res[current_label] = 1
                    return res
                res[current_label] += 1/dist

        res /= sum(res)
        return res

    def predict_proba(self, X_test):
        '''
        Takes a DataFrame with columns the same as the columns 
        of ``X_train``.  Returns a DataFrame with same index 
        as the input and with columns given by the classes
        in ``Y_train`` where each row represents the probability 
        vector for the input being in each respective class.  
        '''

        X_test = pd.DataFrame(X_test)

        if not all(X_test.columns == self.X_train.columns):
            raise Exception("Input must have index the same \
                            as the columns of the training data.")

        return X_test.apply(lambda x: self._predict_proba_row(x),
                            axis=1)

    def predict(self, X_test):
        '''
        Takes a DataFrame with columns the same as the columns 
        of ``X_train``.  Returns a Series with the same index
        as the index of ``X_train`` and with the entries given 
        by the predicted classes.   
        '''

        X_test = pd.DataFrame(X_test)

        if not all(X_test.columns == self.X_train.columns):
            raise Exception("Input must have index the same \
                            as the columns of the training data.")

        probs = self.predict_proba(X_test)
        return probs.apply(lambda x: x.idxmax(),
                           axis=1)
