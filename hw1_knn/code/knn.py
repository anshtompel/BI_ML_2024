import numpy as np


class KNNClassifier:
    """
    K-neariest-neighbor classifier using L1 loss
    """
    
    def __init__(self, k=1):
        self.k = k
    

    def fit(self, X, y):
        self.train_X = X
        self.train_y = y


    def predict(self, X, n_loops=0):
        """
        Uses the KNN model to predict clases for the data samples provided
        
        Arguments:
        X, np array (num_samples, num_features) - samples to run
           through the model
        num_loops, int - which implementation to use

        Returns:
        predictions, np array of ints (num_samples) - predicted class
           for each sample
        """
        
        if n_loops == 0:
            distances = self.compute_distances_no_loops(X)
        elif n_loops == 1:
            distances = self.compute_distances_one_loops(X)
        else:
            distances = self.compute_distances_two_loops(X)
        
        if len(np.unique(self.train_y)) == 2:
            return self.predict_labels_binary(distances)
        else:
            return self.predict_labels_multiclass(distances)


    def compute_distances_two_loops(self, X):
        """
        Computes L1 distance from every sample of X to every training sample
        Uses simplest implementation with 2 Python loops

        Arguments:
        X, np array (num_test_samples, num_features) - samples to run
        
        Returns:
        distances, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        """
        
        """
        YOUR CODE IS HERE
        """
        num_train = self.train_X.shape[0]
        num_test = X.shape[0]
        distances = np.zeros((num_test, num_train))
        for i in range(num_test):
            for j in range(num_train):
                dist = np.sum(np.abs(X[i] - self.train_X[j]))
                distances[i,j] = dist
        return distances


    def compute_distances_one_loop(self, X):
        """
        Computes L1 distance from every sample of X to every training sample
        Vectorizes some of the calculations, so only 1 loop is used

        Arguments:
        X, np array (num_test_samples, num_features) - samples to run
        
        Returns:
        distances, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        """

        """
        YOUR CODE IS HERE
        """
        num_train = self.train_X.shape[0]
        num_test = X.shape[0]
        distances = np.zeros((num_test, num_train))
        for i in range(num_test):
            distances[i, :] = np.sum(np.abs(X[i,:] - self.train_X), axis = 1)
        return distances

    def compute_distances_no_loops(self, X):
        """
        Computes L1 distance from every sample of X to every training sample
        Fully vectorizes the calculations using numpy

        Arguments:
        X, np array (num_test_samples, num_features) - samples to run
        
        Returns:
        distances, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        """

        """
        YOUR CODE IS HERE
        """
        pass


    def predict_labels_binary(self, distances):
        """
        Returns model predictions for binary classification case
        
        Arguments:
        distances, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        Returns:
        pred, np array of bool (num_test_samples) - binary predictions 
           for every test sample
        """
        #n_train = distances.shape[1]
        n_test = distances.shape[0]
        prediction = np.zeros(n_test)
        self.train_y = self.train_y.astype(int)
        for i in range(n_test):
            y_indicies = np.argsort(distances[i, :])
            predict_y = self.train_y[y_indicies[ : self.k]]
            prediction[i] = np.argmax(np.bincount(predict_y))
        return prediction


    def predict_labels_multiclass(self, distances):
        """
        Returns model predictions for multi-class classification case
        
        Arguments:
        distances, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        Returns:
        pred, np array of int (num_test_samples) - predicted class index 
           for every test sample
        """

        #n_train = distances.shape[0]
        n_test = distances.shape[0]
        prediction = np.zeros(n_test)
        self.train_y = self.train_y.astype(int)
        for i in range(n_test):
            y_indicies = np.argsort(distances[i, :])
            predict_y = self.train_y[y_indicies[ : self.k]]
            prediction[i] = np.argmax(np.bincount(predict_y))
        return prediction
