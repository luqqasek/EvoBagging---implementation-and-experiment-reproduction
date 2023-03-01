import numpy as np
from implementation_comparison.data_processing import load_data
from numpy import genfromtxt
from sklearn.model_selection import train_test_split


class DataProcessing:
    def __init__(self):
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def load_from_csv_file(self, path, test_size=0,delimiter=","):
        """Loading file from csv with train test split if test_size parameter > 0"""
        raw_data = genfromtxt(path, delimiter=delimiter)
        X = raw_data[: , :1]
        y = raw_data[: , -1]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=test_size)

    @staticmethod
    def check_quality(matrix):
        """
        Checks whether all elements of matrix are integers or floats. Returns True if all are
        """
        is_numeric = np.vectorize(lambda x: isinstance(x, (int, float)))
        return np.sum(~is_numeric(matrix)) == 0

    def from_original_paper(self,
                            dataset_name,
                            test_size,
                            random_state=1):
        """
        Load data from original paper with test split if test_size parameter is given. T
        est_size is float number between 0 and 1
        """

        self.reset()
        X_train, X_test, y_train, y_test = load_data(dataset_name, test_size=test_size, random_state=random_state)

        self.X_train = np.asarray(X_train)
        if not(X_test is None):
            self.X_test = np.asarray(X_test)
        self.y_train = np.asarray(y_train).flatten()
        if not(y_test is None):
            self.y_test = np.asarray(y_test).flatten()

    def reset(self):
        """
        Set class attributes to None
        """
        self.__init__()
