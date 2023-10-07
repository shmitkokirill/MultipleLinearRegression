import numpy as np

class MultipleRegression:
    def __init__(self, learning_rate=0.001, step_cnt=10000):
        self.__learning_rate = learning_rate
        self.__step_cnt = step_cnt
        self.__weights = None
        self.__offsets = None
        self.__mae_vals = []
        self.__mape_vals = []
        self.__step_arr = []

    def getMaeErrs(self):
      return self.__mae_vals, self.__step_arr

    def getMapeErrs(self):
      return self.__mape_vals, self.__step_arr

    def fit(self, x : np.ndarray, y : np.ndarray):
        num_samples, num_features = x.shape

        self.__weights = np.zeros(num_features)
        self.__offsets = 0

        for i in range(self.__step_cnt):
            y_pred = self.predict(x)

            # cals grads
            dw = (1 / num_samples) * np.dot(x.T, (y_pred - y))
            db = (1 / num_samples) * np.sum(y_pred - y)

            # append errors
            self.__mae_vals.append(self.__mae(y, y_pred))
            self.__mape_vals.append(self.__mape(y, y_pred))
            self.__step_arr.append(i)

            # update weights and offsets
            self.__weights = self.__weights - self.__learning_rate * dw
            self.__offsets = self.__offsets - self.__learning_rate * db

    def predict(self, x):
        return np.dot(x, self.__weights) + self.__offsets

    def __mae(self, y_train, y_pred):
        return np.mean(np.abs(y_train - y_pred))

    def __mape(self, y_train, y_pred):
        return np.mean(np.abs((y_train - y_pred) / y_train)) * 100

