from abc import ABC, abstractmethod


class IModel(ABC):
    @abstractmethod
    def predict(self, X):
        pass

    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def score(self, X, y):
        pass
