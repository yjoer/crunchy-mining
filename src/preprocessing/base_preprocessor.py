from abc import ABC
from abc import abstractmethod


class BasePreprocessor(ABC):
    def __init__(self, variables: dict):
        self.train_val_sets = {}
        self.encoders = {}
        self.variables = variables

    @abstractmethod
    def fit(self):
        pass

    def get_train_val_sets(self):
        return self.train_val_sets

    def get_encoders(self):
        return self.encoders
