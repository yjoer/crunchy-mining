from abc import ABC
from abc import abstractmethod


class BasePreprocessor(ABC):
    train_val_sets = {}
    encoders = {}

    def __init__(self, variables: dict):
        self.variables = variables

    @abstractmethod
    def fit(self):
        pass

    def get_train_val_sets(self):
        return self.train_val_sets

    def get_encoders(self):
        return self.encoders
