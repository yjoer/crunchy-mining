import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler

from src.preprocessing.base_preprocessor import BasePreprocessor


# Enable the loading of the preprocessed datasets from any preprocessors.
class GenericPreprocessor(BasePreprocessor):
    def fit():
        pass


# Ordinal Encoding + No Scaling
class PreprocessorV1(BasePreprocessor):
    def fit(self, df_train: pd.DataFrame, df_test: pd.DataFrame, name: str):
        oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        X_train_cat = oe.fit_transform(df_train[self.variables["categorical"]])
        X_test_cat = oe.transform(df_test[self.variables["categorical"]])

        X_train_num = df_train[self.variables["numerical"]]
        X_test_num = df_test[self.variables["numerical"]]

        X_train = np.hstack((X_train_cat, X_train_num))
        X_test = np.hstack((X_test_cat, X_test_num))

        le = LabelEncoder()
        y_train = le.fit_transform(df_train[self.variables["target"]])
        y_test = le.transform(df_test[self.variables["target"]])

        self.train_val_sets[name] = (X_train, y_train, X_test, y_test)
        self.encoders["ordinal"] = oe
        self.encoders["label"] = le


# One-Hot Encoding + No Scaling
class PreprocessorV2(BasePreprocessor):
    def fit(self, df_train: pd.DataFrame, df_test: pd.DataFrame, name: str):
        hot = OneHotEncoder(handle_unknown="infrequent_if_exist")
        X_train_cat = hot.fit_transform(df_train[self.variables["categorical"]])
        X_test_cat = hot.transform(df_test[self.variables["categorical"]])

        X_train_num = df_train[self.variables["numerical"]]
        X_test_num = df_test[self.variables["numerical"]]

        X_train = sp.hstack((X_train_cat, X_train_num), format="csr")
        X_test = sp.hstack((X_test_cat, X_test_num), format="csr")

        le = LabelEncoder()
        y_train = le.fit_transform(df_train[self.variables["target"]])
        y_test = le.transform(df_test[self.variables["target"]])

        self.train_val_sets[name] = (X_train, y_train, X_test, y_test)
        self.encoders["one_hot"] = hot
        self.encoders["label"] = le


# Ordinal Encoding + Standard Scaling
class PreprocessorV3(BasePreprocessor):
    def fit(self, df_train: pd.DataFrame, df_test: pd.DataFrame, name: str):
        oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        X_train_cat = oe.fit_transform(df_train[self.variables["categorical"]])
        X_test_cat = oe.transform(df_test[self.variables["categorical"]])

        ss = StandardScaler()
        X_train_num = ss.fit_transform(df_train[self.variables["numerical"]])
        X_test_num = ss.transform(df_test[self.variables["numerical"]])

        X_train = np.hstack((X_train_cat, X_train_num))
        X_test = np.hstack((X_test_cat, X_test_num))

        le = LabelEncoder()
        y_train = le.fit_transform(df_train[self.variables["target"]])
        y_test = le.transform(df_test[self.variables["target"]])

        self.train_val_sets[name] = (X_train, y_train, X_test, y_test)
        self.encoders["ordinal"] = oe
        self.encoders["standard"] = ss
        self.encoders["label"] = le


# Ordinal Encoding + Standard Scaling + Min-Max Scaling
class PreprocessorV4(BasePreprocessor):
    def fit(self, df_train: pd.DataFrame, df_test: pd.DataFrame, name: str):
        oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        X_train_cat = oe.fit_transform(df_train[self.variables["categorical"]])
        X_test_cat = oe.transform(df_test[self.variables["categorical"]])

        ss = StandardScaler()
        X_train_num = ss.fit_transform(df_train[self.variables["numerical"]])
        X_test_num = ss.transform(df_test[self.variables["numerical"]])

        X_train = np.hstack((X_train_cat, X_train_num))
        X_test = np.hstack((X_test_cat, X_test_num))

        mm = MinMaxScaler()
        X_train_scaled = mm.fit_transform(X_train)
        X_test_scaled = mm.transform(X_test)

        le = LabelEncoder()
        y_train = le.fit_transform(df_train[self.variables["target"]])
        y_test = le.transform(df_test[self.variables["target"]])

        self.train_val_sets[name] = (X_train_scaled, y_train, X_test_scaled, y_test)
        self.encoders["ordinal"] = oe
        self.encoders["standard"] = ss
        self.encoders["min_max"] = mm
        self.encoders["label"] = le
