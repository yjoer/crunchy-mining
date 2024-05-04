import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import TargetEncoder

from .base_preprocessor import BasePreprocessor


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


# Target Encoder + No Scaling
class PreprocessorV3(BasePreprocessor):
    def fit(self, df_train: pd.DataFrame, df_test: pd.DataFrame, name: str):
        # Encode the target first because the target encoder needs them.
        le = LabelEncoder()
        y_train = le.fit_transform(df_train[self.variables["target"]])
        y_test = le.transform(df_test[self.variables["target"]])

        te = TargetEncoder(target_type="binary", random_state=12345)
        X_train_cat = te.fit_transform(df_train[self.variables["categorical"]], y_train)
        X_test_cat = te.transform(df_test[self.variables["categorical"]])

        X_train_num = df_train[self.variables["numerical"]]
        X_test_num = df_test[self.variables["numerical"]]

        X_train = np.hstack((X_train_cat, X_train_num))
        X_test = np.hstack((X_test_cat, X_test_num))

        self.train_val_sets[name] = (X_train, y_train, X_test, y_test)
        self.encoders["target"] = te
        self.encoders["label"] = le


# Ordinal Encoding + Standard Scaling
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

        le = LabelEncoder()
        y_train = le.fit_transform(df_train[self.variables["target"]])
        y_test = le.transform(df_test[self.variables["target"]])

        self.train_val_sets[name] = (X_train, y_train, X_test, y_test)
        self.encoders["ordinal"] = oe
        self.encoders["standard"] = ss
        self.encoders["label"] = le


# Target Encoder + Standard Scaling
class PreprocessorV5(BasePreprocessor):
    def fit(self, df_train: pd.DataFrame, df_test: pd.DataFrame, name: str):
        le = LabelEncoder()
        y_train = le.fit_transform(df_train[self.variables["target"]])
        y_test = le.transform(df_test[self.variables["target"]])

        te = TargetEncoder(target_type="binary", random_state=12345)
        X_train_cat = te.fit_transform(df_train[self.variables["categorical"]], y_train)
        X_test_cat = te.transform(df_test[self.variables["categorical"]])

        ss = StandardScaler()
        X_train_num = ss.fit_transform(df_train[self.variables["numerical"]])
        X_test_num = ss.transform(df_test[self.variables["numerical"]])

        X_train = np.hstack((X_train_cat, X_train_num))
        X_test = np.hstack((X_test_cat, X_test_num))

        self.train_val_sets[name] = (X_train, y_train, X_test, y_test)
        self.encoders["target"] = te
        self.encoders["standard"] = ss
        self.encoders["label"] = le


# Ordinal Encoding + Standard Scaling + Min-Max Scaling
class PreprocessorV6(BasePreprocessor):
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


# Target Encoder + Standard Scaling + Min-Max Scaling
class PreprocessorV7(BasePreprocessor):
    def fit(self, df_train: pd.DataFrame, df_test: pd.DataFrame, name: str):
        le = LabelEncoder()
        y_train = le.fit_transform(df_train[self.variables["target"]])
        y_test = le.transform(df_test[self.variables["target"]])

        te = TargetEncoder(target_type="binary", random_state=12345)
        X_train_cat = te.fit_transform(df_train[self.variables["categorical"]], y_train)
        X_test_cat = te.transform(df_test[self.variables["categorical"]])

        ss = StandardScaler()
        X_train_num = ss.fit_transform(df_train[self.variables["numerical"]])
        X_test_num = ss.transform(df_test[self.variables["numerical"]])

        X_train = np.hstack((X_train_cat, X_train_num))
        X_test = np.hstack((X_test_cat, X_test_num))

        mm = MinMaxScaler()
        X_train_scaled = mm.fit_transform(X_train)
        X_test_scaled = mm.transform(X_test)

        self.train_val_sets[name] = (X_train_scaled, y_train, X_test_scaled, y_test)
        self.encoders["target"] = te
        self.encoders["standard"] = ss
        self.encoders["min_max"] = mm
        self.encoders["label"] = le


# Ordinal Encoding + Standard Scaling + Min-Max Scaling
class PreprocessorReg(BasePreprocessor):
    def fit(self, df_train: pd.DataFrame, df_test: pd.DataFrame, name: str):
        oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        X_train_cat = oe.fit_transform(df_train[self.variables["categorical"]])
        X_test_cat = oe.transform(df_test[self.variables["categorical"]])

        ss = StandardScaler()
        X_train_num = ss.fit_transform(df_train[self.variables["numerical"]])
        X_test_num = ss.transform(df_test[self.variables["numerical"]])

        X_train = np.hstack((X_train_cat, X_train_num))
        X_test = np.hstack((X_test_cat, X_test_num))

        x_scaler = MinMaxScaler()
        X_train_scaled = x_scaler.fit_transform(X_train)
        X_test_scaled = x_scaler.transform(X_test)

        y_scaler = MinMaxScaler()
        y_train_scaled = y_scaler.fit_transform(df_train[self.variables["target"]].to_numpy().reshape(-1,1))
        # y_test = y_scaler.transform(df_test[self.variables["target"]].to_numpy().reshape(-1,1))

        self.train_val_sets[name] = (X_train_scaled, y_train_scaled, X_test_scaled, df_test[self.variables["target"]])
        self.encoders["ordinal"] = oe
        self.encoders["standard"] = ss
        self.encoders["x_min_max"] = x_scaler
        self.encoders["y_min_max"] = y_scaler
