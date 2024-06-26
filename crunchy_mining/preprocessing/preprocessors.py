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
        X_train_cat = oe.fit_transform(df_train[self.cfg.vars.categorical])
        X_test_cat = oe.transform(df_test[self.cfg.vars.categorical])

        X_train_num = df_train[self.cfg.vars.numerical]
        X_test_num = df_test[self.cfg.vars.numerical]

        X_train = np.hstack((X_train_cat, X_train_num))
        X_test = np.hstack((X_test_cat, X_test_num))

        le = LabelEncoder()
        y_train = le.fit_transform(df_train[self.cfg.vars.target])
        y_test = le.transform(df_test[self.cfg.vars.target])

        self.train_val_sets[name] = (X_train, y_train, X_test, y_test)
        self.encoders["ordinal"] = oe
        self.encoders["label"] = le


# One-Hot Encoding + No Scaling
class PreprocessorV2(BasePreprocessor):
    def fit(self, df_train: pd.DataFrame, df_test: pd.DataFrame, name: str):
        hot = OneHotEncoder(handle_unknown="infrequent_if_exist")
        X_train_cat = hot.fit_transform(df_train[self.cfg.vars.categorical])
        X_test_cat = hot.transform(df_test[self.cfg.vars.categorical])

        X_train_num = df_train[self.cfg.vars.numerical]
        X_test_num = df_test[self.cfg.vars.numerical]

        X_train = sp.hstack((X_train_cat, X_train_num), format="csr")
        X_test = sp.hstack((X_test_cat, X_test_num), format="csr")

        le = LabelEncoder()
        y_train = le.fit_transform(df_train[self.cfg.vars.target])
        y_test = le.transform(df_test[self.cfg.vars.target])

        self.train_val_sets[name] = (X_train, y_train, X_test, y_test)
        self.encoders["one_hot"] = hot
        self.encoders["label"] = le


# One-Hot Encoding (Dense) + No Scaling
class PreprocessorV2D(BasePreprocessor):
    def fit(self, df_train: pd.DataFrame, df_test: pd.DataFrame, name: str):
        hot = OneHotEncoder(sparse_output=False, handle_unknown="infrequent_if_exist")
        X_train_cat = hot.fit_transform(df_train[self.cfg.vars.categorical])
        X_test_cat = hot.transform(df_test[self.cfg.vars.categorical])

        X_train_num = df_train[self.cfg.vars.numerical]
        X_test_num = df_test[self.cfg.vars.numerical]

        X_train = np.hstack((X_train_cat, X_train_num))
        X_test = np.hstack((X_test_cat, X_test_num))

        le = LabelEncoder()
        y_train = le.fit_transform(df_train[self.cfg.vars.target])
        y_test = le.transform(df_test[self.cfg.vars.target])

        self.train_val_sets[name] = (X_train, y_train, X_test, y_test)
        self.encoders["one_hot"] = hot
        self.encoders["label"] = le


# Target Encoder + No Scaling
class PreprocessorV3(BasePreprocessor):
    def fit(self, df_train: pd.DataFrame, df_test: pd.DataFrame, name: str):
        # Encode the target first because the target encoder needs them.
        le = LabelEncoder()
        y_train = le.fit_transform(df_train[self.cfg.vars.target])
        y_test = le.transform(df_test[self.cfg.vars.target])

        te = TargetEncoder(target_type="binary", random_state=12345)
        X_train_cat = te.fit_transform(df_train[self.cfg.vars.categorical], y_train)
        X_test_cat = te.transform(df_test[self.cfg.vars.categorical])

        X_train_num = df_train[self.cfg.vars.numerical]
        X_test_num = df_test[self.cfg.vars.numerical]

        X_train = np.hstack((X_train_cat, X_train_num))
        X_test = np.hstack((X_test_cat, X_test_num))

        self.train_val_sets[name] = (X_train, y_train, X_test, y_test)
        self.encoders["target"] = te
        self.encoders["label"] = le

    def transform(self, X_new):
        te = self.encoders["target"]
        n_cat = len(self.cfg.vars.categorical)

        X_cat = te.transform(X_new[:, :n_cat].tolist())
        X_num = X_new[:, n_cat:]

        return np.hstack((X_cat, X_num))


# Ordinal Encoding + Standard Scaling
class PreprocessorV4(BasePreprocessor):
    def fit(self, df_train: pd.DataFrame, df_test: pd.DataFrame, name: str):
        oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        X_train_cat = oe.fit_transform(df_train[self.cfg.vars.categorical])
        X_test_cat = oe.transform(df_test[self.cfg.vars.categorical])

        ss = StandardScaler()
        X_train_num = ss.fit_transform(df_train[self.cfg.vars.numerical])
        X_test_num = ss.transform(df_test[self.cfg.vars.numerical])

        X_train = np.hstack((X_train_cat, X_train_num))
        X_test = np.hstack((X_test_cat, X_test_num))

        le = LabelEncoder()
        y_train = le.fit_transform(df_train[self.cfg.vars.target])
        y_test = le.transform(df_test[self.cfg.vars.target])

        self.train_val_sets[name] = (X_train, y_train, X_test, y_test)
        self.encoders["ordinal"] = oe
        self.encoders["standard"] = ss
        self.encoders["label"] = le


# Target Encoder + Standard Scaling
class PreprocessorV5(BasePreprocessor):
    def fit(self, df_train: pd.DataFrame, df_test: pd.DataFrame, name: str):
        le = LabelEncoder()
        y_train = le.fit_transform(df_train[self.cfg.vars.target])
        y_test = le.transform(df_test[self.cfg.vars.target])

        te = TargetEncoder(target_type="binary", random_state=12345)
        X_train_cat = te.fit_transform(df_train[self.cfg.vars.categorical], y_train)
        X_test_cat = te.transform(df_test[self.cfg.vars.categorical])

        ss = StandardScaler()
        X_train_num = ss.fit_transform(df_train[self.cfg.vars.numerical])
        X_test_num = ss.transform(df_test[self.cfg.vars.numerical])

        X_train = np.hstack((X_train_cat, X_train_num))
        X_test = np.hstack((X_test_cat, X_test_num))

        self.train_val_sets[name] = (X_train, y_train, X_test, y_test)
        self.encoders["target"] = te
        self.encoders["standard"] = ss
        self.encoders["label"] = le

    def transform(self, X_new):
        te = self.encoders["target"]
        ss = self.encoders["standard"]
        n_cat = len(self.cfg.vars.categorical)

        X_cat = te.transform(X_new[:, :n_cat].tolist())
        X_num = ss.transform(X_new[:, n_cat:])

        return np.hstack((X_cat, X_num))


# Ordinal Encoding + Standard Scaling + Min-Max Scaling
class PreprocessorV6(BasePreprocessor):
    def fit(self, df_train: pd.DataFrame, df_test: pd.DataFrame, name: str):
        oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        X_train_cat = oe.fit_transform(df_train[self.cfg.vars.categorical])
        X_test_cat = oe.transform(df_test[self.cfg.vars.categorical])

        ss = StandardScaler()
        X_train_num = ss.fit_transform(df_train[self.cfg.vars.numerical])
        X_test_num = ss.transform(df_test[self.cfg.vars.numerical])

        X_train = np.hstack((X_train_cat, X_train_num))
        X_test = np.hstack((X_test_cat, X_test_num))

        mm = MinMaxScaler()
        X_train_scaled = mm.fit_transform(X_train)
        X_test_scaled = mm.transform(X_test)

        le = LabelEncoder()
        y_train = le.fit_transform(df_train[self.cfg.vars.target])
        y_test = le.transform(df_test[self.cfg.vars.target])

        self.train_val_sets[name] = (X_train_scaled, y_train, X_test_scaled, y_test)
        self.encoders["ordinal"] = oe
        self.encoders["standard"] = ss
        self.encoders["min_max"] = mm
        self.encoders["label"] = le

    def transform(self, X_new):
        oe = self.encoders["ordinal"]
        ss = self.encoders["standard"]
        mm = self.encoders["min_max"]
        n_cat = len(self.cfg.vars.categorical)

        X_cat = oe.transform(X_new[:, :n_cat].tolist())
        X_num = ss.transform(X_new[:, n_cat:])

        X_new = np.hstack((X_cat, X_num))
        X_new_scaled = mm.transform(X_new)

        return X_new_scaled


# Target Encoder + Standard Scaling + Min-Max Scaling
class PreprocessorV7(BasePreprocessor):
    def fit(self, df_train: pd.DataFrame, df_test: pd.DataFrame, name: str):
        le = LabelEncoder()
        y_train = le.fit_transform(df_train[self.cfg.vars.target])
        y_test = le.transform(df_test[self.cfg.vars.target])

        te = TargetEncoder(target_type="binary", random_state=12345)
        X_train_cat = te.fit_transform(df_train[self.cfg.vars.categorical], y_train)
        X_test_cat = te.transform(df_test[self.cfg.vars.categorical])

        ss = StandardScaler()
        X_train_num = ss.fit_transform(df_train[self.cfg.vars.numerical])
        X_test_num = ss.transform(df_test[self.cfg.vars.numerical])

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

    def transform(self, X_new):
        te = self.encoders["target"]
        ss = self.encoders["standard"]
        mm = self.encoders["min_max"]
        n_cat = len(self.cfg.vars.categorical)

        X_cat = te.transform(X_new[:, :n_cat].tolist())
        X_num = ss.transform(X_new[:, n_cat:])

        X_new = np.hstack((X_cat, X_num))
        X_new_scaled = mm.transform(X_new)

        return X_new_scaled
