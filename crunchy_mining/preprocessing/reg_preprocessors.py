import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import TargetEncoder

from .base_preprocessor import BasePreprocessor


# Ordinal Encoding + No Scaling + No Scaling (T)
class PreprocessorV1(BasePreprocessor):
    def fit(self, df_train: pd.DataFrame, df_test: pd.DataFrame, name: str):
        oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        X_train_cat = oe.fit_transform(df_train[self.cfg.vars.categorical])
        X_test_cat = oe.transform(df_test[self.cfg.vars.categorical])

        X_train_num = df_train[self.cfg.vars.numerical]
        X_test_num = df_test[self.cfg.vars.numerical]

        X_train = np.hstack((X_train_cat, X_train_num))
        X_test = np.hstack((X_test_cat, X_test_num))

        y_train = df_train[self.cfg.vars.target].to_numpy()
        y_test = df_test[self.cfg.vars.target].to_numpy()

        self.train_val_sets[name] = (X_train, y_train, X_test, y_test)
        self.encoders["ordinal"] = oe


#  One-Hot Encoding + No Scaling + No Scaling (T)
class PreprocessorV2(BasePreprocessor):
    def fit(self, df_train: pd.DataFrame, df_test: pd.DataFrame, name: str):
        hot = OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=250)
        X_train_cat = hot.fit_transform(df_train[self.cfg.vars.categorical])
        X_test_cat = hot.transform(df_test[self.cfg.vars.categorical])

        X_train_num = df_train[self.cfg.vars.numerical]
        X_test_num = df_test[self.cfg.vars.numerical]

        X_train = sp.hstack((X_train_cat, X_train_num), format="csr")
        X_test = sp.hstack((X_test_cat, X_test_num), format="csr")

        y_train = df_train[self.cfg.vars.target].to_numpy()
        y_test = df_test[self.cfg.vars.target].to_numpy()

        self.train_val_sets[name] = (X_train, y_train, X_test, y_test)
        self.encoders["one_hot"] = hot


# Target Encoding + No Scaling + No Scaling (T)
class PreprocessorV3(BasePreprocessor):
    def fit(self, df_train: pd.DataFrame, df_test: pd.DataFrame, name: str):
        y_train = df_train[self.cfg.vars.target].to_numpy()
        y_test = df_test[self.cfg.vars.target].to_numpy()

        te = TargetEncoder(target_type="continuous", random_state=12345)
        X_train_cat = te.fit_transform(df_train[self.cfg.vars.categorical], y_train)
        X_test_cat = te.transform(df_test[self.cfg.vars.categorical])

        X_train_num = df_train[self.cfg.vars.numerical]
        X_test_num = df_test[self.cfg.vars.numerical]

        X_train = np.hstack((X_train_cat, X_train_num))
        X_test = np.hstack((X_test_cat, X_test_num))

        self.train_val_sets[name] = (X_train, y_train, X_test, y_test)
        self.encoders["target"] = te


# Ordinal Encoding + No Scaling + Min-Max Scaling (T)
class PreprocessorV4(BasePreprocessor):
    def fit(self, df_train: pd.DataFrame, df_test: pd.DataFrame, name: str):
        oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        X_train_cat = oe.fit_transform(df_train[self.cfg.vars.categorical])
        X_test_cat = oe.transform(df_test[self.cfg.vars.categorical])

        X_train_num = df_train[self.cfg.vars.numerical]
        X_test_num = df_test[self.cfg.vars.numerical]

        X_train = np.hstack((X_train_cat, X_train_num))
        X_test = np.hstack((X_test_cat, X_test_num))

        y_train = df_train[self.cfg.vars.target].to_numpy()
        y_test = df_test[self.cfg.vars.target].to_numpy()

        mm = MinMaxScaler()
        y_train = mm.fit_transform(y_train)

        self.train_val_sets[name] = (X_train, y_train, X_test, y_test)
        self.encoders["ordinal"] = oe
        self.encoders["y_min_max"] = mm


# Target Encoding + No Scaling + Min-Max Scaling (T)
class PreprocessorV5(BasePreprocessor):
    def fit(self, df_train: pd.DataFrame, df_test: pd.DataFrame, name: str):
        y_train = df_train[self.cfg.vars.target].to_numpy()
        y_test = df_test[self.cfg.vars.target].to_numpy()

        mm = MinMaxScaler()
        y_train = mm.fit_transform(y_train)

        te = TargetEncoder(target_type="continuous", random_state=12345)
        X_train_cat = te.fit_transform(df_train[self.cfg.vars.categorical], y_train)
        X_test_cat = te.transform(df_test[self.cfg.vars.categorical])

        X_train_num = df_train[self.cfg.vars.numerical]
        X_test_num = df_test[self.cfg.vars.numerical]

        X_train = np.hstack((X_train_cat, X_train_num))
        X_test = np.hstack((X_test_cat, X_test_num))

        self.train_val_sets[name] = (X_train, y_train, X_test, y_test)
        self.encoders["target"] = te
        self.encoders["y_min_max"] = mm


# Ordinal Encoding + No Scaling + Log Scaling (T)
class PreprocessorV6(BasePreprocessor):
    def fit(self, df_train: pd.DataFrame, df_test: pd.DataFrame, name: str):
        oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        X_train_cat = oe.fit_transform(df_train[self.cfg.vars.categorical])
        X_test_cat = oe.transform(df_test[self.cfg.vars.categorical])

        X_train_num = df_train[self.cfg.vars.numerical]
        X_test_num = df_test[self.cfg.vars.numerical]

        X_train = np.hstack((X_train_cat, X_train_num))
        X_test = np.hstack((X_test_cat, X_test_num))

        y_train = np.log1p(df_train[self.cfg.vars.target].to_numpy())
        y_test = df_test[self.cfg.vars.target].to_numpy()

        self.train_val_sets[name] = (X_train, y_train, X_test, y_test)
        self.encoders["ordinal"] = oe
        self.encoders["y_log"] = True


# Target Encoding + No Scaling + Log Scaling (T)
class PreprocessorV7(BasePreprocessor):
    def fit(self, df_train: pd.DataFrame, df_test: pd.DataFrame, name: str):
        y_train = np.log1p(df_train[self.cfg.vars.target].to_numpy())
        y_test = df_test[self.cfg.vars.target].to_numpy()

        te = TargetEncoder(target_type="continuous", random_state=12345)
        X_train_cat = te.fit_transform(df_train[self.cfg.vars.categorical], y_train)
        X_test_cat = te.transform(df_test[self.cfg.vars.categorical])

        X_train_num = df_train[self.cfg.vars.numerical]
        X_test_num = df_test[self.cfg.vars.numerical]

        X_train = np.hstack((X_train_cat, X_train_num))
        X_test = np.hstack((X_test_cat, X_test_num))

        self.train_val_sets[name] = (X_train, y_train, X_test, y_test)
        self.encoders["target"] = te
        self.encoders["y_log"] = True

