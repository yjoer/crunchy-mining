# %%
import altair as alt
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import (
    LabelEncoder,
    MinMaxScaler,
    OrdinalEncoder,
    StandardScaler,
)
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

# %%
df = pd.read_parquet("data/clean.parquet")

# %%
df.info()

# %% [markdown]
# ## Sample

# %%
counts = df["MIS_Status"].value_counts()
majority_class = counts.index[np.argmax(counts)]
minority_class = counts.index[np.argmin(counts)]
n_minority_class = np.min(counts)

# %%
df_sampled = pd.concat(
    [
        df[df["MIS_Status"] == majority_class].sample(
            n_minority_class,
            random_state=12345,
        ),
        df[df["MIS_Status"] == minority_class],
    ]
)

# %%
# Should the year be categorical or numerical?
# How to deal with dates?
variables = {
    "categorical": [
        "City",
        "State",
        "Zip",
        "Bank",
        "BankState",
        "ApprovalFY",
        "NewExist",
        "Is_Franchised",
        "UrbanRural",
        "RevLineCr",
        "LowDoc",
        "Industry",
        "RealEstate",
        "Recession",
        "DisbursementFY",
        "StateSame",
        "SBA_AppvPct",
    ],
    "numerical": [
        "Term",
        "NoEmp",
        "CreateJob",
        "RetainedJob",
        "DisbursementGross",
        "BalanceGross",
        "ChgOffPrinGr",
        "GrAppv",
        "SBA_Appv",
        "DaysTerm",
        "DaysToDisbursement",
    ],
    "target": "MIS_Status",
}

# %%
# Do we leak future information if we ignore the application date?
df_train, df_test = train_test_split(
    df_sampled,
    test_size=0.15,
    random_state=12345,
    stratify=df_sampled[variables["target"]],
)

# %%
pd.concat(
    [
        df_train[variables["target"]].value_counts(),
        df_test[variables["target"]].value_counts(),
    ]
)

# %%
df_train_sm, df_val = train_test_split(
    df_train,
    test_size=0.15 / 0.85,
    random_state=12345,
    stratify=df_train[variables["target"]],
)

# %%
pd.concat(
    [
        df_train_sm[variables["target"]].value_counts(),
        df_val[variables["target"]].value_counts(),
    ]
)

# %% [markdown]
# ## Modify

# %%
# What's the better way to handle categorical variables without one-hot encoding to
# avoid the curse of dimensionality?
oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
X_train_sm_cat = oe.fit_transform(df_train_sm[variables["categorical"]])
X_val_cat = oe.transform(df_val[variables["categorical"]])
X_test_cat = oe.transform(df_test[variables["categorical"]])

# %%
# Do we need to scale the outputs of the ordinal encoder?
mm = MinMaxScaler()
X_train_sm_cat_scaled = mm.fit_transform(X_train_sm_cat)
X_val_cat_scaled = mm.transform(X_val_cat)
X_test_cat_scaled = mm.transform(X_test_cat)

# %%
ss = StandardScaler()
X_train_sm_num = ss.fit_transform(df_train_sm[variables["numerical"]])
X_val_num = ss.transform(df_val[variables["numerical"]])
X_test_num = ss.transform(df_test[variables["numerical"]])

# %%
X_train_sm_ft = np.hstack((X_train_sm_cat_scaled, X_train_sm_num))
X_val_ft = np.hstack((X_val_cat_scaled, X_val_num))
X_test_ft = np.hstack((X_test_cat_scaled, X_test_num))

# %%
le = LabelEncoder()
y_train_sm = le.fit_transform(df_train_sm[variables["target"]])
y_val = le.transform(df_val[variables["target"]])
y_test = le.transform(df_test[variables["target"]])

# %% [markdown]
# ## Model

# %%
knn = KNeighborsClassifier()
knn.fit(X_train_sm_ft, y_train_sm)

# %%
logreg = LogisticRegression(random_state=12345, n_jobs=-1)
logreg.fit(X_train_sm_ft, y_train_sm)

# %%
gnb = GaussianNB()
gnb.fit(X_train_sm_ft, y_train_sm)

# %%
svc = LinearSVC(dual="auto", random_state=12345)
svc.fit(X_train_sm_ft, y_train_sm)

# %%
dt = DecisionTreeClassifier(random_state=12345)
dt.fit(X_train_sm_ft, y_train_sm)

# %%
ab = AdaBoostClassifier(algorithm="SAMME", random_state=12345)
ab.fit(X_train_sm_ft, y_train_sm)

# %%
rf = RandomForestClassifier(n_jobs=-1, random_state=12345)
rf.fit(X_train_sm_ft, y_train_sm)

# %%
xgb = XGBClassifier(n_jobs=-1, random_state=12345)
xgb.fit(X_train_sm_ft, y_train_sm)

# %%
lgb = LGBMClassifier(random_state=12345, n_jobs=-1)
lgb.fit(X_train_sm_ft, y_train_sm)

# %%
catb = CatBoostClassifier(metric_period=250, random_state=12345)
catb.fit(X_train_sm_ft, y_train_sm)

# %% [markdown]
# ## Assess

# %%
y_knn = knn.predict(X_val_ft)
print(classification_report(y_val, y_knn))

# %%
y_logreg = logreg.predict(X_val_ft)
print(classification_report(y_val, y_logreg))

# %%
y_gnb = gnb.predict(X_val_ft)
print(classification_report(y_val, y_gnb))

# %%
y_svc = svc.predict(X_val_ft)
print(classification_report(y_val, y_svc))

# %%
y_dt = dt.predict(X_val_ft)
print(classification_report(y_val, y_dt))

# %%
y_ab = ab.predict(X_val_ft)
print(classification_report(y_val, y_ab))

# %%
y_rf = rf.predict(X_val_ft)
print(classification_report(y_val, y_rf))

# %%
y_xgb = xgb.predict(X_val_ft)
print(classification_report(y_val, y_xgb))

# %%
y_lgb = lgb.predict(X_val_ft)
print(classification_report(y_val, y_lgb))

# %%
y_catb = catb.predict(X_val_ft)
print(classification_report(y_val, y_catb))

# %%
