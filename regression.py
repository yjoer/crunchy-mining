# %%
import os
import warnings

import mlflow
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from hydra import compose
from hydra import initialize
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

from crunchy_mining.pipeline import inspect_cv_split_size
from crunchy_mining.pipeline import inspect_holdout_split_size
from crunchy_mining.preprocessing.reg_preprocessors import PreprocessorV1
from crunchy_mining.preprocessing.reg_preprocessors import PreprocessorV2
from crunchy_mining.preprocessing.reg_preprocessors import PreprocessorV3
from crunchy_mining.preprocessing.reg_preprocessors import PreprocessorV4
from crunchy_mining.preprocessing.reg_preprocessors import PreprocessorV5
from crunchy_mining.preprocessing.reg_preprocessors import PreprocessorV6
from crunchy_mining.preprocessing.reg_preprocessors import PreprocessorV7
from crunchy_mining.reg_pipeline import validate_adaboost
from crunchy_mining.reg_pipeline import validate_catboost
from crunchy_mining.reg_pipeline import validate_decision_tree
from crunchy_mining.reg_pipeline import validate_elastic_net
from crunchy_mining.reg_pipeline import validate_lasso
from crunchy_mining.reg_pipeline import validate_lightgbm
from crunchy_mining.reg_pipeline import validate_linear_regression
from crunchy_mining.reg_pipeline import validate_random_forest
from crunchy_mining.reg_pipeline import validate_ridge
from crunchy_mining.reg_pipeline import validate_xgboost
from crunchy_mining.sampling.reg_samplers import SamplerV1
from crunchy_mining.sampling.reg_samplers import SamplerV2

load_dotenv()

# %load_ext autoreload
# %autoreload 2

mlflow.set_tracking_uri("http://localhost:5001")

warnings.filterwarnings(
    action="ignore",
    message=".*Distutils was imported before Setuptools.*",
)

warnings.filterwarnings(
    action="ignore",
    message=".*Setuptools is replacing distutils.*",
)

warnings.filterwarnings(
    action="ignore",
    message=".*Attempting to set identical low and high xlims.*",
)

# %%
experiment = os.environ.get("CM_EXPERIMENT", "bank/sampling_v1")
task_name, experiment_file = experiment.split("/")

with initialize(version_base=None, config_path="conf"):
    cfg = compose(overrides=[f"+experiment_{task_name}={experiment_file}"])

# %%
cfg

# %%
mlflow.set_experiment(cfg.mlflow.experiment_name)

# %%
df = pd.read_parquet("data/output.parquet")

# %%
df.info()

# %% [markdown]
# ## Sample

# %%
match cfg.sampling.variant:
    case 1:
        sampler = SamplerV1(cfg)
    case 2:
        sampler = SamplerV2(cfg)

sampler.sample(df)
train_val_sets_raw = sampler.train_val_sets

# %%
inspect_holdout_split_size(train_val_sets_raw, cfg.vars.stratify)

# %%
inspect_cv_split_size(train_val_sets_raw, cfg.vars.stratify)

# %% [markdown]
# ## Modify

# %% [markdown]
# ### Preprocessing

# %%
match cfg.preprocessing.variant:
    case 1:
        preprocessor = PreprocessorV1(cfg)
    case 2:
        preprocessor = PreprocessorV2(cfg)
    case 3:
        preprocessor = PreprocessorV3(cfg)
    case 4:
        preprocessor = PreprocessorV4(cfg)
    case 5:
        preprocessor = PreprocessorV5(cfg)
    case 6:
        preprocessor = PreprocessorV6(cfg)
    case 7:
        preprocessor = PreprocessorV7(cfg)

for name, (df_train, df_val) in train_val_sets_raw.items():
    preprocessor.fit(df_train, df_val, name=name)

# "name": (X_train, y_train, X_val, y_val)
preprocessor.save_train_val_sets()
preprocessor.save_encoders()

# %% [markdown]
# ## Model

# %% [markdown]
# ### Cross-Validation

# %%
if cfg.validation.models.linear_regression:
    validate_linear_regression(preprocessor)

# %%
if cfg.validation.models.lasso:
    validate_lasso(preprocessor)

# %%
if cfg.validation.models.ridge:
    validate_ridge(preprocessor)

# %%
if cfg.validation.models.elastic_net:
    validate_elastic_net(preprocessor)

# %%
if cfg.validation.models.decision_tree:
    validate_decision_tree(preprocessor)

# %%
if cfg.validation.models.random_forest:
    validate_random_forest(preprocessor)

# %%
if cfg.validation.models.adaboost:
    validate_adaboost(preprocessor)

# %%
if cfg.validation.models.xgboost:
    validate_xgboost(preprocessor)

# %%
if cfg.validation.models.lightgbm:
    validate_lightgbm(preprocessor)

# %%
if cfg.validation.models.catboost:
    validate_catboost(preprocessor)

# %%
