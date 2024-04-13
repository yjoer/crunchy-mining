# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import altair as alt
import numpy as np
import pandas as pd
from numpy import argmax
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import math
from datetime import date
from scipy import stats
import nltk
from nltk.tokenize import word_tokenize
from collections import Counter
import warnings
nltk.download('punkt')
warnings.filterwarnings("ignore")

# %%
cleaned_df = pd.read_csv("data/output.csv")

# %%
f, ax = plt.subplots(figsize=(16,9))
sns.barplot(x="ApprovalFY", y="DisbursementGross", color='Salmon', data=cleaned_df)
plt.title('Total Loan Based on Year', fontsize=20)
plt.xlabel('Year', fontsize=15)
plt.ylabel('Total loan ($)', fontsize=15)
plt.xticks(rotation='vertical');

# %%
cleaned_df.DisbursementGross.describe()

# %%
f, ax = plt.subplots(figsize=(16,9))
sns.barplot(x="DisbursementGross", y="Industry", data=cleaned_df)
plt.title('Total Loan Based on Industry', fontsize=20)
plt.xlabel('Total Loan ($)', fontsize=15)
plt.ylabel('Industry', fontsize=15)

# %%
cleaned_df.groupby('Industry')['DisbursementGross'].describe().style.highlight_max(color='green').highlight_min(color='blue')

# %%
f, ax = plt.subplots(figsize=(16,9))
sns.countplot(x="ApprovalFY", data=cleaned_df,hue='MIS_Status')
plt.title('Total PIF vs Total CHGOFF based on Year', fontsize=20)
plt.xlabel('Year', fontsize=15)
plt.ylabel('Total Loan($)', fontsize=15)
plt.legend(["PIF", "CHGOFF"],loc='upper right')
plt.xticks(rotation='vertical');

# %%
f, ax = plt.subplots(figsize=(16,9))
sns.countplot(y="Industry", hue="MIS_Status", data=cleaned_df)
plt.title('Total PIF vs Total CHGOFF based on Industry', fontsize=20)
plt.xlabel('Total CHGOFF', fontsize=15)
plt.ylabel('Industry', fontsize=15)
plt.legend(["PIF", "CHGOFF"],loc='lower right')

# %%
pd.DataFrame(cleaned_df.groupby('Industry')['MIS_Status'].value_counts()).unstack(level=1).style.highlight_max(color='green').highlight_min(color='blue')

# %%
cleaned_df.info()

# %%
