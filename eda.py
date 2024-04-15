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
cleaned_df.info()

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
plt.ylabel('Total Case', fontsize=15)
plt.legend(["PIF", "CHGOFF"],loc='upper right')
plt.xticks(rotation='vertical');

# %%
#Total PIF vs Total CHGOFF based on Year Table
count_table = cleaned_df.groupby(["ApprovalFY", "MIS_Status"]).size().reset_index(name='Count')
pivot_table = count_table.pivot_table(index='ApprovalFY', columns='MIS_Status', values='Count', fill_value=0)
pivot_table_sorted = pivot_table.sort_index()
print(pivot_table_sorted)

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
f, ax = plt.subplots(figsize=(16,9))
sns.countplot(y="Industry", hue="Recession", data=cleaned_df)
plt.title('Active Case during Recession based on Industry', fontsize=20)
plt.xlabel('Number of Active Case', fontsize=15)
plt.ylabel('Industry', fontsize=15)
plt.legend(["Inactive", "Active"],loc='lower right')

# %%
f, ax = plt.subplots(figsize=(16,9))
sns.countplot(y="Industry", hue="RealEstate", data=cleaned_df)
plt.title('Cases with Real Estate based on Industry', fontsize=20)
plt.xlabel('Number of Cases', fontsize=15)
plt.ylabel('Industry', fontsize=15)
plt.legend(["No", "Yes"],loc='lower right')
plt.show()

# %%
pd.DataFrame(cleaned_df.groupby('Industry')['RealEstate'].value_counts()).unstack(level=1).style.highlight_max(color='green').highlight_min(color='blue')

# %%
f, ax = plt.subplots(figsize=(16,9))
sns.boxplot(cleaned_df['Term'])
plt.title('Term', fontsize=20)
plt.xlabel('Month', fontsize=15)

# %%
cleaned_df['Term'].describe()

# %%
f, ax = plt.subplots(figsize=(16,9))
sns.boxplot(x="Term", y="Industry", data=cleaned_df)
plt.title('Term', fontsize=20)
plt.xlabel('Month', fontsize=15)
plt.ylabel('Industry', fontsize=15)

# %%
cleaned_df.groupby('Industry')['Term'].describe().style.highlight_max(color='green').highlight_min(color='blue')

# %%
f, ax = plt.subplots(figsize=(16,9))
sns.countplot(x="ApprovalFY", data=cleaned_df,hue='NewExist')
plt.title('Total New Business based on Year', fontsize=20)
plt.xlabel('Year', fontsize=15)
plt.ylabel('Number of Business', fontsize=15)
plt.legend(["Existing", "New"],loc='upper right')
plt.xticks(rotation='vertical');

# %%
f, ax = plt.subplots(figsize=(16,9))
sns.countplot(y="Industry", hue="NewExist", data=cleaned_df)
plt.title('Total New Business based on Industry', fontsize=20)
plt.xlabel('Number of Business', fontsize=15)
plt.ylabel('Industry', fontsize=15)
plt.legend(["Existing", "New"],loc='lower right')


# %% [markdown]
# Handle Outlier using Inter Quartile Range

# %%
def limit(i):
    Q1 = cleaned_df[i].quantile(0.25)
    Q3 = cleaned_df[i].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_limit = cleaned_df[i].quantile(0.25) - (IQR * 1.5)
    lower_limit_extreme = cleaned_df[i].quantile(0.25) - (IQR * 3)
    upper_limit = cleaned_df[i].quantile(0.75) + (IQR * 1.5)
    upper_limit_extreme = cleaned_df[i].quantile(0.75) + (IQR * 3)
    print('Lower Limit:', lower_limit)
    print('Lower Limit Extreme:', lower_limit_extreme)
    print('Upper Limit:', upper_limit)
    print('Upper Limit Extreme:', upper_limit_extreme)

def percent_outliers(i):
    Q1 = cleaned_df[i].quantile(0.25)
    Q3 = cleaned_df[i].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_limit = cleaned_df[i].quantile(0.25) - (IQR * 1.5)
    lower_limit_extreme = cleaned_df[i].quantile(0.25) - (IQR * 3)
    upper_limit = cleaned_df[i].quantile(0.75) + (IQR * 1.5)
    upper_limit_extreme = cleaned_df[i].quantile(0.75) + (IQR * 3)

    print('Lower Limit: {} %'.format(cleaned_df[(cleaned_df[i] >= lower_limit)].shape[0]/ cleaned_df.shape[0]*100))
    print('Lower Limit Extereme: {} %'.format(cleaned_df[(cleaned_df[i] >= lower_limit_extreme)].shape[0]/cleaned_df.shape[0]*100))
    print('Upper Limit: {} %'.format(cleaned_df[(cleaned_df[i] >= upper_limit)].shape[0]/ cleaned_df.shape[0]*100))
    print('Upper Limit Extereme: {} %'.format(cleaned_df[(cleaned_df[i] >= upper_limit_extreme)].shape[0]/cleaned_df.shape[0]*100))


# %%
f, ax = plt.subplots(figsize=(16,9))
sns.boxplot(x=cleaned_df['DisbursementGross'])
plt.title('DisbursementGross Ouliers', fontsize=20)
plt.xlabel('Total', fontsize=15)

# %%
plt.figure(figsize=(16, 9))

sns.kdeplot(cleaned_df['DisbursementGross'], shade=True)

mean_val = np.mean(cleaned_df['DisbursementGross'])
median_val = np.median(cleaned_df['DisbursementGross'])
mode_val = cleaned_df['DisbursementGross'].mode().values[0]

plt.axvline(mean_val, color='r', linestyle='--', label=f'Mean: {mean_val:.2f}')
plt.axvline(median_val, color='g', linestyle='--', label=f'Median: {median_val:.2f}')
plt.axvline(mode_val, color='b', linestyle='--', label=f'Mode: {mode_val:.2f}')

plt.title('DisbursementGross Distribution with Mean, Median, and Mode', fontsize=20)
plt.xlabel('DisbursementGross', fontsize=15)
plt.ylabel('Density', fontsize=15)

plt.legend()
plt.show()

# %%
plt.figure(figsize=(16, 9))
sns.histplot(cleaned_df['DisbursementGross'], kde=False, bins=30)
plt.title('DisbursementGross Distribution', fontsize=20)
plt.xlabel('DisbursementGross', fontsize=15)
plt.ylabel('Frequency', fontsize=15)
plt.show()

# %%
print(limit('DisbursementGross'))
print('-'*50)
print(percent_outliers('DisbursementGross'))

# %%
cleaned_df['DisbursementGross'] = np.log(cleaned_df['DisbursementGross'])
cleaned_df['DisbursementGross'].skew()

# %%
print(limit('DisbursementGross'))
print('-'*50)
print(percent_outliers('DisbursementGross'))

# %%
outliers1_drop = cleaned_df[(cleaned_df['DisbursementGross'] > 14.7)].index
cleaned_df.drop(outliers1_drop, inplace=True)

# %%
f, ax = plt.subplots(figsize=(16,9))
sns.boxplot(x=cleaned_df['DisbursementGross'])
plt.title('DisbursementGross Ouliers', fontsize=20)
plt.xlabel('Total', fontsize=15)

# %%
plt.figure(figsize=(16, 9))

sns.kdeplot(cleaned_df['DisbursementGross'], shade=True)

mean_val = np.mean(cleaned_df['DisbursementGross'])
median_val = np.median(cleaned_df['DisbursementGross'])
mode_val = cleaned_df['DisbursementGross'].mode().values[0]

plt.axvline(mean_val, color='r', linestyle='--', label=f'Mean: {mean_val:.2f}')
plt.axvline(median_val, color='g', linestyle='--', label=f'Median: {median_val:.2f}')
plt.axvline(mode_val, color='b', linestyle='--', label=f'Mode: {mode_val:.2f}')

plt.title('DisbursementGross Distribution with Mean, Median, and Mode', fontsize=20)
plt.xlabel('DisbursementGross', fontsize=15)
plt.ylabel('Density', fontsize=15)

plt.legend()
plt.show()

# %%
