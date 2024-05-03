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
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math

# %%
df = pd.read_csv("data/SBA.csv", low_memory=False)

# %%
cleaned_df = pd.read_parquet("data/output.parquet")

# %%
cleaned_df.info()

# %%
int_columns = df.select_dtypes(include=['int64']).columns.tolist()

# %%
# Before cleaning
num_rows = math.ceil(len(int_columns) / 3)

fig, axes = plt.subplots(num_rows, 3, figsize=(15, 5*num_rows))

for i, feature in enumerate(int_columns):
    row = i // 3
    col = i % 3
    sns.boxplot(y=feature, data=df, ax=axes[row, col])
    axes[row, col].set_title(feature)

plt.tight_layout()
plt.show()

# %%
# Count of MIS_Status (Before cleaning)
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='MIS_Status', hue='MIS_Status', palette='pastel')
plt.title('Count of Each Unique Value in MIS_Status (Before cleaning)')
plt.xlabel('MIS_Status')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

# %%
# Count of MIS_Status (After cleaning)
plt.figure(figsize=(8, 6))
sns.countplot(data=cleaned_df, x='MIS_Status', hue='MIS_Status', palette='pastel')
plt.title('Count of Each Unique Value in MIS_Status (After cleaning)')
plt.xlabel('MIS_Status')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

# %%
currency_col = [
    "DisbursementGross",
    "BalanceGross",
    "ChgOffPrinGr",
    "GrAppv",
    "SBA_Appv",
]

# %%
sns.pairplot(data=cleaned_df, vars=currency_col, hue='MIS_Status', palette="husl")
plt.show()

# %%
# From cleaned df
cat_columns = ['Is_Existing', 'Is_Franchised', 'Is_CreatedJob', 'Is_RetainedJob', 'UrbanRural', 'RevLineCr', 'LowDoc']

for column in cat_columns:
    print(f"Unique values for column '{column}':")

    unique_values = cleaned_df[column].unique()
    for value in unique_values:
        print(value)
    
    print()

# %%
status_labels = {0: 'PIF', 1: 'CHGOFF'}

num_rows = (len(cat_columns) + 1) // 2
fig, axes = plt.subplots(num_rows, 2, figsize=(15, 6*num_rows))
axes = axes.flatten()

for i, column in enumerate(cat_columns):
    sns.countplot(data=cleaned_df, x=column, hue='MIS_Status', palette='pastel', ax=axes[i])
    axes[i].set_title(f'Distribution of MIS_Status using {column}')
    axes[i].set_xlabel(column)
    axes[i].set_ylabel('MIS_Status')
    
    handles, labels = axes[i].get_legend_handles_labels()
    axes[i].legend(handles, [status_labels[int(label)] for label in labels], title='MIS_Status')

for j in range(i+1, len(axes)):
    axes[j].axis('off')
    
plt.tight_layout()
plt.show()
