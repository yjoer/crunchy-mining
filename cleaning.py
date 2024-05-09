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
import datetime
import math
from collections import Counter
from datetime import date

import altair as alt
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import seaborn as sns
from nltk.tokenize import word_tokenize
from numpy import argmax
from scipy import stats
from scipy.stats import skew
from sklearn.preprocessing import OrdinalEncoder

pd.set_option("future.no_silent_downcasting", True)
pd.set_option("mode.copy_on_write", True)
nltk.download("punkt")

# %%
df = pd.read_csv("data/SBA.csv", low_memory=False)
df.info()

# %%
vw = df[:]

# %% [markdown]
# ## Modify

# %% [markdown]
# ### Handle Inconsistent Data

# %%
# Identify the values leading to mixed type
non_numeric_mask = pd.to_numeric(vw["ApprovalFY"], errors="coerce").isna()
vw.loc[non_numeric_mask, "ApprovalFY"].unique()

# %%
# Year to Int
vw["ApprovalFY"] = vw["ApprovalFY"].replace("1976A", 1976).astype(int)

# %%
# 1 = New; 2 = Exist
# We dont make assumption since this column might be one of the important factors
vw["NewExist"].value_counts(dropna=False)

# %%
vw = vw[vw["NewExist"].isin([1, 2])]
vw["NewExist"] = vw["NewExist"].astype(int)

# %%
# For RevLineCr, Based on the data description, only Y and N, thus we will ingore others
vw["RevLineCr"].value_counts(dropna=False)

# %%
vw = vw[vw["RevLineCr"].isin(["Y", "N"])]

# %%
# LowDoc valid input only Y or N
vw["LowDoc"].value_counts()

# %%
null_mask = vw["LowDoc"].isnull()
vw = vw[vw["LowDoc"].isin(["Y", "N"]) | null_mask]

# %%
vw["MIS_Status"].unique()

# %% [markdown]
# ### Remove Duplicates

# %%
# Check duplicate row
duplicate_rows = vw[vw.duplicated()]

if not duplicate_rows.empty:
    print("Duplicate rows found:")
    print(duplicate_rows)
else:
    print("No duplicate rows found.")

# %% [markdown]
# ### Convert Data Types

# %%
# To Date
date_col = ["ApprovalDate", "ChgOffDate", "DisbursementDate"]
vw[date_col] = pd.to_datetime(vw[date_col].stack(), format="%d-%b-%y").unstack()

# %%
# Tranform Data With String to Float
currency_col = [
    "DisbursementGross",
    "BalanceGross",
    "ChgOffPrinGr",
    "GrAppv",
    "SBA_Appv",
]
vw[currency_col] = vw[currency_col].replace("[\$,]", "", regex=True).astype(float)

# %%
df["DisbursementFY"] = df["DisbursementDate"].map(lambda x: x.year)

# %%
# NAICS, the first 2 numbers represent the Industry
df["NAICS"] = df["NAICS"].astype(str).str[:2]
df["NAICS"]

# %%
df.info()

# %% [markdown]
# Exploration:

# %%
# separate columns by data type (Numerical and Categorical Data)
numerical_cols = df.select_dtypes(include=["int64", "float64"]).columns
categorical_cols = df.select_dtypes(include=["object"]).columns

# create new data frames with only numerical and categorical columns
numerical_data = df[numerical_cols]
categorical_data = df[categorical_cols]

# %%
numerical_data.head()

# %%
fig, ax = plt.subplots(8, 2, figsize=(20, 40))
ax = ax.flatten()

right_skew_count = 0
left_skew_count = 0

for i, col in enumerate(numerical_cols):
    ax[i].hist(numerical_data[col], bins=30, density=True, alpha=0.5, color="green")
    ax[i].set_xlabel(col)
    ax[i].set_ylabel("Probability density")

    skewness = skew(numerical_data[col])
    # left skew, blue colour, if right skew, red colour
    if skewness > 0:
        right_skew_count += 1
        ax[i].axvline(
            np.mean(numerical_data[col]),
            color="r",
            linestyle="--",
            label="Mean",
        )
    elif skewness < 0:
        left_skew_count += 1
        ax[i].axvline(
            np.median(numerical_data[col]),
            color="b",
            linestyle="--",
            label="Median",
        )

    sns.kdeplot(numerical_data[col], ax=ax[i], color="blue", linewidth=2)

# Show the plot
plt.tight_layout()
plt.show()
print(f"Number of columns with right skew: {right_skew_count}")
print(f"Number of columns with left skew: {left_skew_count}")

# %%
categorical_data.head()

# %%
# Get unique values
unique_value = list()

for i in categorical_cols:
    unique_value.append(categorical_data[i].unique())

unique_values_tables = {
    "Categorical Columns": categorical_cols,
    "Unique Value": unique_value,
}
unique_values_tables = pd.DataFrame(unique_values_tables)
unique_values_tables

# %%
# Plot count distribution for each categorical column
fig, ax = plt.subplots(3, 2, figsize=(20, 40))
ax = ax.flatten()

temp_categorical_cols = ["State", "BankState", "LowDoc", "MIS_Status", "NAICS"]
for i, col in enumerate(temp_categorical_cols):
    sns.countplot(x=col, data=categorical_data, ax=ax[i])
    ax[i].set_xlabel(col)
    ax[i].set_ylabel("Count")

# Show the plot
plt.tight_layout()
plt.show()

# %%
plt.rcParams.update({"font.size": 18})

fig, ax = plt.subplots(5, 2, figsize=(20, 40))

numerical_var_count = 0
outlier_vars = []

for i in range(ax.shape[0]):
    for j in range(ax.shape[1]):

        k = numerical_cols[numerical_var_count]

        sns.boxplot(x=numerical_data[k], ax=ax[i, j])
        ax[i, j].set(xlabel=k, ylabel=k)
        ax[i, j].set(title="Boxplot of " + k)

        Q1 = numerical_data[k].quantile(0.25)
        Q3 = numerical_data[k].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        if any(numerical_data[k] < lower) or any(numerical_data[k] > upper):
            outlier_vars.append(k)

        numerical_var_count = numerical_var_count + 1

plt.tight_layout()

# %%
# Perform ordinal encoding on categorical columns
encoder = OrdinalEncoder()
dataset_encode = df.copy()
categorical_cols_encode = categorical_cols.copy()

dataset_encode[categorical_cols_encode] = encoder.fit_transform(
    dataset_encode[categorical_cols_encode]
)

# Print the updated dataset
dataset_encode

# %% [markdown]
# Correlation

# %%
pearsoncorr = dataset_encode.corr(method="pearson")
pearsoncorr = pearsoncorr[
    ((pearsoncorr >= 0.5) | (pearsoncorr <= -0.5)) & (pearsoncorr != 1.000)
]
plt.figure(figsize=(100, 100))
sns.heatmap(
    dataset_encode.corr(method="pearson"),
    xticklabels=pearsoncorr.columns,
    yticklabels=pearsoncorr.columns,
    cmap="RdBu_r",
    annot=True,
    linewidth=0.4,
)

# %%
corr = dataset_encode.corr().abs()
corr[corr == 1] = 0
corr_cols = corr.max().sort_values(ascending=False)
print("Correlation > 0.8: ")
display(corr_cols[corr_cols > 0.8])

dfCorr = dataset_encode.corr()
filteredDf = dfCorr[((dfCorr >= 0.8) | (dfCorr <= -0.8)) & (dfCorr != 1.000)]
plt.figure(figsize=(100, 100))
sns.heatmap(filteredDf, annot=True, cmap="Reds", linecolor="black", linewidths=1)
plt.show()

# %% [markdown]
# Observation:

# %% [markdown]
# M: Modification (Data Cleaning)

# %%
df.head()

# %%
df.shape

# %%
df.describe()

# %%
df.describe(include="O")

# %% [markdown]
# ### Handle Missing Values

# %%
# Check Null
vw.isnull().sum()

# %% [markdown]
# #### Deletion

# %%
# Drop DisbursementDate with NA
vw.dropna(subset=["DisbursementDate"], how="all", inplace=True)

# %% [markdown]
# #### Imputation

# %%
vw.fillna({"Name": "Unknown Company"}, inplace=True)

# %%
vw.fillna({"City": "Unknown City"}, inplace=True)

# %%
temp_empty_state = vw.loc[vw["State"].isnull()]
temp_empty_state

# %%
vw.loc[vw["State"].isnull(), "Zip"].unique()

# %%
# Fill the null 'State' based on the zip code group
vw.sort_values(by="Zip", inplace=True)
vw["State"] = vw.groupby("Zip")["State"].ffill()
vw.sort_index(inplace=True)

# %%
# To cross check the imputation Result for State
if not temp_empty_state.empty:
    null_indices = temp_empty_state.index
    imputed_rows = vw.loc[null_indices, ["City", "State", "Zip", "BankState"]]
    print("Table with imputed data for rows with null values in the 'State' column:")
    print(imputed_rows)
else:
    print("No rows with null values in the 'State' column.")

# %%
# Still got one State is NA, We will fill in manually based on Zip Code
vw.fillna({"State": "AP"}, inplace=True)

# %%
# Fill in NA Bank
vw.fillna({"Bank": "Unknown Bank"}, inplace=True)

# %%
vw[vw["BankState"].isnull()]

# %%
# Fill in Bank State based on Bank
vw.sort_values(by="Bank", inplace=True)
vw["BankState"] = vw.groupby("Bank")["BankState"].ffill()
vw.sort_index(inplace=True)

# %%
# Drop the BankState NA Row since we cant do any imputation
vw.dropna(subset=["BankState"], how="all", inplace=True)

# %%
# For MIS_Status, if got charge-off date, we will fill in charge off, others we cannot
# impute, we will drop it
possible_charge_off = (
    (vw["MIS_Status"].isnull())
    & (vw["ChgOffDate"].notnull())
    & (vw["ChgOffPrinGr"] > 0)
)

vw["MIS_Status"] = np.where(possible_charge_off, "CHGOFF", vw["MIS_Status"])

# %%
vw.dropna(subset=["MIS_Status"], how="all", inplace=True)

# %%
# Guideline: LowDoc(Y:Yes, N:No): In order to process more loans efficiently, a
# "LowDoc Loan" program was implemented where loans under $150,000 can be processed
# using a one-page application. "Yes" indicates loans with a one-page application, and
# "No" indicates loans with more information attached to the application
low_doc_missing = vw["LowDoc"].isnull()
disbursement_gross_lt_150k = low_doc_missing & (vw["DisbursementGross"] < 150000)
disbursement_gross_gte_150k = low_doc_missing & (vw["DisbursementGross"] >= 150000)

vw["LowDoc"] = np.where(disbursement_gross_lt_150k, "Y", vw["LowDoc"])
vw["LowDoc"] = np.where(disbursement_gross_gte_150k, "N", vw["LowDoc"])

# %%
vw.isnull().sum()

# %%
# Is_Franchised
df["Is_Franchised"] = df["FranchiseCode"].replace(1, 0)
df["Is_Franchised"] = np.where((df.FranchiseCode != 0), 1, df.FranchiseCode)
df["Is_Franchised"] = df["Is_Franchised"].astype(int)
df.Is_Franchised.value_counts()

# %%
# Is_CreatedJob
df["Is_CreatedJob"] = np.where((df.CreateJob > 0), 1, df.CreateJob)
df["Is_CreatedJob"] = df["Is_CreatedJob"].astype(int)
df.Is_CreatedJob.value_counts()

# %%
# Is_RetainedJob
df["Is_RetainedJob"] = np.where((df.RetainedJob > 0), 1, df.RetainedJob)
df["Is_RetainedJob"] = df["Is_RetainedJob"].astype(int)
df.Is_RetainedJob.value_counts()

# %%
# NAICS, the first 2 numbers represent the Industry
df["NAICS"].unique()

# %%
# A lot of null value, do we want to use nltk for this?
naics_counts = df["NAICS"].value_counts()

if "0" in naics_counts.index:
    print("Number of occurrences of '0' in the 'NAICS' column:", naics_counts["0"])
else:
    print("Number of occurrences of '0' in the 'NAICS' column: 0")

# %%
# Use NLTK to fill in Industry based on Company Name
all_text = " ".join(df.loc[df["NAICS"] == "0", "Name"])
words = word_tokenize(all_text)
word_counts = Counter(words)
# Print the most common words and their counts
most_common = word_counts.most_common(60)
for word, count in most_common:
    print(f"{word}: {count}")

# %%
# Keyword 1: Accommodation (72)
df[df["Name"].str.contains(" INN|MOTEL", case=False)]

# %%
df.loc[(df["Name"].str.contains(" INN|MOTEL")) & (df["NAICS"] == "0"), "NAICS"] = "72"
df.loc[df["NAICS"] == "0", "NAICS"].value_counts()

# %%
# Keyword: Food (72)
df[df["Name"].str.contains("RESTAURANT|PIZZA|CAFE", case=False)]

# %%
df.loc[
    (df["Name"].str.contains("RESTUARANT|PIZZA|CAFE")) & (df["NAICS"] == "0"), "NAICS"
] = "72"
df.loc[df["NAICS"] == "0", "NAICS"].value_counts()

# %%
df["Industry"] = df["NAICS"].astype("str").apply(lambda x: x[:2])
df["Industry"] = df["Industry"].map(
    {
        "11": "Ag/Forest/Fish/Hunt",
        "21": "Min/Quar/OilGas",
        "22": "Utilities",
        "23": "Construction",
        "31": "Manufacturing",
        "32": "Manufacturing",
        "33": "Manufacturing",
        "42": "WholesaleTrade",
        "44": "RetailTrade",
        "45": "RetailTrade",
        "48": "Trans/Warehouse",
        "49": "Trans/Warehouse",
        "51": "Information",
        "52": "Finance/Insurance",
        "53": "REst/Rental/Lease",
        "54": "Prof/Science/Tech",
        "55": "MgmtCompEnt",
        "56": "Admin/Support/WasteMgmtRem",
        "61": "Educational",
        "62": "Healthcare/SocialAssist",
        "71": "Arts/Entertain/Rec",
        "72": "Accom/Food",
        "81": "OthersNoPublicAdmin",
        "92": "PublicAdmin",
    }
)

# %%
# df = df.fillna({'Industry':'Others'})
df = df.drop(df[df["NAICS"] == "0"].index)

# %%
# Guideline: 4.1.5. Loans Backed by Real Estate
df["RealEstate"] = df["Term"].apply(lambda x: 1 if x >= 240 else 0)

# %%
# Guideline: 4.1.6. Economic Recession
df["DaysTerm"] = df["Term"] * 30
df["Active"] = df["DisbursementDate"] + pd.to_timedelta(df["DaysTerm"], unit="D")

# %%
startdate = datetime.datetime.strptime("2007-12-1", "%Y-%m-%d").date()
enddate = datetime.datetime.strptime("2009-06-30", "%Y-%m-%d").date()
df["Recession"] = df["Active"].dt.date.apply(
    lambda x: 1 if startdate <= x <= enddate else 0
)

# %%
# DaysToDisbursement
df["DaysToDisbursement"] = df["DisbursementDate"] - df["ApprovalDate"]
df["DaysToDisbursement"] = (
    df["DaysToDisbursement"]
    .astype("str")
    .apply(lambda x: x[: x.index("d") - 1])
    .astype("int64")
)

# %%
# Check if Company state is same as Bank State
df["StateSame"] = np.where(df["State"] == df["BankState"], 1, 0)

# %%
# SBA_AppvPct : guaranteed amount is based on a percentage of the gross loan amount
df["SBA_AppvPct"] = df["SBA_Appv"] / df["GrAppv"]

# %%
# AppvDisbursed: Check loan amount disbursed was equal to the full amount approved
df["AppvDisbursed"] = np.where(df["DisbursementGross"] == df["GrAppv"], 1, 0)

# %%
# Change MIS Status PIF = 0, CHGOFF = 1
df["MIS_Status"] = df["MIS_Status"].replace({"P I F": 0, "CHGOFF": 1})
df.MIS_Status.value_counts()

# %%
# Change Y / N to 1 / 0
df["LowDoc"] = df["LowDoc"].replace({"Y": 1, "N": 0}).astype(int)
df.LowDoc.value_counts()

# %%
# Change Y / N to 1 / 0
df["RevLineCr"] = df["RevLineCr"].replace({"Y": 1, "N": 0}).astype(int)
df.RevLineCr.value_counts()

# %%
# Is_Existing
# 1 = New; 2 = Exist
df["Is_Existing"] = df["NewExist"].replace({1: 0, 2: 1})
df.Is_Existing.value_counts()

# %%
df.info()

# %%
# df.drop(
#     columns=["LoanNr_ChkDgt", "Name", "Zip", "Bank", "NAICS", "NewExist", "ChgOffDate"],
#     inplace=True,
# )

# %%
# For Cross check BalanceGross got value (because too many 0)
df["BalanceGross"].unique()

# %%
# Export Cleaned Data to CSV
file_path = "data/output.parquet"
df.to_parquet(file_path, engine="pyarrow", index=False)
print(f"DataFrame has been successfully exported to {file_path}.")
