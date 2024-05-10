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
from collections import Counter

import nltk
import numpy as np
import pandas as pd
import pandera as pa
from nltk.tokenize import word_tokenize
from ydata_profiling import ProfileReport

pd.set_option("future.no_silent_downcasting", True)
pd.set_option("mode.copy_on_write", True)
nltk.download("punkt")

# %%
df = pd.read_csv("data/SBA.csv", low_memory=False)
df.info()

# %%
vw = df[:]

# %% [markdown]
# ## Explore

# %%
if False:
    profile = ProfileReport(df, title="Profiling Report")
    profile.to_file("report_1.html")

# %% [markdown]
# ## Modify

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
vw.drop(vw[vw["ApprovalDate"] > vw["DisbursementDate"]].index, inplace=True)

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
vw["MIS_Status"].value_counts(dropna=False)

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
temp_empty_state.iloc[:, :5]

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

# %% [markdown]
# ### Data Transformation

# %%
# NAICS, the first 2 numbers represent the Industry
vw["NAICS"] = vw["NAICS"].astype(str).str[:2]

# %%
# Change Y / N to 1 / 0
vw["RevLineCr"] = vw["RevLineCr"].replace({"Y": 1, "N": 0}).astype(int)
vw.RevLineCr.value_counts()

# %%
# Change Y / N to 1 / 0
vw["LowDoc"] = vw["LowDoc"].replace({"Y": 1, "N": 0}).astype(int)
vw.LowDoc.value_counts()

# %%
# Change MIS Status PIF = 0, CHGOFF = 1
vw["MIS_Status"] = vw["MIS_Status"].replace({"P I F": 0, "CHGOFF": 1}).astype(int)
vw.MIS_Status.value_counts()

# %% [markdown]
# ### Feature Engineering

# %%
vw["DisbursementFY"] = vw["DisbursementDate"].map(lambda x: x.year)

# %%
# IsFranchised
vw["IsFranchised"] = vw["FranchiseCode"].replace(1, 0)
vw["IsFranchised"] = np.where((vw.FranchiseCode != 0), 1, vw.FranchiseCode)
vw["IsFranchised"] = vw["IsFranchised"].astype(int)
vw.IsFranchised.value_counts()

# %%
# Is_CreatedJob
vw["IsCreatedJob"] = np.where((vw.CreateJob > 0), 1, vw.CreateJob)
vw["IsCreatedJob"] = vw["IsCreatedJob"].astype(int)
vw.IsCreatedJob.value_counts()

# %%
# Is_RetainedJob
vw["IsRetainedJob"] = np.where((vw.RetainedJob > 0), 1, vw.RetainedJob)
vw["IsRetainedJob"] = vw["IsRetainedJob"].astype(int)
vw.IsRetainedJob.value_counts()

# %%
# NAICS, the first 2 numbers represent the Industry
vw["NAICS"].unique()

# %%
# A lot of null value, do we want to use nltk for this?
naics_counts = vw["NAICS"].value_counts()

if "0" in naics_counts.index:
    print("Number of occurrences of '0' in the 'NAICS' column:", naics_counts["0"])
else:
    print("Number of occurrences of '0' in the 'NAICS' column: 0")

# %%
# Use NLTK to fill in Industry based on Company Name
all_text = " ".join(vw.loc[vw["NAICS"] == "0", "Name"])
words = word_tokenize(all_text)
word_counts = Counter(words)
# Print the most common words and their counts
most_common = word_counts.most_common(60)
for word, count in most_common:
    print(f"{word}: {count}")

# %%
# Keyword 1: Accommodation (72)
keyword_accommodation = vw["Name"].str.contains(" INN|MOTEL", case=False)
vw[keyword_accommodation].iloc[:, :5]

# %%
vw.loc[keyword_accommodation & (vw["NAICS"] == "0"), "NAICS"] = "72"
vw.loc[vw["NAICS"] == "0", "NAICS"].value_counts()

# %%
# Keyword: Food (72)
keyword_food = vw["Name"].str.contains("RESTAURANT|PIZZA|CAFE", case=False)
vw[keyword_food].iloc[:, :5]

# %%
vw.loc[keyword_food & (vw["NAICS"] == "0"), "NAICS"] = "72"
vw.loc[vw["NAICS"] == "0", "NAICS"].value_counts()

# %%
vw["Industry"] = vw["NAICS"].map(
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
vw.drop(vw[vw["NAICS"] == "0"].index, inplace=True)

# %%
# Guideline: 4.1.5. Loans Backed by Real Estate
vw["RealEstate"] = vw["Term"].apply(lambda x: 1 if x >= 240 else 0)

# %%
# Guideline: 4.1.6. Economic Recession
vw["DaysTerm"] = vw["Term"] * 30
vw["Active"] = vw["DisbursementDate"] + pd.to_timedelta(vw["DaysTerm"], unit="D")

# %%
startdate = datetime.datetime.strptime("2007-12-1", "%Y-%m-%d").date()
enddate = datetime.datetime.strptime("2009-06-30", "%Y-%m-%d").date()
vw["Recession"] = vw["Active"].dt.date.apply(
    lambda x: 1 if startdate <= x <= enddate else 0
)

# %%
# DaysToDisbursement
vw["DaysToDisbursement"] = vw["DisbursementDate"] - vw["ApprovalDate"]
vw["DaysToDisbursement"] = (
    vw["DaysToDisbursement"]
    .astype("str")
    .apply(lambda x: x[: x.index("d") - 1])
    .astype("int64")
)

# %%
# Check if Company state is same as Bank State
vw["StateSame"] = np.where(vw["State"] == vw["BankState"], 1, 0)

# %%
# SBA_AppvPct : guaranteed amount is based on a percentage of the gross loan amount
vw["SBA_AppvPct"] = vw["SBA_Appv"] / vw["GrAppv"]

# %%
# AppvDisbursed: Check loan amount disbursed was equal to the full amount approved
vw["AppvDisbursed"] = np.where(vw["DisbursementGross"] == vw["GrAppv"], 1, 0)

# %%
# Is_Existing
# 1 = New; 2 = Exist
vw["IsExisting"] = vw["NewExist"].replace({1: 0, 2: 1})
vw.IsExisting.value_counts()

# %%
vw.info()

# %%
# df.drop(
#     columns=["LoanNr_ChkDgt", "Name", "Zip", "Bank", "NAICS", "NewExist", "ChgOffDate"],
#     inplace=True,
# )

# %%
# For Cross check BalanceGross got value (because too many 0)
vw["BalanceGross"].value_counts()

# %% [markdown]
# ### Data Validation

# %%
schema = pa.DataFrameSchema(
    {
        "LoanNr_ChkDgt": pa.Column(pa.dtypes.Int64, unique=True),
        "Name": pa.Column(pa.dtypes.String),
        "City": pa.Column(pa.dtypes.String),
        "State": pa.Column(pa.dtypes.String),
        "Zip": pa.Column(pa.dtypes.Int64, checks=pa.Check.in_range(0, 99999)),
        "Bank": pa.Column(pa.dtypes.String),
        "BankState": pa.Column(pa.dtypes.String),
        "NAICS": pa.Column(pa.dtypes.String, checks=pa.Check.str_length(max_value=2)),
        "ApprovalDate": pa.Column(pa.dtypes.Timestamp),
        "ApprovalFY": pa.Column(pa.dtypes.Int32, checks=pa.Check.in_range(1970, 2014)),
        "Term": pa.Column(pa.dtypes.Int64, checks=pa.Check.ge(0)),
        "NoEmp": pa.Column(pa.dtypes.Int64, checks=pa.Check.ge(0)),
        "NewExist": pa.Column(pa.dtypes.Int32, checks=pa.Check.isin([1, 2])),
        "CreateJob": pa.Column(pa.dtypes.Int64, checks=pa.Check.ge(0)),
        "RetainedJob": pa.Column(pa.dtypes.Int64, checks=pa.Check.ge(0)),
        "FranchiseCode": pa.Column(pa.dtypes.Int64),
        "UrbanRural": pa.Column(pa.dtypes.Int64, checks=pa.Check.isin([0, 1, 2])),
        "RevLineCr": pa.Column(pa.dtypes.Int32, checks=pa.Check.isin([0, 1])),
        "LowDoc": pa.Column(pa.dtypes.Int32, checks=pa.Check.isin([0, 1])),
        "ChgOffDate": pa.Column(pa.dtypes.Timestamp, nullable=True),
        "DisbursementDate": pa.Column(pa.dtypes.Timestamp),
        "DisbursementGross": pa.Column(pa.dtypes.Float64, checks=pa.Check.ge(0)),
        "BalanceGross": pa.Column(pa.dtypes.Float64, checks=pa.Check.ge(0)),
        "MIS_Status": pa.Column(pa.dtypes.Int32, checks=pa.Check.isin([0, 1])),
        "ChgOffPrinGr": pa.Column(pa.dtypes.Float64, checks=pa.Check.ge(0)),
        "GrAppv": pa.Column(pa.dtypes.Float64, checks=pa.Check.ge(0)),
        "SBA_Appv": pa.Column(pa.dtypes.Float64, checks=pa.Check.ge(0)),
        "DisbursementFY": pa.Column(
            pa.dtypes.Int64,
            checks=pa.Check.in_range(1970, 2028),
        ),
        "IsFranchised": pa.Column(pa.dtypes.Int32, checks=pa.Check.isin([0, 1])),
        "IsCreatedJob": pa.Column(pa.dtypes.Int32, checks=pa.Check.isin([0, 1])),
        "IsRetainedJob": pa.Column(pa.dtypes.Int32, checks=pa.Check.isin([0, 1])),
        "Industry": pa.Column(pa.dtypes.String),
        "RealEstate": pa.Column(pa.dtypes.Int64, checks=pa.Check.isin([0, 1])),
        "DaysTerm": pa.Column(pa.dtypes.Int64, checks=pa.Check.ge(0)),
        "Active": pa.Column(pa.dtypes.Timestamp),
        "Recession": pa.Column(pa.dtypes.Int64, checks=pa.Check.isin([0, 1])),
        "DaysToDisbursement": pa.Column(pa.dtypes.Int64, checks=pa.Check.ge(0)),
        "StateSame": pa.Column(pa.dtypes.Int32, checks=pa.Check.isin([0, 1])),
        "SBA_AppvPct": pa.Column(pa.dtypes.Float64, checks=pa.Check.in_range(0, 100)),
        "AppvDisbursed": pa.Column(pa.dtypes.Int32, checks=pa.Check.isin([0, 1])),
        "IsExisting": pa.Column(pa.dtypes.Int32, checks=pa.Check.isin([0, 1])),
    },
    strict=True,
)

# %%
try:
    schema.validate(vw, lazy=True)
except pa.errors.SchemaError as exc:
    print(exc)

# %%
if False:
    profile = ProfileReport(vw, title="Profiling Report")
    profile.to_file("report_2.html")

# %%
# Export Cleaned Data to CSV
file_path = "data/output.parquet"
vw.to_parquet(file_path, engine="pyarrow", index=False)
print(f"DataFrame has been successfully exported to {file_path}.")

# %%
