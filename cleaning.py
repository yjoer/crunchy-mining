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
import warnings
warnings.filterwarnings("ignore")

# %%
df = pd.read_csv("data/SBA.csv")

# %%
df.info()

# %%
df.head()

# %%
df.shape

# %%
#Check duplicate row
duplicate_rows = df[df.duplicated()]

if not duplicate_rows.empty:
    print("Duplicate rows found:")
    print(duplicate_rows)
else:
    print("No duplicate rows found.")

# %%
#Check Null
df.isnull().sum()

# %%
df.describe()

# %%
df.describe(include='O')

# %%
#Check for Name Column with Null
df.loc[df['Name'].isnull()]

# %%
df = df.fillna({'Name':'Unknown Company'})

# %%
df = df.fillna({'City':'Unknown City'})

# %%
df.loc[df['State'].isnull(), 'Zip'].unique()

# %%
#Sort Zip in Ascending
df_sorted = df.sort_values(by='Zip')

#Group df based on zip code
grouped = df_sorted.groupby('Zip')

#Fill the null 'State'based on the zip code group
df_sorted['State'] = grouped['State'].fillna(method='ffill')
df = df_sorted.sort_index()

# %%
df.isnull().sum()

# %%
#Still got one State is NA, We will fill in manually based on Zip Code
df.loc[df['State'].isnull()]

# %%
df = df.fillna({'State':'AP'})

# %%
#Fill in NA Bank
df = df.fillna({'Bank':'Unknown Bank'})

# %%
df.loc[df['BankState'].isnull()]

# %%
#Fill in Bank State based on Bank
df_sorted = df.sort_values(by='Bank')
grouped = df_sorted.groupby('Bank')
df_sorted['BankState'] = grouped['BankState'].fillna(method='ffill')
df = df_sorted.sort_index()

# %%
df.loc[df['BankState'].isnull()]

# %%
#Drop the BankState NA Row since we cant do any imputation
df = df.dropna(subset=['BankState'], how='all')
df.shape

# %%
#IsExiting
#We dont make assumption based on the FranchiseCode for this col
df = df.dropna(subset=['NewExist'], how='all')
df.shape

# %%
#For RevLineCr, Based on the data description, only Y and N, thus we will ingore others
df['RevLineCr'].unique()

# %%
df = df[df['RevLineCr'].isin(['Y', 'N'])]
df['RevLineCr'].unique()

# %%
df.shape

# %%
df.isnull().sum()

# %%
df.shape

# %%
df.isnull().sum()

# %%
#Drop DisbursementDate with NA
df = df.dropna(subset=['DisbursementDate'], how='all')

# %%
df['MIS_Status'].unique()

# %%
#For MIS_Status, if got change off date, we will fill in change off, others we cannot impute, we will drop it
df['MIS_Status'] = np.where((df['MIS_Status'] == "CHGOFF") & (df['ChgOffDate'] != np.nan),"CHGOFF",df.MIS_Status)

df = df[(df['MIS_Status'] == "P I F") | (df['MIS_Status'] == "CHGOFF")]

# %%
print(df[['MIS_Status', 'ChgOffDate']].head(10))

# %%
df.isnull().sum()

# %% [markdown]
# Data Transformation

# %%
#Tranform Data With String to Int/Float
currency_col = ['DisbursementGross', 'BalanceGross', 'ChgOffPrinGr', 'GrAppv', 'SBA_Appv']
df[currency_col] = df[currency_col].replace('[\$,]', '', regex=True).astype(float)

# %%
#LowDoc also only Y or N
df['LowDoc'].unique()

# %%
#Guideline: LowDoc(Y:Yes, N:No): In order to process more loansefficiently, a“LowDoc Loan”program was implemented whereloans under $150,000 can be processed using a one-page appli-cation.“Yes”indicates loans with a one-page application, and“No”indicates loans with more information attached to the application
df['LowDoc'] = np.where((df['LowDoc'] == np.nan) & (df['DisbursementGross'] < 150000),'Y',df.LowDoc)
df['LowDoc'] = np.where((df['LowDoc'] == np.nan) & (df['DisbursementGross'] >= 150000),'N',df.LowDoc)

df = df[(df['LowDoc'] == 'Y') | (df['LowDoc'] == 'N')]

# %%
df.isnull().sum()

# %%
#To Date
date_col = ['ApprovalDate', 'ChgOffDate','DisbursementDate']
df[date_col] = pd.to_datetime(df[date_col].stack(),format='%d-%b-%y').unstack()

# %%
# Identify the values leading to mixed type
non_numeric_mask = pd.to_numeric(df["ApprovalFY"], errors="coerce").isna()
df.loc[non_numeric_mask, "ApprovalFY"].unique()

# %%
#Year to Int
df['ApprovalFY'].replace('1976A', 1976, inplace=True)
df['ApprovalFY']= df['ApprovalFY'].astype(int)

# %%
# Change Franchise = Is_Franchise
df['FranchiseCode'] = df['FranchiseCode'].replace(1,0 )	
df['FranchiseCode'] = np.where((df.FranchiseCode != 0 ),1,df.FranchiseCode)
df.rename(columns={"FranchiseCode": "Is_Franchised"}, inplace=True)
df.Is_Franchised.value_counts()

# %%
#NAICS, the first 2 numbers represent the Industry
df['NAICS'] = df['NAICS'].astype(str).str[:2]
df['NAICS']

# %%
df['NAICS'].unique()

# %%
df['Industry'] = df['NAICS'].astype('str').apply(lambda x: x[:2])
df['Industry'] = df['Industry'].map({
    '11': 'Ag/For/Fish/Hunt',
    '21': 'Min/Quar/Oil_Gas_ext',
    '22': 'Utilities',
    '23': 'Construction',
    '31': 'Manufacturing',
    '32': 'Manufacturing',
    '33': 'Manufacturing',
    '42': 'Wholesale_trade',
    '44': 'Retail_trade',
    '45': 'Retail_trade',
    '48': 'Trans/Ware',
    '49': 'Trans/Ware',
    '51': 'Information',
    '52': 'Finance/Insurance',
    '53': 'RE/Rental/Lease',
    '54': 'Prof/Science/Tech',
    '55': 'Mgmt_comp',
    '56': 'Admin_sup/Waste_Mgmt_Rem',
    '61': 'Educational',
    '62': 'Healthcare/Social_assist',
    '71': 'Arts/Entertain/Rec',
    '72': 'Accom/Food_serv',
    '81': 'Other_no_pub',
    '92': 'Public_Admin'
})

# %%
#A lot of null value, do we want to use nltk for this?
naics_counts = df['NAICS'].value_counts()

if '0' in naics_counts.index:
    print("Number of occurrences of '0' in the 'NAICS' column:", naics_counts['0'])
else:
    print("Number of occurrences of '0' in the 'NAICS' column: 0")

# %%
df = df.fillna({'Industry':'Others'})

# %%
#Guideline: 4.1.5. Loans Backed by Real Estate
df['RealEstate'] = df['Term'].apply(lambda x: 1 if x >= 240 else 0)

# %%
#Guideline: 4.1.6. Economic Recession
df['DaysTerm'] =  df['Term']*30
df['Active'] = df['DisbursementDate'] + pd.TimedeltaIndex(df['DaysTerm'], unit='D')

# %%
startdate = datetime.datetime.strptime('2007-12-1', "%Y-%m-%d").date()
enddate = datetime.datetime.strptime('2009-06-30', "%Y-%m-%d").date()
df['Recession'] = df['Active'].dt.date.apply(lambda x: 1 if startdate <= x <= enddate else 0)

# %%
#DaysToDisbursement
df['DaysToDisbursement'] = df['DisbursementDate'] - df['ApprovalDate']
df['DaysToDisbursement'] = df['DaysToDisbursement'].astype('str').apply(lambda x: x[:x.index('d') - 1]).astype('int64')

# %%
df['DisbursementFY'] = df['DisbursementDate'].map(lambda x: x.year)

# %%
#Check if Company state is same as Bank State
df['StateSame'] = np.where(df['State'] == df['BankState'], 1, 0)

# %%
#SBA_AppvPct : guaranteed amount is based on a percentage of the gross loan amount
df['SBA_AppvPct'] = df['SBA_Appv'] / df['GrAppv']

# %%
#AppvDisbursed: Check loan amount disbursed was equal to the full amount approved
df['AppvDisbursed'] = np.where(df['DisbursementGross'] == df['GrAppv'], 1, 0)

# %%
df.info()

# %% [markdown]
# Questions: 
# 1. Data Transformation need to transform to boolean?
# 2. Are we using NAICS while training? Was thinking should we use nltk to map the data.
# 3. Double check Zip Code

# %% [markdown]
# EDA

# %%
df = df.astype({'UrbanRural': 'object', 
                    'RevLineCr': 'object', 
                    'LowDoc':'object', 
                    'MIS_Status':'object'})

# %%
f, ax = plt.subplots(figsize=(16,9))
sns.barplot(x="ApprovalFY", y="DisbursementGross", color='Salmon', data=df)
plt.title('Total Loan VS Year', fontsize=20)
plt.xlabel('Year', fontsize=15)
plt.ylabel('Total loan ($)', fontsize=15)
plt.xticks(rotation='vertical');

# %%
df.DisbursementGross.describe()

# %%
f, ax = plt.subplots(figsize=(16,9))
sns.barplot(x="DisbursementGross", y="Industry", data=df)
plt.title('Total Loan Based on Industry', fontsize=20)
plt.xlabel('Total Loan ($)', fontsize=15)
plt.ylabel('Industry', fontsize=15)

# %%
df.groupby('Industry')['DisbursementGross'].describe().style.highlight_max(color='green').highlight_min(color='blue')

# %%
f, ax = plt.subplots(figsize=(16,9))
sns.countplot(x="ApprovalFY", data=df,hue='MIS_Status')
plt.title('Total PIF vs Total CHGOFF based on Year', fontsize=20)
plt.xlabel('Year', fontsize=15)
plt.ylabel('Total Loan($)', fontsize=15)
plt.legend(["PIF", "CHGOFF"],loc='upper right')
plt.xticks(rotation='vertical');

# %%
f, ax = plt.subplots(figsize=(16,9))
sns.countplot(y="Industry", hue="MIS_Status", data=df)
plt.title('Total PIF vs Total CHGOFF based on Industry', fontsize=20)
plt.xlabel('Total CHGOFF', fontsize=15)
plt.ylabel('Industry', fontsize=15)
plt.legend(["Tidak", "Gagal"],loc='lower right')

# %%
pd.DataFrame(df.groupby('Industry')['MIS_Status'].value_counts()).unstack(level=1).style.highlight_max(color='green').highlight_min(color='blue')


# %%
df.to_parquet("data/clean.parquet", engine="pyarrow")

# %%
