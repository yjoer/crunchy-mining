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

# %%
#Tranform Data With String to Int/Float
currency_col = ['DisbursementGross', 'BalanceGross', 'ChgOffPrinGr', 'GrAppv', 'SBA_Appv']
df[currency_col] = df[currency_col].replace('[\$,]', '', regex=True).astype(float)

# %%
#LowDoc also only Y or N
df['LowDoc'].unique()

# %%
#LowDoc(Y:Yes, N:No): In order to process more loansefficiently, a“LowDoc Loan”program was implemented whereloans under $150,000 can be processed using a one-page appli-cation.“Yes”indicates loans with a one-page application, and“No”indicates loans with more information attached to theapplication
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
#Year to Int
df['ApprovalFY'].replace('1976A', 1976, inplace=True)
df['ApprovalFY']= df['ApprovalFY'].astype(int)

# %%
df.info()

# %%
