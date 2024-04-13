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
temp_empty_state = df.loc[df['State'].isnull()]
temp_empty_state

# %%
df.loc[df['State'].isnull(), 'Zip'].unique()

# %%
#Sort Zip in Ascending
df_sorted = df.sort_values(by='Zip')

#Group df based on zip code
grouped = df_sorted.groupby('Zip')

# %%
#Fill the null 'State' based on the zip code group
df_sorted['State'] = grouped['State'].fillna(method='ffill')
df = df_sorted.sort_index()


# %%
#To cross check the imputation Result for State
def print_imputed_State_rows(df, temp_empty_state):
    if not temp_empty_state.empty:
        null_indices = temp_empty_state.index
        imputed_rows = df.loc[null_indices, ["City", "State", "Zip", "BankState"]]
        print("Table with imputed data for rows with null values in the 'State' column:")
        print(imputed_rows)
    else:
        print("No rows with null values in the 'State' column.")

print_imputed_State_rows(df, temp_empty_state)

# %%
#Still got one State is NA, We will fill in manually based on Zip Code
df.loc[df['State'].isnull()]
df = df.fillna({'State':'AP'})


# %%
#Notice that Row 49244 Zip code is 0, the state is incorrect, we impute manually
def change_state_value(df, row_id, new_value):
    df_copy = df.copy()
    df_copy.loc[row_id, 'State'] = new_value
    return df_copy

row_id = 49244
new_value = 'NY' 
df = change_state_value(df, row_id, new_value)

# %%
print_imputed_State_rows(df, temp_empty_state)

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
#Drop NA for IsExiting
#We dont make assumption since this column might be one of the important factors
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
#Drop DisbursementDate with NA
df = df.dropna(subset=['DisbursementDate'], how='all')

# %%
df['MIS_Status'].unique()

# %%
#For MIS_Status, if got change off date, we will fill in change off, others we cannot impute, we will drop it
df['MIS_Status'] = np.where((df['MIS_Status'] == "CHGOFF") & (df['ChgOffDate'] != np.nan),"CHGOFF",df.MIS_Status)

# %%
df = df[(df['MIS_Status'] == "P I F") | (df['MIS_Status'] == "CHGOFF")]
print(df[['MIS_Status', 'ChgOffDate']].head(10))

# %%
df.isnull().sum()

# %% [markdown]
# Data Transformation

# %%
#Tranform Data With String to Float
currency_col = ['DisbursementGross', 'BalanceGross', 'ChgOffPrinGr', 'GrAppv', 'SBA_Appv']
df[currency_col] = df[currency_col].replace('[\$,]', '', regex=True).astype(float)

# %%
#LowDoc valid input only Y or N
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
#A lot of null value, do we want to use nltk for this?
naics_counts = df['NAICS'].value_counts()

if '0' in naics_counts.index:
    print("Number of occurrences of '0' in the 'NAICS' column:", naics_counts['0'])
else:
    print("Number of occurrences of '0' in the 'NAICS' column: 0")

# %%
#Use NLTK to fill in Industry based on Company Name
all_text = ' '.join(df.loc[df['NAICS'] == "0", 'Name'])
words = word_tokenize(all_text)
word_counts = Counter(words)
# Print the most common words and their counts
most_common = word_counts.most_common(60)
for word, count in most_common:
    print(f'{word}: {count}')

# %%
#Keyword 1: Accommodation (72)
df[df['Name'].str.contains(' INN|MOTEL', case=False)]

# %%
df.loc[(df['Name'].str.contains(' INN|MOTEL')) & (df['NAICS'] == "0"), 'NAICS'] = 72
df.loc[df['NAICS'] == "0", 'NAICS'].value_counts()

# %%
#Keyword: Food (72)
df[df['Name'].str.contains('RESTAURANT|PIZZA|CAFE', case=False)]

# %%
df.loc[(df['Name'].str.contains('RESTUARANT|PIZZA|CAFE')) & (df['NAICS'] == '0'), 'NAICS'] = 72
df.loc[df['NAICS'] == '0', 'NAICS'].value_counts()

# %%
df['Industry'] = df['NAICS'].astype('str').apply(lambda x: x[:2])
df['Industry'] = df['Industry'].map({
    '11': 'Ag/Forest/Fish/Hunt',
    '21': 'Min/Quar/OilGas',
    '22': 'Utilities',
    '23': 'Construction',
    '31': 'Manufacturing',
    '32': 'Manufacturing',
    '33': 'Manufacturing',
    '42': 'WholesaleTrade',
    '44': 'RetailTrade',
    '45': 'RetailTrade',
    '48': 'Trans/Warehouse',
    '49': 'Trans/Warehouse',
    '51': 'Information',
    '52': 'Finance/Insurance',
    '53': 'REst/Rental/Lease',
    '54': 'Prof/Science/Tech',
    '55': 'MgmtCompEnt',
    '56': 'Admin/Support/WasteMgmtRem',
    '61': 'Educational',
    '62': 'Healthcare/SocialAssist',
    '71': 'Arts/Entertain/Rec',
    '72': 'Accom/Food',
    '81': 'OthersNoPublicAdmin',
    '92': 'PublicAdmin'
})

# %%
df = df.fillna({'Industry':'Others'})
#df  = df.drop(df[df['NAICS'] == 0].index)

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
#3.3 Time Period section
#Excluding those disbursed after 2010 since the loan term is typically 5 years or more
df = df[df['DisbursementFY'] <= 2010]

# %%
df.info()

# %%
#Export Cleaned Data to CSV
file_path = "data/output.csv"
df.to_csv(file_path, index=False)
print(f"DataFrame has been successfully exported to {file_path}.")

# %%
