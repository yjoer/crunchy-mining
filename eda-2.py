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
df.shape

# %%
# Check for missing values in each column
missing_values = df.isnull().sum()
missing_values_percentage = (df.isnull().sum() / len(df)) * 100

missing_data = pd.DataFrame({"Missing Values": missing_values, "Percentage (%)": round(missing_values_percentage, 2)})
missing_data.sort_values(by="Percentage (%)", ascending=False)

# %%
# Tranform Data With String to Float
currency_col = [
    "DisbursementGross",
    "BalanceGross",
    "ChgOffPrinGr",
    "GrAppv",
    "SBA_Appv"
]
df[currency_col] = df[currency_col].replace("[\$,]", "", regex=True).astype(float)

# %%
continuous_cols = currency_col

continuous_cols.extend(["NoEmp","CreateJob","RetainedJob","Term"])

# %%
pd.options.display.float_format = '{:.2f}'.format

# %%
df[continuous_cols].describe().T

# %%
import matplotlib.pyplot as plt

# Count the occurrences of each category in the 'MIS_Status' column
status_counts = df['MIS_Status'].value_counts()

# Plot a pie chart
plt.figure(figsize=(8, 6))
plt.pie(status_counts, labels=status_counts.index, autopct=lambda p: '{:.0f} ({:.1f}%)'.format(p * sum(status_counts) / 100, p), startangle=140, colors=plt.cm.Set3.colors)
plt.title('Distribution of MIS_Status')
plt.show()


# %%
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def plot_histogram_with_curve_line(df, column, figsize=(6, 6), bins=20):
    fig, ax = plt.subplots(figsize=figsize)
    sns.histplot(df[column], kde=True, ax=ax, color='skyblue', edgecolor='black', linewidth=1.5, bins=bins)
    ax.set_title("Distribution of " + column)
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:,.0f}'.format(x)))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, pos: '{:,.0f}'.format(y)))
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.show()


# %%
for col in continuous_cols:
    plot_histogram_with_curve_line(df, col)

# %%
for col in continuous_cols:
    plt.figure()
    sns.barplot(x='MIS_Status', y=col, data=df)
    plt.xlabel('MIS_Status')
    plt.ylabel(col)
    plt.title(f'Average {col} vs MIS_Status')
    plt.show()

# %%
import seaborn as sns
import matplotlib.pyplot as plt

# Define the list of categorical attributes
categorical_attributes = ['RevLineCr', 'LowDoc','NewExist','UrbanRural','CreateJob','RetainedJob','NoEmp']  # Replace with your categorical attributes

# Set up subplots
num_plots = len(categorical_attributes)
num_cols = 2  # Number of columns for subplots
num_rows = (num_plots + 1) // num_cols  # Number of rows for subplots

fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, num_rows*6))

# Flatten axes if there's only one row
if num_rows == 1:
    axes = [axes]

# Loop through each categorical attribute and create countplots
for i, attribute in enumerate(categorical_attributes):
    row = i // num_cols
    col = i % num_cols
    sns.countplot(x=attribute, hue='MIS_Status', data=df, ax=axes[row][col])
    axes[row][col].set_title(f'Distribution of {attribute} with respect to MIS_Status')

# Remove empty subplots
for i in range(num_plots, num_rows * num_cols):
    row = i // num_cols
    col = i % num_cols
    fig.delaxes(axes[row][col])

plt.tight_layout()
plt.show()


# %%
df = df.fillna({"Name": "Unknown Company"})

# %%
from nltk.tokenize import word_tokenize

df["NAICS"] = df["NAICS"].astype(str).str[:2]

# %%
from collections import Counter
all_text = " ".join(df.loc[df["NAICS"] == 0, "Name"])
words = word_tokenize(all_text)
word_counts = Counter(words)
# Print the most common words and their counts
most_common = word_counts.most_common(60)
for word, count in most_common:
    print(f"{word}: {count}")

df[df["Name"].str.contains(" INN|MOTEL", case=False)]
df.loc[(df["Name"].str.contains(" INN|MOTEL")) & (df["NAICS"] == "0"), "NAICS"] = "72"
df.loc[df["NAICS"] == "0", "NAICS"].value_counts()
df[df["Name"].str.contains("RESTAURANT|PIZZA|CAFE", case=False)]
df.loc[
    (df["Name"].str.contains("RESTUARANT|PIZZA|CAFE")) & (df["NAICS"] == "0"), "NAICS"
] = "72"
df.loc[df["NAICS"] == "0", "NAICS"].value_counts()
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
df['Industry']

# %%
# Calculate count of loans for each industry
industry_counts = df['Industry'].value_counts().sort_values(ascending=False)

plt.figure(figsize=(12, 6))
sns.barplot(x=industry_counts.index, y=industry_counts.values, color="skyblue")
plt.title("Count of Loans by Industry")
plt.xlabel("Industry")
plt.ylabel("Count")
plt.xticks(rotation=90)
plt.show()

# %%
plt.figure(figsize=(12, 6))
sns.countplot(x="Industry", hue="MIS_Status", data=df)
plt.title("Industry vs MIS_Status")
plt.xlabel("Industry (NAICS)")
plt.ylabel("Count")
plt.xticks(rotation=90) 
plt.show()

# %%
industry_counts

# %%
colors = ['#0291bc', '#c1e7b3']
industry_status_counts = df.groupby(['Industry', 'MIS_Status']).size().unstack(fill_value=0)
industry_status_counts_sorted = industry_status_counts.sort_values(by='CHGOFF', ascending=False)
ax = industry_status_counts_sorted.plot(kind='barh',figsize=(12, 8), color=colors)
plt.title('Loan Status by Industry')
plt.xlabel('Count')
plt.ylabel('Industry')
plt.legend(title='MIS_Status')
plt.show()

# %%
industry_job_creation = df.groupby('Industry')['CreateJob'].sum().sort_values(ascending=False)

plt.figure(figsize=(12, 6))
sns.barplot(x=industry_job_creation.index, y=industry_job_creation.values, color="skyblue" )
plt.title("Job Creation by Industry")
plt.xlabel("Industry")
plt.ylabel("Total Job Creation")
plt.xticks(rotation=90)
plt.show()

# %%
pd.DataFrame(df.groupby('Industry')['MIS_Status'].value_counts()).unstack(level=1).style.highlight_max(color='green').highlight_min(color='blue')

# %%
f, ax = plt.subplots(figsize=(16,9))
sns.boxplot(x="Term", y="Industry", data=df, color="skyblue")
plt.title('Term', fontsize=20)
plt.xlabel('Month', fontsize=15)
plt.ylabel('Industry', fontsize=15)

# %%
f, ax = plt.subplots(figsize=(16, 9))
sns.countplot(x="ApprovalFY", data=df.sort_values(by='ApprovalFY'), hue='NewExist',palette=["#1f77b4", "#ff7f0e", "#2ca02c"])
plt.title('Total New Business based on Year', fontsize=20)
plt.xlabel('Year', fontsize=15)
plt.ylabel('Number of Business', fontsize=15)
plt.xticks(rotation='vertical')
plt.show()

# %%
import pandas as pd

categorical_columns = ['Industry', 'RevLineCr', 'LowDoc', 'NewExist', 'UrbanRural','State', 'Bank','City','State','ApprovalFY']
# Initialize lists to store results
unique_counts = []
missing_counts = []
unique_values = []

# Iterate over each categorical column
for column in categorical_columns:
    # Count the number of unique values
    unique_count = df[column].nunique()
    unique_counts.append(unique_count)
    
    # Count the number of missing values
    missing_count = df[column].isnull().sum()
    missing_counts.append(missing_count)
    
    # Get the array of unique values
    unique_value_array = df[column].unique()
    unique_values.append(unique_value_array)

# Create a DataFrame to display the results
results_df = pd.DataFrame({
    'Column': categorical_columns,
    'Unique Values': unique_values,
    'Unique Count': unique_counts,
    'Missing Count': missing_counts
})

# Display the results
print(results_df)

# %%
# Year to Int
df["ApprovalFY"] = df["ApprovalFY"].replace("1976A", 1976).astype(int)

# %%
# Convert the year column to integer type just for the plot
status_counts = df.groupby(['ApprovalFY', 'MIS_Status']).size().unstack(fill_value=0)
status_counts.index = status_counts.index.astype(int)
colors = ['#0291bc', '#c1e7b3']
# Plotting the bar graph
ax = status_counts.plot(kind='bar', stacked=True, figsize=(10, 6), color=colors)

# Adding labels and title
plt.title('Loan Status Over the Years')
plt.xlabel('Year')
plt.ylabel('Count')

# Show plot
plt.show()

# %%

# %%
import pandas as pd
import matplotlib.pyplot as plt

def plot_stacked_bar(df, category_col, target_col):
    # Group the data by category and target columns to get counts
    counts = df.groupby([category_col, target_col]).size().unstack(fill_value=0)
    
    # Plot the stacked bar chart
    counts.plot(kind='bar', stacked=True, figsize=(12, 8), colormap='tab10')
    
    # Add title and labels
    plt.title(f'Distribution of {target_col} by {category_col}')
    plt.xlabel(category_col)
    plt.ylabel('Count')
    
    # Add legend
    plt.legend(title=target_col, loc='upper right')
    
    # Show the plot
    plt.show()

plot_stacked_bar(df, 'Industry', 'MIS_Status')


# %%
def chgoff_percentage(df):
    # Group the data by Industry and calculate the proportion of CHGOFF status
    chgoff_pct = df.groupby('Industry')['MIS_Status'].apply(lambda x: (x == 'CHGOFF').mean() * 100)
    
    # Create a DataFrame to display the results
    chgoff_pct_df = pd.DataFrame(chgoff_pct).reset_index()
    chgoff_pct_df.columns = ['Industry', 'CHGOFF Percentage']
    
    return chgoff_pct_df

result = chgoff_percentage(df)
print(result)

# %%
import matplotlib.pyplot as plt

df_copy = df.copy()
industry_disbursement = df_copy.groupby('Industry')['DisbursementGross']

# Data frames based on groupby by Industry looking at aggregate and average values
df_industrySum = industry_disbursement.sum().sort_values(ascending=False)
df_industryAve = industry_disbursement.mean().sort_values(ascending=False)

# Plot 1: Gross SBA Loan Disbursement by Industry
plt.figure(figsize=(12, 6))
plt.bar(df_industrySum.index, df_industrySum / 1000000000)
plt.xticks(rotation=90)
plt.title('Gross SBA Loan Disbursement by Industry', fontsize=15)
plt.xlabel('Industry')
plt.ylabel('Gross Loan Disbursement (Billions)')
plt.tight_layout()  # Adjust layout to prevent overlapping
plt.show()

# Plot 2: Average SBA Loan Disbursement by Industry
plt.figure(figsize=(12, 6))
plt.bar(df_industryAve.index, df_industryAve / 1000000000)
plt.xticks(rotation=90)
plt.title('Average SBA Loan Disbursement by Industry', fontsize=15)
plt.xlabel('Industry')
plt.ylabel('Average Loan Disbursement (Billions)')
plt.tight_layout()  # Adjust layout to prevent overlapping
plt.show()


# %%
import matplotlib.pyplot as plt

# Plotting scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(df['DisbursementGross'], df['BalanceGross'], alpha=0.5)
plt.title('Scatter Plot of DisbursementGross vs BalanceGross')
plt.xlabel('DisbursementGross')
plt.ylabel('BalanceGross')
plt.grid(True)
plt.show()


# %%
correlation_matrix = df[continuous_cols].corr()

# Plot the correlation matrix as a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='BuPu', fmt=".2f")
plt.title('Correlation Plot')
plt.show()

# %%
print("Correlation Matrix:")
print(correlation_matrix.to_string())

# %%
# State

plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='State', palette='Set2')
plt.xticks(rotation=90)
plt.xlabel('State')
plt.ylabel('Count')
plt.title('Count of each state')
plt.show()

# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df['SBA_Appv_Bin'] = pd.cut(df['SBA_Appv'], bins=np.arange(0, df['SBA_Appv'].max() + 400000, 400000))

binned_data = df.groupby(['SBA_Appv_Bin', 'MIS_Status']).size().unstack(fill_value=0)

ax = binned_data.plot(kind='bar', stacked=True, figsize=(14, 7), colormap='tab10')

ax.set_xlabel('Approved Loan Amount ($)')
ax.set_ylabel('Count of Loans')
ax.set_title('Approved Loan Amount by Status')
ax.set_yscale('log')

tick_labels = binned_data.index.categories.mid
tick_labels_str = [f'{int(label):,}' for label in tick_labels]
ax.set_xticks(np.arange(len(tick_labels)))
ax.set_xticklabels(tick_labels_str, rotation=45, ha='right')

plt.tight_layout()
plt.show()


# %%
binned_data

# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df['GrAppv_bin'] = pd.cut(df['GrAppv'], bins=np.arange(0, df['GrAppv'].max() + 400000, 400000))

binned_data = df.groupby(['GrAppv_bin', 'MIS_Status']).size().unstack(fill_value=0)

ax = binned_data.plot(kind='bar', stacked=True, figsize=(14, 7), colormap='tab10')

ax.set_xlabel('SBA Guarantees ($)')
ax.set_ylabel('Count of Loans')
ax.set_title('SBA Guarantees by Status')
ax.set_yscale('log')

tick_labels = binned_data.index.categories.mid
tick_labels_str = [f'{int(label):,}' for label in tick_labels]
ax.set_xticks(np.arange(len(tick_labels)))
ax.set_xticklabels(tick_labels_str, rotation=45, ha='right')

plt.tight_layout()
plt.show()

# %%
binned_data
