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
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

from sklearn import metrics
from collections import Counter


# %%
def Reg_Models_Evaluation_Metrics (model,X_train,y_train,X_test,y_test,y_pred):
    cv_score = cross_val_score(estimator = model, X = X_train, y = y_train, cv = 10)
    
    r2 = model.score(X_test, y_test)
    n = X_test.shape[0]
    p = X_test.shape[1]
    adjusted_r2 = 1-(1-r2)*(n-1)/(n-p-1)
    RMSE = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
    R2 = model.score(X_test, y_test)
    CV_R2 = cv_score.mean()

    return R2, adjusted_r2, CV_R2, RMSE
    
    print('RMSE:', round(RMSE,4))
    print('R2:', round(R2,4))
    print('Adjusted R2:', round(adjusted_r2, 4) )
    print("Cross Validated R2: ", round(cv_score.mean(),4) )



# %%
df = pd.read_csv("data/output.csv")

# %%
df.info()

# %%
variables = {
    "categorical": [
        "City",
        "State",
        "BankState",
        "ApprovalDate",
        "DisbursementDate",
        "Industry",
        "Active"
    ],
    "numerical": [
        "ApprovalFY",
        "Term",
        "NoEmp",
        "CreateJob",
        "RetainedJob",
        "FranchiseCode",
        "UrbanRural",
        "RevLineCr",
        "LowDoc",
        "DisbursementGross",
        "BalanceGross",
        "ChgOffPrinGr",
        "SBA_Appv",
        "DisbursementFY",
        "Is_Franchised",
        "Is_CreatedJob",
        "Is_RetainedJob",
        "RealEstate",
        "DaysTerm",
        "Recession",
        "DaysToDisbursement",
        "StateSame",
        "SBA_AppvPct",
        "AppvDisbursed",
        "Is_Existing",
        "MIS_Status"
    ],
    "target": "GrAppv",
}

# %%
df.drop(columns=['ChgOffDate'], inplace=True)

# %%
#Encode Data
encoder = OrdinalEncoder()

df[variables["categorical"]] = encoder.fit_transform(df[variables["categorical"]])

# Print the updated dataset
df.head()

# %%
X = df.drop('GrAppv', axis=1)
y = df['GrAppv']

# %%
#Split data set
from sklearn.model_selection import train_test_split

# Split the data into train and test sets (85% train, 15% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

# Split the train set further into train and validation sets (70% train, 15% validation)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15 / 0.85, random_state=42)

print("Train set shape:", X_train.shape)
print("Validation set shape:", X_val.shape)
print("Test set shape:", X_test.shape)

validate_rmse_score = []
test_rmse_score = []

# %%
#Data Scaling
# create a MinMaxScaler object
scaler = MinMaxScaler()
#TODO: Log Transform

#Normalize the dataset
x_train_scaler = scaler.fit_transform(X_train)
x_test_scaler =  scaler.transform(X_test)
x_val_scaler =  scaler.transform(X_val)

y_scaler = MinMaxScaler()
y_train_scaler = y_scaler.fit_transform(y_train.to_numpy().reshape(-1,1))
y_test_scaler =  y_scaler.transform(y_test.to_numpy().reshape(-1,1))
y_val_scaler =  y_scaler.transform(y_val.to_numpy().reshape(-1,1))

# %%
from sklearn.linear_model import LinearRegression

lm = LinearRegression()
lm.fit(x_train_scaler, y_train_scaler)

# Model making a prediction on test data
y_pred = lm.predict(x_test_scaler)

# %%
print(y_pred)

# %%
result = y_scaler.inverse_transform(y_pred)

# %%
ndf = [Reg_Models_Evaluation_Metrics(lm,x_train_scaler,y_train_scaler,x_test_scaler,y_test,result)]

lm_score = pd.DataFrame(data = ndf, columns=['R2 Score','Adjusted R2 Score','Cross Validated R2 Score','RMSE'])
lm_score.insert(0, 'Model', 'Linear Regression')
lm_score

# %%
plt.figure(figsize = (10,5))
sns.regplot(x=y_test,y=result)
plt.title('Linear regression', fontsize = 20)

# %% [markdown]
# Others Testing Code, can ignore it

# %%
validate_rmse_score = []
test_rmse_score = []

# %%
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import root_mean_squared_error

dt_reg = DecisionTreeRegressor()
dt_reg.fit(x_train_scaler, y_train_scaler)
dt_y_pred = dt_reg.predict(x_test_scaler)

# %%
dt_result = y_scaler.inverse_transform(dt_y_pred.reshape(-1,1))

# %%
test_rmse = root_mean_squared_error(y_test, dt_result)
print("Test RMSE:", test_rmse)
test_rmse_score.append(test_rmse)

# %%
# Make predictions on the validation set
# y_val_pred = dt_reg.predict(x_val_scaler)
# val_rmse = root_mean_squared_error(y_val_scaler, y_val_pred)
# print("Validation RMSE:", val_rmse)
# validate_rmse_score.append(val_rmse)
# Make predictions on the test set
