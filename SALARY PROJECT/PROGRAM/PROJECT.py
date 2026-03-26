# STEP 1 — IMPORT LIBRARIES

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb


# STEP 2 — LOAD DATASET

df = pd.read_csv("expected_ctc.csv")

# clean column names
df.columns = df.columns.str.strip().str.lower()

print(df.head())
print(df.columns)


# STEP 3 — DROP USELESS COLUMNS

df = df.drop(["idx", "applicant_id"], axis=1, errors="ignore")


# STEP 4 — HANDLE MISSING VALUES

# numeric columns
for col in df.select_dtypes(include=["int64","float64"]):
    df[col] = df[col].fillna(df[col].median())

# categorical columns
for col in df.select_dtypes(include=["object","str"]):
    df[col] = df[col].fillna(df[col].mode()[0])


# check remaining missing values
print("Missing values after cleaning:")
print(df.isnull().sum())


# STEP 5 — ENCODE CATEGORICAL DATA

le = LabelEncoder()

for col in df.select_dtypes(include=["object","str"]):
    df[col] = le.fit_transform(df[col])


# STEP 6 — DEFINE FEATURES & TARGET

X = df.drop("expected_ctc", axis=1)
y = df["expected_ctc"]


# STEP 7 — TRAIN TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# STEP 8 — TRAIN MODELS

# Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)

# Random Forest
rf = RandomForestRegressor()
rf.fit(X_train, y_train)

# XGBoost
xg = xgb.XGBRegressor()
xg.fit(X_train, y_train)


# STEP 9 — MODEL EVALUATION

def evaluate(model):

    pred = model.predict(X_test)

    print("MAE :", mean_absolute_error(y_test, pred))
    print("RMSE:", np.sqrt(mean_squared_error(y_test, pred)))
    print("R2 :", r2_score(y_test, pred))
    print("------------------------------")


print("Linear Regression")
evaluate(lr)

print("Random Forest")
evaluate(rf)

print("XGBoost")
evaluate(xg)


# STEP 10 — FEATURE IMPORTANCE

xgb.plot_importance(xg)
plt.show()


# STEP 11 — SAVE MODEL (OPTIONAL)

import joblib
joblib.dump(xg, "salary_model.pkl")

print("Model saved successfully")

# SALARY DISTRIBUTION
plt.figure(figsize=(8,5))

sns.histplot(df["expected_ctc"], kde=True)

plt.title("Distribution of Expected Salary")
plt.xlabel("Expected CTC")
plt.ylabel("Frequency")

plt.show()

# Experience vs Salary
plt.figure(figsize=(8,5))

sns.scatterplot(
    x=df["total_experience"],
    y=df["expected_ctc"]
)

plt.title("Experience vs Expected Salary")

plt.show()

# Salary by Education
plt.figure(figsize=(8,5))

sns.boxplot(
    x=df["education"],
    y=df["expected_ctc"]
)

plt.xticks(rotation=45)

plt.title("Salary Distribution by Education")

plt.show()

# Correlation Heatmap (very important for ML)
plt.figure(figsize=(12,8))

sns.heatmap(
    df.corr(),
    annot=False,
    cmap="coolwarm"
)

plt.title("Feature Correlation Heatmap")

plt.show()

# Top Feature Importance (XGBoost)
xgb.plot_importance(xg, max_num_features=10)

plt.title("Top Important Features")

plt.show()
# Hyperparameter Tuning with GridSearchCV
from sklearn.model_selection import GridSearchCV

param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [3,5,7],
    "learning_rate": [0.01, 0.1],
    "subsample": [0.8,1]
}

xgb_model = xgb.XGBRegressor()

grid = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    cv=3,
    scoring="r2",
    verbose=1,
    n_jobs=-1
)

grid.fit(X_train, y_train)

best_model = grid.best_estimator_

print("Best Parameters:", grid.best_params_)

# Evaluate Tuned Model
pred = best_model.predict(X_test)

print("MAE:", mean_absolute_error(y_test, pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, pred)))
print("R2:", r2_score(y_test, pred))
