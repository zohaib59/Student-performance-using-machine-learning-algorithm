import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from autoviz.AutoViz_Class import AutoViz_Class
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor,
                              ExtraTreesRegressor, BaggingRegressor)
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
import warnings
warnings.filterwarnings("ignore")

# Setup
os.chdir("C:\\Users\\zohaib khan\\OneDrive\\Desktop\\USE ME\\dump\\zk")
data = pd.read_csv("student_performance.csv")

# Outlier removal
def remove_outliers(data, col):
    Q1 = data[col].quantile(0.25)
    Q3 = data[col].quantile(0.75)
    IQR = Q3 - Q1
    return data.loc[~((data[col] < (Q1 - 1.5 * IQR)) | (data[col] > (Q3 + 1.5 * IQR)))]

data = remove_outliers(data, "exam_score")

# Encode categoricals
for col in data.select_dtypes(include='object').columns:
    data[col] = LabelEncoder().fit_transform(data[col])

# Features and label
X = data.drop("exam_score", axis=1)
y = data["exam_score"]

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Evaluation function
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero = y_true != 0
    return np.mean(np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero])) * 100

# Models
models = {
    'LinearRegression': LinearRegression(),
    'Ridge': Ridge(),
    'Lasso': Lasso(),
    'DecisionTreeRegressor': DecisionTreeRegressor(random_state=42),
    'KNeighborsRegressor': KNeighborsRegressor(),
    'XGBRegressor': XGBRegressor(random_state=42),
    'CatBoostRegressor': CatBoostRegressor(verbose=False, random_state=42),
    'LGBMRegressor': LGBMRegressor(random_state=42, verbose=-1),
    'RandomForestRegressor': RandomForestRegressor(random_state=42),
    'GradientBoostingRegressor': GradientBoostingRegressor(random_state=42),
    'AdaBoostRegressor': AdaBoostRegressor(random_state=42),
    'ExtraTreesRegressor': ExtraTreesRegressor(random_state=42),
    'BaggingRegressor': BaggingRegressor(random_state=42),
    'SVR': SVR()
}

results_df = pd.DataFrame()

# Train and evaluate
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    comparison = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred, 'Model': name})
    results_df = pd.concat([results_df, comparison], axis=0)

    print(f"{name}: MAE={mean_absolute_error(y_test, y_pred):.2f}, "
          f"MSE={mean_squared_error(y_test, y_pred):.2f}, "
          f"R2={r2_score(y_test, y_pred):.2f}, "
          f"MAPE={mean_absolute_percentage_error(y_test, y_pred):.2f}%")

results_df.to_csv("model_comparisons.csv", index=False)

# Best model plot
best_model = results_df.groupby("Model").apply(lambda x: mean_absolute_error(x['Actual'], x['Predicted'])).idxmin()
best_df = results_df[results_df["Model"] == best_model]

plt.figure(figsize=(10, 6))
sns.scatterplot(x="Actual", y="Predicted", data=best_df)
sns.lineplot(x=best_df["Actual"], y=best_df["Actual"], color='red', label='Perfect Fit')
plt.title(f"Actual vs Predicted: {best_model}")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.legend()
plt.tight_layout()
plt.show()
