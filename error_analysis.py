import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.ensemble import StackingRegressor
df = pd.read_csv("final_dataset.csv")


x = df.drop(columns=['AQI', 'Date_time'])
x = x.select_dtypes(include='number')
y = df['AQI']
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42)

estimators = [
    ('lr', LinearRegression()),
    ('rf', RandomForestRegressor(n_estimators=200, random_state=42)),
    ('xgb', XGBRegressor(n_estimators=200, random_state=42))
]
stack = StackingRegressor(estimators=estimators, final_estimator=LinearRegression())
stack.fit(x_train, y_train)
y_pred_stack = stack.predict(x_test)
print("Stacking Regressor R²:", r2_score(y_test, y_pred_stack))


#Predicted vs Actual AQI Plot
rf = RandomForestRegressor(
    n_estimators=200, max_depth=None, random_state=42, n_jobs=-1
)
rf.fit(x_train, y_train)
y_pred_rf = rf.predict(x_test)
best_pred = y_pred_rf 

plt.figure(figsize=(6,6))
sns.scatterplot(x=y_test, y=best_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.title("Predicted vs Actual AQI")
plt.xlabel("Actual AQI")
plt.ylabel("Predicted AQI")
plt.show()

#Residual Analysis
residuals = y_test - best_pred
plt.figure(figsize=(10,4))
sns.histplot(residuals, kde=True, bins=30, color='teal')
plt.title("Distribution of Residuals")
plt.xlabel("Residual (Error)")
plt.show()

