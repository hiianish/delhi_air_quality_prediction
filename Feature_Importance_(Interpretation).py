import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("final_dataset")

#Select Features and Target
x = df.drop(columns=['AQI', 'Date_time'])
x = x.select_dtypes(include='number')
y = df['AQI']



#Split Dataset
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42)

print("Train shape:", x_train.shape)
print("Test shape:", x_test.shape)



#Baseline Model – Linear Regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np

lr = LinearRegression()
lr.fit(x_train, y_train)

y_pred_lr = lr.predict(x_test)


r2 = r2_score(y_test, y_pred_lr)
mae = mean_absolute_error(y_test, y_pred_lr)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_lr))

print("Linear Regression →  R2:", r2, " MAE:", mae, " RMSE:", rmse)



#Random Forest Regressor
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(
    n_estimators=200, max_depth=None, random_state=42, n_jobs=-1
)
rf.fit(x_train, y_train)
y_pred_rf = rf.predict(x_test)

r2_rf = r2_score(y_test, y_pred_rf)
mae_rf = mean_absolute_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))

print("Random Forest →  R2:", r2_rf, " MAE:", mae_rf, " RMSE:", rmse_rf)




#XGBoost Regressor (Optimized Gradient Boosting)
from xgboost import XGBRegressor

xgb = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
xgb.fit(x_train, y_train)
y_pred_xgb = xgb.predict(x_test)

r2_xgb = r2_score(y_test, y_pred_xgb)
mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))

print("XGBoost →  R2:", r2_xgb, " MAE:", mae_xgb, " RMSE:", rmse_xgb)


results = pd.DataFrame({
    'Model': ['Linear Regression', 'Random Forest', 'XGBoost'],
    'R2': [r2, r2_rf, r2_xgb],
    'MAE': [mae, mae_rf, mae_xgb],
    'RMSE': [rmse, rmse_rf, rmse_xgb]
})
print(results)

sns.barplot(x='Model', y='R2', data=results)
plt.title("Model Performance Comparison (R² Score)")
plt.show()

###########################

#Random Forest
importances = pd.Series(rf.feature_importances_, index=x.columns)
importances.nlargest(10).sort_values().plot(kind='barh', figsize=(8,5))
plt.title("Random Forest Feature Importance")
plt.show()


#XGBoost
xgb_importances = pd.Series(xgb.feature_importances_, index=x.columns)
xgb_importances.nlargest(10).sort_values().plot(kind='barh', color='teal', figsize=(8,5))
plt.title("XGBoost Feature Importance")
plt.show()

#using shap
import shap
explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(x_test)
shap.summary_plot(shap_values, x_test)
