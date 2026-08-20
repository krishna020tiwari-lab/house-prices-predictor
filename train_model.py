import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


csv_content = """SquareFeet,Bedrooms,Age,Price
1200,2,10,250000
1500,3,5,310000
1800,3,15,340000
2400,4,2,480000
3000,4,8,570000
3500,5,1,680000
1100,2,20,210000
2100,3,12,400000
2600,4,6,510000
1700,3,18,320000
2900,4,4,550000
3200,4,3,610000"""

with open('house_prices.csv', 'w') as file:
    file.write(csv_content)

print("Saved 'house_prices.csv' to project directory.\n")

df = pd.read_csv('house_prices.csv')

X = df[['SquareFeet', 'Bedrooms', 'Age']]
y = df['Price']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = LinearRegression()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)

print("--- Model Performance Metrics ---")
print(f"R-squared (R2 Score): {r2_score(y_test, y_pred):.4f}")
print(f"Root Mean Squared Error (RMSE): ${np.sqrt(mean_squared_error(y_test, y_pred)):,.2f}\n")

print("--- Feature Coefficients (Weights) ---")
for col, coef in zip(X.columns, model.coef_):
    print(f"{col}: ${coef:,.2f}")
print(f"Intercept: ${model.intercept_:,.2f}")

joblib.dump(model, 'linear_house_model.pkl')


plt.scatter(y_test, y_pred, color='blue', s=100)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
plt.xlabel("Actual Price ($)")
plt.ylabel("Predicted Price ($)")
plt.title("Actual vs Predicted House Prices")
plt.tight_layout()
plt.show()
