import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Read your Tesla dataset
df = pd.read_csv('tesla_data.csv')
print("======== Model Validation and Accuracy Calculations ========")

df['Date'] = pd.to_datetime(df['Date'])
# Create new columns for 'Day_of_Week' and 'Month' extracted from the 'Date' column
df['Day_of_Week'] = df['Date'].dt.weekday
df['Month'] = df['Date'].dt.month

# Define the features (X) and the target variable (y)
from sklearn.model_selection import train_test_split
X = df[['Day_of_Week', 'Month', 'Open', 'High', 'Low', 'Volume']]
y = df['Close']              # target variable (close)
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


from sklearn.model_selection import KFold
kf = KFold(n_splits=5)

from sklearn.linear_model import LinearRegression

# Loop over each fold in the dataset
for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    model = RandomForestRegressor(n_estimators=100, max_depth=5)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)


# Print the actual and predicted values for a few sample indices
sample_indices = [3, 10, 12]
for i in sample_indices:
       print("Actual Close Price:", y_test.iloc[i], "Predicted Close Price:", y_pred[i])

from sklearn.metrics import mean_absolute_percentage_error
# Calculate the MAPE
mape = mean_absolute_percentage_error(y_test, y_pred)
print('Mean Absolute Percentage Error (MAPE):', mape)

from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
mse_scores = -scores
print('Cross-Validation MSE Scores:', mse_scores)
print('Average Cross-Validation MSE:', mse_scores.mean())

# Calculate evaluation metrics
from sklearn.metrics import mean_squared_error, r2_score

# Calculate and print the MSE and R^2
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print('Mean Squared Error (MSE):', mse)
print('R-squared:', r2)

import matplotlib.pyplot as plt

importances = model.feature_importances_
feature_names = ['Day_of_Week', 'Month', 'Open', 'High', 'Low', 'Volume']

plt.bar(feature_names, importances)
plt.xticks(rotation=45)
plt.title("Feature Importances")
plt.show()

import joblib
joblib.dump(model, 'model.pkl')