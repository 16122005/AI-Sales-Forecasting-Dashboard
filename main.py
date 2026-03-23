import pandas as pd
from src.model import train_model
from src.evaluate import evaluate
from src.visualize import plot_predictions

# Load dataset
df = pd.read_csv('data/Superstore.csv', encoding='latin1')

# Convert date
df['Order Date'] = pd.to_datetime(df['Order Date'])

# Sort by date
df = df.sort_values('Order Date')

# Group by date (daily sales)
df = df.groupby('Order Date')['Sales'].sum().reset_index()

# Rename columns (for simplicity)
df.columns = ['date', 'sales']

# Feature engineering
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['dayofweek'] = df['date'].dt.dayofweek

# Lag features
df['lag_1'] = df['sales'].shift(1)
df['lag_7'] = df['sales'].shift(7)

df = df.dropna()

# Split data
train_size = int(len(df) * 0.8)
train = df[:train_size]
test = df[train_size:]

features = ['year', 'month', 'day', 'dayofweek', 'lag_1', 'lag_7']

X_train = train[features]
y_train = train['sales']
X_test = test[features]
y_test = test['sales']

# Train model
model = train_model(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
mae, rmse = evaluate(y_test, y_pred)

print("MAE:", mae)
print("RMSE:", rmse)

# Plot
plot_predictions(test['date'], y_test, y_pred)
# Predict next 7 days
last_row = df.iloc[-1:].copy()

future_predictions = []

for i in range(7):
    last_row['lag_1'] = last_row['sales']
    
    pred = model.predict(last_row[features])[0]
    future_predictions.append(pred)
    
    last_row['sales'] = pred

print("Next 7 days predictions:")
print(future_predictions)