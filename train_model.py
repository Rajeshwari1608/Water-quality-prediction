import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
import joblib

# Load and parse data
df = pd.read_csv("PB_All_2000_2021.csv", sep=';')
df['year'] = pd.to_datetime(df['date'], dayfirst=True).dt.year

# Features and targets
features = ['year', 'id']
pollutants = ['O2', 'NO3', 'NO2', 'SO4', 'PO4', 'CL']

# Drop rows with missing values
df = df.dropna(subset=features + pollutants)

X = df[features]
X_encoded = pd.get_dummies(X, columns=['id'])
model_columns = X_encoded.columns.tolist()

y = df[pollutants]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2)

# Train model
model = MultiOutputRegressor(RandomForestRegressor())
model.fit(X_train, y_train)

# Save model and columns
joblib.dump(model, "pollution_model.pkl")
joblib.dump(model_columns, "model_columns.pkl")

print("✅ Model and column list saved successfully!")
