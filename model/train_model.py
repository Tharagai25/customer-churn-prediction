import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Dataset
DATASET_PATH = "../dataset/customer_churn.csv"

# Load dataset
df = pd.read_csv(DATASET_PATH)

print("Dataset loaded successfully!")
print("Columns:")
print(df.columns.tolist())

# Target column
TARGET_COLUMN = "Churn"

# Remove customer ID
if "customerID" in df.columns:
    df = df.drop("customerID", axis=1)

# Convert target
df[TARGET_COLUMN] = df[TARGET_COLUMN].map({
    "Yes": 1,
    "No": 0
})

# Convert categorical columns
categorical_columns = df.select_dtypes(include=["object"]).columns

for column in categorical_columns:
    encoder = LabelEncoder()
    df[column] = encoder.fit_transform(df[column].astype(str))

# Remove missing values
df = df.dropna()

# Features and target
X = df.drop(TARGET_COLUMN, axis=1)
y = df[TARGET_COLUMN]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Random Forest
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Train
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Model trained successfully!")
print("Accuracy:", accuracy)

# Save model into backend folder
joblib.dump(model, "../backend/churn_model.pkl")

print("Model saved successfully!")
print("../backend/churn_model.pkl")