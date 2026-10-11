import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


# ==============================
# 1. Load Dataset
# ==============================

df = pd.read_csv(
    "dataset/WA_Fn-UseC_-Telco-Customer-Churn.csv"
)

print("Dataset loaded successfully!")
print("Original shape:", df.shape)


# ==============================
# 2. Data Cleaning
# ==============================

# Remove customer ID
df = df.drop("customerID", axis=1)

# Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

# Remove missing values
df = df.dropna()

# Convert Churn to 0 and 1
df["Churn"] = df["Churn"].map({
    "No": 0,
    "Yes": 1
})


# ==============================
# 3. Separate Input and Target
# ==============================

X = df.drop("Churn", axis=1)
y = df["Churn"]


# ==============================
# 4. Identify Columns
# ==============================

categorical_columns = X.select_dtypes(
    include=["object"]
).columns.tolist()

numerical_columns = X.select_dtypes(
    exclude=["object"]
).columns.tolist()


# ==============================
# 5. Preprocessing
# ==============================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            StandardScaler(),
            numerical_columns
        ),
        (
            "cat",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            ),
            categorical_columns
        )
    ]
)


# ==============================
# 6. Train Test Split
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ==============================
# 7. Transform Data
# ==============================

X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)

y_train = np.array(y_train)
y_test = np.array(y_test)

print("Training shape:", X_train.shape)
print("Testing shape:", X_test.shape)


# ==============================
# 8. Build Deep Learning Model
# ==============================

model = Sequential([

    Dense(
        64,
        activation="relu",
        input_shape=(X_train.shape[1],)
    ),

    Dropout(0.3),

    Dense(
        32,
        activation="relu"
    ),

    Dropout(0.2),

    Dense(
        16,
        activation="relu"
    ),

    Dense(
        1,
        activation="sigmoid"
    )
])


# ==============================
# 9. Compile Model
# ==============================

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)


# ==============================
# 10. Early Stopping
# ==============================

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)


# ==============================
# 11. Train Model
# ==============================

print("\nStarting Deep Learning training...\n")

history = model.fit(
    X_train,
    y_train,
    validation_split=0.2,
    epochs=50,
    batch_size=32,
    callbacks=[early_stopping],
    verbose=1
)


# ==============================
# 12. Evaluate Model
# ==============================

loss, accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=0
)

print("\n==============================")
print("MODEL EVALUATION")
print("==============================")
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")
print(f"Test Accuracy: {accuracy * 100:.2f}%")


# ==============================
# 13. Save Model
# ==============================

model.save(
    "model/customer_churn_model.keras"
)

joblib.dump(
    preprocessor,
    "model/preprocessor.pkl"
)

print("\nModel saved successfully!")
print("Preprocessor saved successfully!")
print("Training completed!")