
import sys
import subprocess
import os
import warnings
warnings.filterwarnings("ignore")

def install(package: str):
    try:
        __import__(package)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

for pkg in ["pandas", "numpy", "scikit-learn", "xgboost"]:
    install(pkg)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
try:
    from xgboost import XGBRegressor
    USE_XGB = True
except Exception:
    USE_XGB = False

RANDOM_STATE = 42
INPUT_DIR = "./input"
OUTPUT_DIR = "./output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

train_path = os.path.join(INPUT_DIR, "train.csv")
test_path = os.path.join(INPUT_DIR, "test.csv")
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

def add_features(df):
    df = df.copy()
    df["rooms_per_household"] = df["total_rooms"] / df["households"]
    df["bedrooms_per_household"] = df["total_bedrooms"] / df["households"]
    df["population_per_household"] = df["population"] / df["households"]
    df["rooms_per_bedroom"] = df["total_rooms"] / (df["total_bedrooms"] + 1e-6)
    return df

X = add_features(train_df.drop(columns=["median_house_value"]))
y = train_df["median_house_value"]
test_X = add_features(test_df)

numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent"))
])

if categorical_features:
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ])
else:
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features)
        ])

if USE_XGB:
    model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        n_jobs=4,
        objective="reg:squarederror",
        reg_alpha=0.0,
        reg_lambda=1.0
    )
else:
    model = RandomForestRegressor(
        n_estimators=500,
        max_depth=None,
        random_state=RANDOM_STATE,
        n_jobs=4
    )

pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", model)
])

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)

pipeline.fit(X_train, y_train)
pred_val = pipeline.predict(X_val)

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(y_val, pred_val))
print(f"Final Validation Performance: {rmse}")

# Train on full data and predict test set
pipeline.fit(X, y)
test_pred = pipeline.predict(test_X)

output_path = os.path.join(OUTPUT_DIR, "predictions.csv")
pd.DataFrame({"median_house_value": test_pred}).to_csv(output_path, index=False)
