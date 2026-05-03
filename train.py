# =========================
# TRAINING FILE
# =========================

import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# =========================
# LOAD DATASET
# =========================

df = pd.read_csv(r"C:\Users\usee\Documents\HousePricePrediction\Bengaluru_Data.csv")

# =========================
# SELECT REQUIRED COLUMNS
# =========================

df = df[['total_sqft', 'bath', 'size', 'price']]

# =========================
# REMOVE NULL VALUES
# =========================

df = df.dropna()

# =========================
# CONVERT SIZE COLUMN
# Example: "2 BHK" -> 2
# =========================

df['bhk'] = df['size'].apply(lambda x: int(x.split(' ')[0]))

# =========================
# DROP SIZE COLUMN
# =========================

df = df.drop('size', axis=1)

# =========================
# CLEAN total_sqft COLUMN
# =========================

def convert_sqft(x):

    try:

        if '-' in str(x):

            nums = x.split('-')

            return (float(nums[0]) + float(nums[1])) / 2

        return float(x)

    except:

        return None

df['total_sqft'] = df['total_sqft'].apply(convert_sqft)

# Remove invalid rows
df = df.dropna()

# =========================
# FEATURES AND TARGET
# =========================

X = df[['total_sqft', 'bath', 'bhk']]

y = df['price']

# =========================
# TRAIN TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =========================
# TRAIN MODEL
# =========================

model = LinearRegression()

model.fit(X_train, y_train)

# =========================
# CHECK ACCURACY
# =========================

y_pred = model.predict(X_test)

score = r2_score(y_test, y_pred)

print("Model Accuracy:", score)

# =========================
# SAVE MODEL
# =========================

pickle.dump(model, open("model.pkl", "wb"))

print("Model Saved Successfully!")