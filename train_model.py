import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import pickle

df = pd.read_csv("Sneakers.csv")

if df["Discount"].dtype == "object":
    df["Discount"] = df["Discount"].str.replace('%', '', regex=True).astype(float)

df.dropna(inplace=True)

le = LabelEncoder()
df["Brand_encoded"] = le.fit_transform(df["Brand"])
df["isHighlyRated"] = (df["Rating"] >= 4).astype(int)

X = df[["Price", "Discount", "Count_of_Rating", "Brand_encoded"]]
y = df["isHighlyRated"]

xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(class_weight='balanced', random_state=42)
model.fit(xtrain, ytrain)

ypred = model.predict(xtest)
print("Accuracy:", accuracy_score(ytest, ypred))
print("Classification Report:\n", classification_report(ytest, ypred))

pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(le, open("label_encoder.pkl", "wb"))
