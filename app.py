from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import pickle

app = Flask(__name__)

# Load model and label encoder
model = pickle.load(open("model.pkl", "rb"))
label_encoder = pickle.load(open("label_encoder.pkl", "rb"))

# Load dataset
df = pd.read_csv("Sneakers.csv")
if df["Discount"].dtype == "object":
    df["Discount"] = df["Discount"].str.replace('%', '', regex=True).astype(float)
df.dropna(subset=["Title", "Brand", "Price", "Discount", "Count_of_Rating"], inplace=True)

title_list = sorted(set(df["Title"].tolist()))

@app.route('/')
def home():
    return render_template("index.html", titles=title_list)

@app.route('/predict', methods=["POST"])
def predict():
    try:
        selected_title = request.form['title']
        price = float(request.form['price'])
        discount = float(request.form['discount'])
        count = float(request.form['count'])

        sneaker = df[df["Title"] == selected_title].iloc[0]
        brand = sneaker["Brand"]

        if any(v <= 0 for v in [price, discount, count]):
            raise ValueError("All numeric values must be greater than 0.")

        if pd.isna(brand) or brand not in label_encoder.classes_:
            raise ValueError("Brand not recognized or missing.")

        brand_encoded = label_encoder.transform([brand])[0]
        features = np.array([[price, discount, count, brand_encoded]])
        prediction = model.predict(features)[0]

        result = "Highly Rated" if prediction == 1 else "Not Highly Rated"
        return render_template("index.html", titles=title_list,
                               prediction_text=f"'{selected_title}' is likely to be: {result}")
    except Exception as e:
        return render_template("index.html", titles=title_list,
                               prediction_text=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
