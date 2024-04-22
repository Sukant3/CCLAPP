from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.neural_network import MLPClassifier

app = Flask(__name__)

# Load the breast cancer dataset
breast_cancer_dataset = load_breast_cancer()
X = breast_cancer_dataset.data
y = breast_cancer_dataset.target

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train the model
model = MLPClassifier(hidden_layer_sizes=(30,30,30), max_iter=10)
model.fit(X_train, y_train)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    # Get the input data from the form
    input_data = []
    for i in range(30):
        input_data.append(float(request.form[f'param_{i}']))

    # Create a DataFrame with the input data
    input_df = pd.DataFrame([input_data])

    # Standardize the input data
    input_data_scaled = scaler.transform(input_df)

    # Make predictions
    prediction = model.predict(input_data_scaled)[0]
    if prediction == 0:
        result = 'Malignant'
    else:
        result = 'Benign'

    return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
