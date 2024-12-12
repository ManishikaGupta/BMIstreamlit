import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# Define the health risk predictor class
class HealthRiskPredictor:
    def __init__(self):
        self.model = None
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()

    def generate_training_data(self, n_samples=2000):
        np.random.seed(42)

        # Generate synthetic data
        gender = np.random.choice(['Male', 'Female'], n_samples)
        age = np.random.randint(18, 80, n_samples)
        height = np.random.normal(165, 10, n_samples)  # Height in cm
        weight = np.random.normal(70, 15, n_samples)  # Weight in kg
        waist_circumference = np.random.normal(85, 15, n_samples)
        hip_circumference = np.random.normal(95, 15, n_samples)
        body_fat_percentage = np.random.uniform(10, 40, n_samples)
        resting_heart_rate = np.random.randint(60, 100, n_samples)

        bmi = weight / (height / 100) ** 2
        waist_to_hip_ratio = waist_circumference / hip_circumference

        # Calculate risk scores
        risk_score = (
            (bmi > 25).astype(int) +
            (waist_to_hip_ratio > 0.9).astype(int) +
            (body_fat_percentage > 25).astype(int) +
            (resting_heart_rate > 80).astype(int)
        )

        risk_categories = ['Low', 'Medium', 'High']
        health_risk = [
            risk_categories[min(score, len(risk_categories) - 1)] for score in risk_score
        ]

        data = pd.DataFrame({
            'Gender': gender,
            'Age': age,
            'Height': height,
            'Weight': weight,
            'Waist_Circumference': waist_circumference,
            'Hip_Circumference': hip_circumference,
            'Body_Fat_Percentage': body_fat_percentage,
            'Resting_Heart_Rate': resting_heart_rate,
            'BMI': bmi,
            'Waist_to_Hip_Ratio': waist_to_hip_ratio,
            'Health_Risk': health_risk
        })

        return data

    def train_model(self):
        data = self.generate_training_data()

        # Features and target
        X = data[['Gender', 'Age', 'Height', 'Weight', 'Waist_Circumference', 'Hip_Circumference',
                  'Body_Fat_Percentage', 'Resting_Heart_Rate', 'BMI', 'Waist_to_Hip_Ratio']]
        X = pd.get_dummies(X, columns=['Gender'], drop_first=True)  # Encode Gender
        y = self.label_encoder.fit_transform(data['Health_Risk'])

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train model
        self.model = RandomForestClassifier(random_state=42)
        self.model.fit(X_train, y_train)
        self.training_columns = X_train.columns  # Save the training columns for alignment

    def predict(self, gender, age, height, weight, waist_circumference, hip_circumference, body_fat_percentage, resting_heart_rate):
        bmi = weight / (height / 100) ** 2
        waist_to_hip_ratio = waist_circumference / hip_circumference

        # Create input DataFrame
        input_data = pd.DataFrame({
            'Gender': [gender],
            'Age': [age],
            'Height': [height],
            'Weight': [weight],
            'Waist_Circumference': [waist_circumference],
            'Hip_Circumference': [hip_circumference],
            'Body_Fat_Percentage': [body_fat_percentage],
            'Resting_Heart_Rate': [resting_heart_rate],
            'BMI': [bmi],
            'Waist_to_Hip_Ratio': [waist_to_hip_ratio]
        })

        input_data = pd.get_dummies(input_data, columns=['Gender'], drop_first=True)

        # Align input data with training features
        for col in self.training_columns:
            if col not in input_data.columns:
                input_data[col] = 0

        input_data = input_data[self.training_columns]

        # Predict risk
        risk_encoded = self.model.predict(input_data)[0]
        risk_category = self.label_encoder.inverse_transform([risk_encoded])[0]

        return {
            'BMI': bmi,
            'Waist_to_Hip_Ratio': waist_to_hip_ratio,
            'Health_Risk': risk_category
        }

# Initialize predictor and train model
predictor = HealthRiskPredictor()
predictor.train_model()

# Streamlit App
st.title("Fitness Meter")

st.sidebar.header("Input Parameters")
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
age = st.sidebar.slider("Age", 18, 80, 25)
height = st.sidebar.number_input("Height (cm)", min_value=50, max_value=250, value=165)
weight = st.sidebar.number_input("Weight (kg)", min_value=20, max_value=200, value=70)
waist_circumference = st.sidebar.number_input("Waist Circumference (cm)", min_value=20, max_value=200, value=85)
hip_circumference = st.sidebar.number_input("Hip Circumference (cm)", min_value=20, max_value=200, value=95)
body_fat_percentage = st.sidebar.number_input("Body Fat Percentage (%)", min_value=5, max_value=60, value=20)
resting_heart_rate = st.sidebar.number_input("Resting Heart Rate (bpm)", min_value=30, max_value=200, value=70)

if st.sidebar.button("Calculate"):
    result = predictor.predict(gender, age, height, weight, waist_circumference, hip_circumference, body_fat_percentage, resting_heart_rate)
    bmi = result['BMI']
    risk = result['Health_Risk']
    waist_to_hip_ratio = result['Waist_to_Hip_Ratio']

    if bmi < 18.5:
        body_shape = "Underweight"
    elif 18.5 <= bmi < 24.9:
        body_shape = "Normal"
    elif 25 <= bmi < 29.9:
        body_shape = "Overweight"
    else:
        body_shape = "Obese"

    st.write(f"### Results")
    st.write(f"- **BMI:** {bmi:.2f}")
    st.write(f"- **Waist-to-Hip Ratio:** {waist_to_hip_ratio:.2f}")
    st.write(f"- **Health Risk:** {risk}")
    st.write(f"- **Body Shape:** {body_shape}")
