import numpy as np
import pickle
import pandas as pd
import streamlit as st

# Load the pre-trained model
with open('model_and_scaler.pkl', 'rb') as file:
    data = pickle.load(file)

# Retrieve the model and scaler
loaded_model = data['model']
loaded_scaler = data['scaler']


def one_hot_encode(value, num_categories):
    one_hot = [0] * num_categories
    if 1 <= value <= num_categories:
        one_hot[int(value) - 1] = 1
    return one_hot


def predict_heart_disease(age, sex, chest_pain_type, resting_bp, cholesterol,
                          fasting_blood_sugar, resting_ecg, max_heart_rate,
                          exercise_angina, oldpeak, st_slope):
    numerical_features = [age, resting_bp, cholesterol, max_heart_rate, oldpeak]
    one_hot_encoded_chest_pain = one_hot_encode(chest_pain_type, 4)
    one_hot_encoded_resting_ecg = one_hot_encode(resting_ecg, 3)
    one_hot_encoded_st_slope = one_hot_encode(st_slope, 3)

    categorical_features = [sex, exercise_angina, fasting_blood_sugar] + \
                           one_hot_encoded_chest_pain + one_hot_encoded_resting_ecg + one_hot_encoded_st_slope
    categorical_features = np.array(categorical_features)
    categorical_features = pd.DataFrame(categorical_features, dtype=int)
    numerical_features = pd.DataFrame(numerical_features)

    # Scale numerical features
    scaled_numerical = loaded_scaler.transform(numerical_features.values.reshape(1, -1)).flatten()
    scaled_numerical = pd.DataFrame(scaled_numerical)

    # Combine all features
    final_features = pd.concat((scaled_numerical, categorical_features))
    final_features = final_features.T

    # Predict the probability of heart disease
    prediction = loaded_model.predict_proba(final_features)[0][1]
    return prediction


def main():
    st.title("Heart Disease Prediction using ML")
    # html_temp = """
    # <div style="background-color:tomato;padding:10px">
    # <h2 style="color:white;text-align:center;">Heart Disease Prediction ML App</h2>
    # </div>
    # """
    # st.markdown(html_temp, unsafe_allow_html=True)

    # Input fields for the model
    age = st.number_input("Age", min_value=0, max_value=120, value=25)
    sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
    chest_pain_type = st.selectbox("Chest Pain Type", options=[1, 2, 3, 4], format_func=lambda x: f"Type {x}")
    resting_bp = st.number_input("Resting Blood Pressure (in mm Hg)", min_value=0, max_value=300, value=120)
    cholesterol = st.number_input("Cholesterol (in mg/dl)", min_value=0, max_value=600, value=200)
    fasting_blood_sugar = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[0, 1],
                                       format_func=lambda x: "No" if x == 0 else "Yes")
    resting_ecg = st.selectbox("Resting ECG results", options=[0, 1, 2], format_func=lambda x: f"Result {x}")
    max_heart_rate = st.number_input("Max Heart Rate Achieved", min_value=0, max_value=250, value=150)
    exercise_angina = st.selectbox("Exercise Angina", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    oldpeak = st.number_input("Oldpeak (depression induced by exercise relative to rest)", min_value=0.0,
                              max_value=10.0, value=0.0)
    st_slope = st.selectbox("Slope of the Peak Exercise ST Segment", options=[0, 1, 2],
                            format_func=lambda x: f"Slope {x}")

    if st.button("Predict"):
        prediction = predict_heart_disease(age, sex, chest_pain_type, resting_bp, cholesterol,
                                           fasting_blood_sugar, resting_ecg, max_heart_rate,
                                           exercise_angina, oldpeak, st_slope)

        # Define the risk thresholds
        if prediction >= 0.5:
            st.markdown(f"<h3 style='color:red;'>High risk of heart disease ({100*prediction:.2f}%)</h3>",
                        unsafe_allow_html=True)
            st.warning("It is recommended to consult a doctor immediately.")
        elif 0.3 <= prediction < 0.5:
            st.markdown(f"<h3 style='color:orange;'>Moderate risk of heart disease ({100*prediction:.2f}%)</h3>",
                        unsafe_allow_html=True)
            st.info("You should consider improving your lifestyle and consulting a healthcare professional for advice.")
        else:
            st.markdown(f"<h3 style='color:green;'>Low risk of heart disease ({100*prediction:.2f}%)</h3>",
                        unsafe_allow_html=True)
            st.success("You appear to be at low risk. Keep maintaining a healthy lifestyle.")


    # Custom CSS for Streamlit button styling
    st.markdown("""
        <style>
        .stButton > button {
            padding: 10px 20px;
            background-color: white;
            font-size: 16px;
            border: black 1px solid;
            border-radius: 0px;
            cursor: pointer;
            transition: background-color 0.3s;
            color: black;
        }
        .stButton > button:hover {
            background-color: black;
            color: white;
            border: black 1px solid;
        }
        .center-button {
            display: flex;
            justify-content: center;
            margin-top: 20px;
        }
        </style>
    """, unsafe_allow_html=True)

    # Center the button using Streamlit's built-in components
    st.markdown('<div class="center-button">', unsafe_allow_html=True)
    if st.button("About"):
        st.text("Learn more about heart disease prediction using machine learning.")
        st.markdown(
            "[Get the code for this ML model](https://colab.research.google.com/drive/1--80FlDkbe1ogtuzjymtz-yB4HCyiAhH?usp=sharing)",
            unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


if __name__ == '__main__':
    main()
