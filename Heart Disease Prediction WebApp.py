# -*- coding: utf-8 -*-
"""
Created on Sun Apr 13 18:59:19 2025

@author: Udayan
"""

import numpy as np
import pickle
import streamlit as st
import matplotlib.pyplot as plt

# Load the saved model
loaded_model = pickle.load(open("heart_disease.sav", 'rb'))

# Prediction function
def HeartDiseasePrediction(input_data):
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    prediction = loaded_model.predict(input_data_reshaped)

    if prediction[0] == 0:
        result_text = "The person does NOT have any heart disease"
        bar_color = 'green'
        bar_values = [1, 0]
    else:
        result_text = "The person HAS heart disease"
        bar_color = 'red'
        bar_values = [0, 1]

    return result_text, bar_values, bar_color

# Main function
def main():
    st.title("Heart Disease Prediction Web Application")

    # Text input fields
    age = st.text_input('Your Age')
    sex = st.text_input('Your Sex (1 = male, 0 = female)')
    cp = st.text_input('Chest Pain Type (0–3)')
    trestbps = st.text_input('Resting Blood Pressure Value')
    chol = st.text_input('Serum Cholesterol Value')
    fbs = st.text_input('Fasting Blood Sugar > 120 mg/dl (1 = true; 0 = false)')
    restecg = st.text_input('Resting ECG Result (0–2)')
    thalach = st.text_input('Maximum Heart Rate Achieved')
    exang = st.text_input('Exercise Induced Angina (1 = yes; 0 = no)')
    oldpeak = st.text_input('ST Depression Induced by Exercise (0.0-9.9)')
    slope = st.text_input('Slope of Peak Exercise ST Segment (0–2)')
    ca = st.text_input('Number of Major Vessels Colored by Fluoroscopy (0–3)')
    thal = st.text_input('Thalassemia (1 = normal; 2 = fixed defect; 3 = reversible defect)')

    if st.button('Heart Disease Test Result'):
        try:
            # Convert all inputs to float
            input_data = [
                float(age), float(sex), float(cp), float(trestbps), float(chol),
                float(fbs), float(restecg), float(thalach), float(exang),
                float(oldpeak), float(slope), float(ca), float(thal)
            ]

            result_text, bar_values, bar_color = HeartDiseasePrediction(input_data)

            st.success(result_text)

            # Plotting bar chart
            feature_names = ["Healthy Heart", "Diseased Heart"]
            fig, ax = plt.subplots(figsize=(5, 3))
            ax.bar(feature_names, bar_values, color=bar_color, width=0.3)
            ax.set_title("Heart Disease Prediction\n\n" + result_text, fontsize=12)
            ax.set_yticks([])
            ax.set_ylim(0, 1.2)
            ax.grid(axis='y', linestyle='--', alpha=0.2)
            plt.tight_layout()
            st.pyplot(fig)

        except ValueError:
            st.error("❌ Please enter only numeric values in all fields.")

if __name__ == '__main__':
    main()


    
