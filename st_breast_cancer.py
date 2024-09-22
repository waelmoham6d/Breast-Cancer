import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import pandas as pd

model = load_model(r'C:\Users\mwael\OneDrive\Desktop\after_cource\model_name.h5')

feature_labels = ['radius_mean',
 'texture_mean',
 'perimeter_mean',
 'area_mean',
 'smoothness_mean',
 'compactness_mean',
 'concavity_mean',
 'concave_points_mean',
 'symmetry_mean',
 'radius_se',
 'perimeter_se',
 'area_se',
 'compactness_se',
 'concavity_se',
 'concave_points_se',
 'radius_worst',
 'texture_worst',
 'perimeter_worst',
 'area_worst',
 'smoothness_worst',
 'compactness_worst',
 'concavity_worst',
 'concave_points_worst',
 'symmetry_worst',
 'fractal_dimension_worst']

scaler = StandardScaler()

st.title("**Breast Cancer Diagnosis Prediction**")

st.write(
    """
### يرجى إدخال قيم الخصائص التالية لتنبؤ حالة التشخيص ان كان سلبي أم إيجابي 
    """
         )

input_data = []
for i in feature_labels:

    value = st.text_input(f"أدخل {i}")  
    try:
        input_data.append(float(value))  
    except ValueError:
        st.write(f"الرجاء إدخال قيمة رقمية صالحة ل {i}")
        st.stop()

if st.button('Prediction'):

    input_data = np.array(input_data).reshape(1, -1)
    
    input_data_scaled = scaler.fit_transform(input_data)

    prediction = model.predict(input_data_scaled)

    print(prediction)

    diagnosis = 'Positive( + )' if prediction[0][0] > 0.5 else 'Negative( - )'

    st.write(f" Expected diagnosis: **{diagnosis}**")