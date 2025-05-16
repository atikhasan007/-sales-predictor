import streamlit as st
import pickle
import numpy as np

# মডেল লোড করো
loaded_model = pickle.load(open('/home/catpc/linear_regression_model.pkl', 'rb'))

# ওয়েব অ্যাপের UI
st.title("Scikit-learn Linear Regression Model")

tv = st.text_input('Enter TV sales...')
radio = st.text_input('Enter Radio sales...')
newspaper = st.text_input('Enter Newspaper sales...')

if st.button("Predict"):
    try:
        # ইনপুটগুলো float এ কনভার্ট করো
        tv = float(tv)
        radio = float(radio)
        newspaper = float(newspaper)

        # মডেলের ইনপুট আকারে রূপান্তর
        input_data = np.array([[tv, radio, newspaper]])

        # প্রেডিকশন করো
        prediction = loaded_model.predict(input_data)

        # ফলাফল দেখাও
        st.success(f"Predicted Sales: {prediction[0]:.2f}")
    except ValueError:
        st.error("Please enter valid numeric inputs.")
#source /home/catpc/PycharmProjects/PythonProject/.venv/bin/activate
#pip install streamlit scikit-learn numpy
#streamlit run lr_app.py
