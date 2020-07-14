
import numpy as np
import pickle
import pandas as pd
import streamlit as st


# Load the logistic regression model
filename = 'classifier.pkl'
classifier = pickle.load(open(filename, 'rb'))

# Load the standardise data
filename2 = 'scaler.pkl'
scaler = pickle.load(open(filename2, 'rb'))

def predict_prob(age,gender,height,weight,ap_hi,ap_lo,cholesterol,gluc,smoke,alco,active):

    if gender == 'Female':
        gender = 1
    else:
        gender = 2

    if cholesterol == 'Normal':
        cholesterol = 1
    elif cholesterol == 'Above normal':
        cholesterol = 2
    else:
        cholesterol = 3

    if gluc == 'Normal':
        gluc = 1
    elif gluc == 'Above Normal':
        gluc = 2
    else:
        gluc = 3

    if smoke == 'Yes':
        smoke = 1
    else:
        smoke = 0

    if alco == 'Yes':
        alco = 1
    else:
        alco = 0

    if active == 'Yes':
        active = 1
    else:
        active = 0


    bmi = (weight)/(height/100)**2
    pulse_pressure = ap_hi - ap_lo

    data = np.array([[age, gender, height, weight, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active,bmi,pulse_pressure]])
    data_s = scaler.transform(data)
    prediction =classifier.predict_proba(data_s)[:,1]

    return round(prediction[0],2)


def main():

    st.sidebar.header('About')
    st.sidebar.info('This app is created to predict Cardiovascular health')
    st.sidebar.info('The dataset consist of 70,000 records of patients data.')
    st.sidebar.success("""Reference link - 
    https://www.kaggle.com/sulianova/cardiovascular-disease-dataset""")

    from PIL  import Image
    image=Image.open('data-original.jpg')
    st.sidebar.image(image,width=300)

    st.title("Cardiovascular Disease Prediction App")

    html_temp = """
       <div style="background-color:#ff6666;padding:5px">
        <h2 style="color:white;text-align:center;">Web app Build using Streamlit, Deployed on Heroku </h2>
       </div>
       """
    st.markdown(html_temp, unsafe_allow_html=True)

    age = st.number_input("Age (In Years)",min_value=0,max_value=100,value=0)
    gender = st.selectbox('Please specify your Gender',('Male','Female'))
    height = st.number_input("Height (in cm)",min_value=0,max_value=300,value=0)
    weight = st.number_input("Weight (in Kg)",min_value=0,max_value=200,value=0)
    ap_hi = st.number_input("Systolic blood pressure (Upper)",min_value=0,max_value=300,value=0)
    ap_lo = st.number_input("Diastolic blood pressure (Lower)",min_value=0,max_value=200,value=0)
    cholesterol = st.selectbox('Select level of cholesterol',('Normal','Above Normal','Well Above Normal'))
    gluc = st.selectbox('Select Level of Glucose',('Normal','Above Normal','Well Above Normal'))
    smoke = st.selectbox('Do you Smoke?',('Yes','No'))
    alco = st.selectbox('Do you drink alcohol?',('Yes','No'))
    active = st.selectbox('Do you workout?',('Yes','No'))

    if st.button("Predict"):
        result = predict_prob(age,gender,height,weight,ap_hi,ap_lo,cholesterol,gluc,smoke,alco,active)
        print(result)

        if result>0.0 and result<=0.5:
             st.success("The Probability of Cardiovascular disorder  is {}, You are in Safe range.".format(result))
        elif result>0.5 and result<=0.7:
             st.warning("The Probability of Cardiovascular disorder  is {}, You have moderate risk of developing Cardiovascular  disorder.".format(result))
        else:
             st.error("The Probability of Cardiovascular disorder  is {}, You have high risk of developing Cardiovascular disorder.".format(result))


if __name__ == '__main__':
    main()

