import streamlit as st
import numpy as np
import pandas as pd
import string
import pickle
st.set_option('deprecation.showfileUploaderEncoding',False) 
model = pickle.load(open('logmodel.pkl','rb'))

def main():
    st.title('Advert Predictor Web App')
    st.write('This is a web app that would help in indicate whether or not a particular internet user clicked on an Advertisement on a company website.\
            The Model was made using a fake advertising dataset to train the model.')
    
    Daily_Time_Spent_on_Site = st.sidebar.slider("Enter Daily Time Spent",0.0,100.0)
    Age = st.sidebar.slider("Enter age",0,70)
    Area_Income = st.sidebar.slider("Enter Area Income",0.0,90000.00)
    Daily_Internet_Usage = st.sidebar.slider("Enter Daily Internet Usage",0.0,300.00)
    Male = st.sidebar.slider("Select Gender",0,1)


    inputs = {
        'Daily_Time_Spent_on_Site':Daily_Time_Spent_on_Site,
        'Age':Age,
        'Area_Income':Area_Income,
        'Daily_Internet_Usage':Daily_Internet_Usage,
        'Male':Male}
        
    inputs_df = pd.DataFrame([inputs])
    st.table(inputs_df)

    if st.button('Predict'):
        result = model.predict(inputs_df)
        # updated_res = result.flatten().astype(float)
        st.success('Based on this result {}'.format(result))
        if result == 0:
            st.write('User did not click on Ad.')
        else:
            st.write('User clicked on Ad.')

if __name__ =='__main__':
  main()