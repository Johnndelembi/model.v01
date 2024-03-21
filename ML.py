import streamlit as st
import pickle
import numpy as np

from scikit-learn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report, mean_squared_error, mean_absolute_error

def load_model():
    with open('saved_steps1.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

data = load_model()

regressor = data['model']
le_gender = data['le_gender']
le_mainbranch = data['le_mainbranch']
le_country = data['le_country']
le_education = data['le_education']


def show_predict_page():
    st.title("Software developer salary prediction")
    st.write("Please provide us with information to help us predict your salary")
    
    


    gender = (
        "Man",
        "Woman",
        "Other"
    )

    mainbranch = (
      "I am a developer by profession",
      "I am not primarily a developer, but I write code sometimes as part of my work",
    )

    countries = (
        "United States",
        "United Kingdom",
        "Germany",       
        "Canada",        
        "India",         
        "France",    
        "Brazil",               
        "Netherlands",          
        "Spain",                
        "Australia",            
        "Sweden",               
       "Poland",               
       "Italy",                
       "Russian Federation"   
    )

    education = (
        "Bachelors degree",        
        "Masters degree",          
        "Less than a Bachelors",    
        "Post grad"                
    )



    gender = st.selectbox("Gender", gender)
    age = st.slider("How old are you", 15,70,20)
    country = st.selectbox("Country", countries)
    mainbranch = st.selectbox("Are you a developer by profession or You just casually work with code", mainbranch)
    education = st.selectbox("Education Level", education)
    experience = st.slider("Years of Experience", 0,50,3)

    ok = st.button("Calculate the Salary")
    if ok:
        x = np.array([[gender, mainbranch, country, education, experience, age]])
        x[:,0] = le_gender.transform(x[:,0])
        x[:,1] = le_mainbranch.transform(x[:,1])
        x[:,2] = le_country.transform(x[:,2])
        x[:,3] = le_education.transform(x[:,3])
        x = x.astype(float)


        salary = regressor.predict(x)
        st.subheader(f"Your estimated salary is ${salary[0]:.2f} net/year")

    st.write(" --- ")    

    st.write("Machine Learning model trained and tested by John Ndelembi using Stack Overflow developer salary survey data of 2020, used sci-kit learn for preprocessing, model selection and building")  

    
