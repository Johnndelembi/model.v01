import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

def show_explore_page():
    df = pd.read_csv("clean_survey_results_public.csv")
    
    st.title("Explore software developer Salary Insights")
    st.write("**Stack Overflow developer salary 2020**")

    st.write("Boxplot analysis of salary versus Country and Education level....")
    st.image("boxplot.PNG")
    st.image("boxplot 2.PNG")

    st.write("Values counts analysis of the countries used in the model and the education level categories")
    st.image("count1.PNG")
    st.image("count2.PNG")

    st.write("Machine Learning model trained and tested by John Ndelembi using Stack Overflow developer salary survey data of 2020, used sci-kit learn for preprocessing, model selection and building")  


