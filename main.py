import streamlit as st
from ML import show_predict_page
from explore import show_explore_page
#st.set_page_config(page_title='MODEL.v01')



page = st.sidebar.selectbox("Predict/Explore", ("Predict", "Explore"))
    
if page == 'Predict':
    show_predict_page()
else:
    show_explore_page()   
            
st.sidebar.caption("John Ndelembi. 2024")  
st.sidebar.caption("MODEL.v01")

