##############################
########### Block 1 ##########
##############################

# imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from streamlit_option_menu import option_menu

# define dataframe
# mydic = {"one":[3,4,5,2,7,3,4,5,2,7,3,4,5,2,7,3,4,5,2,7,3,4,5,2,7,3,4,5,2,7],
#          "two":[13,4,6,2,9,13,4,6,2,9,13,4,6,2,9,13,4,6,2,9,13,4,6,2,9,13,4,6,2,9],
#          "three":[1,4,23,2,5,1,4,23,2,5,1,4,23,2,5,1,4,23,2,5,1,4,23,2,5,1,4,23,2,5],
#          "four":[2,4,5,12,71,2,4,5,12,71,2,4,5,12,71,2,4,5,12,71,2,4,5,12,71,2,4,5,12,71]}
# df = pd.DataFrame(mydic)


#Streamlit
with st.sidebar:

    
    selected = option_menu("Main Menu", ["Home", "Data", 'Plot', 'Settings'], 
        icons=['house', "table", 'bar-chart-fill', 'gear'], menu_icon="cast", default_index=1)
#     selected

    uploaded_file = st.file_uploader("Choose a file")


    
try:
    df = pd.read_csv(uploaded_file)
    if selected == "Home":
        st.write("hi, This is my app")
        st.markdown("**Sarah**")
        st.code("if x==1:\n x=x+1")

    if selected == "Data":
        st.write(df)
        st.write(df.columns)
        st.write(df.describe())
        st.write(df)


    if selected == "Plot":
        st.header("Plotting")
        col = st.radio(
             "Column:",
             tuple(df.columns))

    #     col = st.select_slider(
    #          'Select column',
    #          options=df.columns)
        index = st.slider('Index', 1, df.shape[0], 1)

        fig, ax = plt.subplots()
        ax.plot(df.loc[:index-1,col])

        st.pyplot(fig)
except: 
    if selected == "Home":
        st.header("Home")
        st.write("Sorry you have not chosen any files")

    if selected == "Data":
        st.header("Data")
        st.write("Sorry you have not chosen any files")


    if selected == "Plot":
        st.header("Plotting")
        st.write("Sorry you have not chosen any files")
    
    
    
