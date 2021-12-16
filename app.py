import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier

st.write(""" 
# Penguin Predicion App
This app predicts the **Palmer Penguin** species!
Data obtained from the [palmerpenguins library](https://github.com/allisonhorst/palmerpenguins) in R by Allison Horst.)
""")

st.sidebar.header("User Input Features")
st.sidebar.markdown("""  
[Example CSV input file](https://raw.githubusercontent.com/dataprofessor/data/master/penguins_example.csv)
""")
## Collects user input features into dataframe
uploaded_file=st.sidebar.file_uploader("Upload your input csv file",type=["csv"])
if uploaded_file is not None:
    input_df=pd.read_csv(uploaded_file)
else:
    def user_input_features():
        island=st.sidebar.selectbox('Island',('Biscoe','Dream','Torgersen'))
        sex=st.sidebar.selectbox('Sex',('male','female'))
        bill_length_mm=st.sidebar.slider('Bill length(mm)',32.1,59.6,43.9)
        flipper_length_mm=st.sidebar.slider('Flipper length(mm)',172.0,231.0,201.0)
        bill_depth_mm=st.sidebar.slider('Bill depth (mm)',13.1,21.5,17.2)
        body_mass_g=st.sidebar.slider('Body mass(g)',2700.0,63000.0,4207.0)
        data={
              'island':island,
               'sex': sex,
               'bill_length_mm':bill_length_mm,
               'flipper_length_mm':flipper_length_mm,
                'bill_depth_mm':bill_depth_mm,
                 'body_mass_g':body_mass_g
            
             }
        features=pd.DataFrame(data,index=[0])
        return features

    input_df = user_input_features()
        
    ## Combines user innput features with entire penguins dataset

    penguins_raw=pd.read_csv('penguins_cleaned.csv')
    penguins=penguins_raw.drop(columns=['species'])
    df=pd.concat([input_df,penguins],axis=0)

    encode = ['sex', 'island']
    for col in encode:
        dummy = pd.get_dummies(df[col], prefix=col)
        df = pd.concat([df, dummy], axis=1)
        del df[col]

    df=df[:1]
    
    st.subheader('User Input Features')
    if uploaded_file is not None:
        st.write(df)
    else:
         st.write("Awaiting CSV file to  be uploaded .currently using example input features.")
         st.write(df)


    load_model=pickle.load(open('penguins_clf.pkl','rb'))

    prediction=load_model.predict(df)
    prediction_proba = load_model.predict_proba(df)

    st.subheader("Prediction")
    penguins_species=np.array(['Adelie','Chinstrap','Gentoo'])
    st.write(penguins_species[prediction])

    st.subheader('Prediction Probability')
    st.write(prediction_proba)
