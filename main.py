# This is a sample Python script.
#import pickle
import joblib as jl
import pandas as pd
import streamlit as st

st.header("A machine learning-based model to predict 1-year mortality among elderly patients with coronary heart disease and impaired glucose tolerance")
st.sidebar.title("Selection Panel")
st.sidebar.markdown("Picking up parameters")

Hb = st.sidebar.slider("Hemoglobin (g/dL)", 65, 160)
HDL = st.sidebar.slider("High density lipoprotein (mmol/l)", 0.25, 2.25)
albumin = st.sidebar.slider("Albumin (mg/dL)", 20.0, 50.0)
Scr = st.sidebar.slider("Serum creatinine concentration (μmol/L)", 60.0, 200.0)
NTproBNP = st.sidebar.slider("NT-proBNP (pg/ml)", 100, 2000)
CHF1 = st.sidebar.selectbox("Cardiac insufficiency", ("No", "Yes"))
Statins = st.sidebar.selectbox("Use of statins", ("No", "Yes"))

if st.button("Submit"):
    gbm_clf = jl.load("gbm_clf_final_round.web.pkl")
    x = pd.DataFrame([[Hb, HDL, albumin, Scr, NTproBNP, CHF1, Statins]],
                     columns=["Hb", "HDL", "albumin", "Scr", "NT.proBNP", "CHF1", "Statins"])
    x = x.replace(["No", "Yes"], [0, 1])
    # Get prediction
    prediction = gbm_clf.predict_proba(x)[0, 1]
        # Output prediction
    st.text(f"One year mortality: {'{:.2%}'.format(round(prediction, 5))}")

st.subheader('Introduction')
st.markdown('This web-based calculator was developed based on the gradient boosting machine, with an AUC of 0.836 (95% CI: 0.743-0.929) and Brier score of 0.116. By choosing parameters and clicking "Calculate" button, users can get the risk of 1-year mortality for specific cases.')