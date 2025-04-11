import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Predição de Evasão", layout="centered")
st.title("🔍 Predição de Evasão Estudantil")

st.markdown("Envie um CSV com os dados dos estudantes para prever se haverá evasão.")

model = joblib.load("model/modelo_treinado.pkl")

uploaded_file = st.file_uploader("Selecione o arquivo .csv", type="csv")

if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
    st.write("📋 Dados Recebidos:")
    st.dataframe(input_df)

    prediction = model.predict(input_df)
    proba = model.predict_proba(input_df)

    results = pd.DataFrame({
        'Evadido (0 = Não, 1 = Sim)': prediction,
        'Probabilidade de Evasão': proba[:, 1]
    })

    st.write("📊 Resultado da Predição:")
    st.dataframe(results)
