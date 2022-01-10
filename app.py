import streamlit as st
import pickle
from datetime import datetime

import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.offline as py
from plotly.offline import iplot
import cufflinks as cf
cf.go_offline()
cf.set_config_file(offline=False, world_readable=True)
startTime = datetime.now()
# import znanych nam bibliotek

filename = "model.sv"
model = pickle.load(open(filename,'rb'))
# otwieramy wcześniej wytrenowany model

sex_d = {0:"Kobieta",1:"Mężczyzna"}
chest_pain_type_d = {0:"Bezobjawowy", 1:"Nietypowa dusznica bolesna", 2:"Ból bez dusznicy bolesnej", 3:"Typowa dusznica bolesna"}
fasting_bs_d = {0:"Inaczej", 1:"BS > 120 mg/dl"}
resting_ecg_d = {0:"LVH", 1:"Prawidłowe", 2:"ST"}
resting_ecg_descr = "1. LVH: Prawdopodobny lub wyraźny przerost lewej komory według kryteriów Estesa \n2. Prawidłowe\n3. ST: Z nieprawidłowościami załamka ST-T (odwrócenie załamka T i/lub uniesienie lub obniżenie odcinka ST > 0,05 mV)"
exercise_angina_d = {0:"Nie", 1:"Tak"}
exercise_angina_descr = "Dusznica piersiowa wywołana wysiłkiem fizycznym"
oldpeak_descr = "Oldpeak = ST [Wartość liczbowa zmierzona w depresji]"
st_slope_d = {0:"Down", 1:"Flat", 2:"Up"}
st_slope_descr = "Nachylenie szczytowego odcinka ST ćwiczenia [Up: upsloping, Flat: flat, Down: downsloping]"

# F = 0, M = 1
# ASY = 0, ATA = 1, NAP = 2, TA = 3
# LVH = 0, Normal = 1, ST = 2
# N = 0, Y = 1
# Down = 0, Flat = 1, Up = 2
def main():
	st.set_page_config(page_title="Przewidywanie niewydolności serca")
	overview = st.container()
	col1, col2, col3, col4 = st.columns([1, 3, 3, 1])

	with overview:


		st.title("Przewidywanie niewydolności serca")
		st.image("heart.png")
		load_data()

		st.subheader("Dane ogólne")

	with col2:
				sex = st.radio( "Płeć", list(sex_d.keys()), format_func=lambda x : sex_d[x] )
	with col3:
				age = st.number_input("Wprowadź wiek", 1, 100, 30, 1)

	st.subheader("Dane medyczne")
	col1, col2, col3 = st.columns([3, 3, 3])
	
	with col1:
				chest_pain_type = st.selectbox( "Rodzaj bólu w klatce piersiowej", list(chest_pain_type_d.keys()), format_func=lambda x : chest_pain_type_d[x])
				fasting_bs = st.selectbox( "Poziom cukru na czczo", list(fasting_bs_d.keys()), format_func=lambda x : fasting_bs_d[x])
				exercise_angina = st.selectbox("Dusznica wywołana WF", list(exercise_angina_d.keys()), format_func=lambda x : exercise_angina_d[x], help=exercise_angina_descr)
	with col2:
				bpm = st.number_input("Spoczynkowe ciśnienie krwi [mm Hg]", 50, 220, 90, 5)
				resting_ecg = st.selectbox( "EKG spoczynkowe", list(resting_ecg_d.keys()), format_func=lambda x : resting_ecg_d[x], help=resting_ecg_descr)
				oldpeak = st.number_input("Oldpeak", -7.00, 7.00, 0.00, 0.50, help=oldpeak_descr, format="%.2f")
	with col3:
				cholesterol = st.number_input("Cholesterol [mm/dl]", 0, 1000, 0, 100)
				max_hr = st.number_input("Maksymalne tętno", 60, 220, 140, 10)
				st_slope = st.selectbox( "EKG spoczynkowe", list(st_slope_d.keys()), format_func=lambda x : st_slope_d[x], help=st_slope_descr)

	data = [[sex, age, chest_pain_type, bpm, cholesterol, fasting_bs, resting_ecg, max_hr, exercise_angina, oldpeak, st_slope]]
	heart_disease = model.predict(data)
	s_confidence = model.predict_proba(data)

	prediction = st.container()

	with prediction:
		st.header("Przeanalizowane wyniki na podstawie danych. \nWynik:  {0}".format("Niewydolności serca" if heart_disease[0] == 0 else "Odczyt prawidłowy"))
		st.subheader("Pewność predykcji {0:.2f} %".format(s_confidence[0][heart_disease][0] * 100))

def load_data():
	base_data = pd.read_csv("heart.csv");

	cols = ["HeartDisease","Age","Sex","ChestPainType","RestingBP","Cholesterol","FastingBS","RestingECG","MaxHR","ExerciseAngina","Oldpeak","ST_Slope",]
	data = base_data[cols].copy()


	with st.expander("Dane źródłowe i ich analiza"):
		st.dataframe(base_data, 900, 250)
	
		fig = px.histogram(base_data, x="Sex", color="HeartDisease",width=400, height=400)
		st.plotly_chart(fig, use_container_width=True)

		numerical= base_data.drop(['HeartDisease'], axis=1).select_dtypes('number').columns
		categorical = base_data.select_dtypes('object').columns

		for i in numerical:
			fig = base_data[i].iplot(asFigure=True, kind="box", title=i, boxpoints="all")
			st.plotly_chart(fig, use_container_width=True)
		
if __name__ == "__main__":
    main()
