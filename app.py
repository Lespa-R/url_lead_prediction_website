import streamlit as st
import requests

'''
# Qualify front
'''

urls = st.text_input('Insert url(s)', 'www.lewagon.com')

param = {'urls': urls}

url = 'https://api-gwwhhqf6zq-uc.a.run.app/'

prediction = requests.get(url, params=param)

st.write(prediction.json())
