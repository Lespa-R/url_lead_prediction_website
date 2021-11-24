# from enum import auto
import streamlit as st
# import altair as alt
from PIL import Image
import pandas as pd
# import numpy as np
import requests
import re
import time

CSS = """
etr89bj1 {
    width:150px!important;
    }

.stApp {
    background-image: url(https://images.prismic.io/ankor/7a8e937c-b781-40d2-b06e-4e4dacec4475_OMY_BANNIERE.jpg);
    background-size: cover;
    font-family: system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif,Apple Color Emoji,Segoe UI Emoji;
}

h1 {
    padding:0;
    font-weight: 600;
}

h2 {
    padding:0;
}

.block-container {
    background-color:#ffffff;
    padding: 1rem 1rem!important;
}
"""

st.write(f'<style>{CSS}</style>', unsafe_allow_html=True)


lewagon = Image.open('images/lewagon.png')
ankorstore = Image.open('images/ankorstore.png')

columns = st.columns(2)

logo1 = columns[0].image(lewagon, use_column_width=False)
logo2 = columns[1].image(ankorstore, use_column_width=False)

'''
# Commercial 2.0
## Predictive qualification from a website
'''

def regex_url(urls_list):
    r = re.compile(
    '((http|https)\:\/\/)?[a-zA-Z0-9\.\/\?\:@\-_=#]+\.([a-zA-Z]){2,6}([a-zA-Z0-9\.\&\/\?\:@\-_=#])*'
    )
    i=0
    for line in urls_list:
        line = line.strip()
        if not r.match(line):
            i=i+1
    return i


def get_dataframe_data():
    return pd.read_csv('data/small_cleaned_data.csv', sep=";")

urls = st.text_area("Insert one or more url(s) separated by commas:",
                    "www.lewagon.com, (...)")
urls_list = urls.split(",")
st.write("Number of url(s):", len(urls_list))

if st.button('Launch qualification 🎉'):
    bad_urls = regex_url(urls_list)

    if bad_urls != 0:
        st.warning('Stay positive, some urls are just maybe not well written.')
    else:
        # Add a placeholder
        bar = st.progress(0)
        latest_iteration = st.empty()

        for i in range(100):
            # Update the progress bar with each iteration.
            bar.progress(i + 1)
            latest_iteration.text(f'{i+1}%')
            time.sleep(0.001)

        try:
            # Call API to get prediction
            param = {'urls': urls}
            url = 'https://api-gwwhhqf6zq-uc.a.run.app/'
            prediction = requests.get(url, params=param)
            st.write(prediction.json())
            st.success('This is a success!')
        except:
            # Error happen when calling API
            st.error('Something goes wrong ! Try another url')

        if len(urls_list) == 1:
            # Display a graph if there is only one URL
            pass
            # df = get_dataframe_data()
            # st.table(df.head())
            # c = alt.Chart(df).mark_circle().encode(x='a',
            #                                        y='b',
            #                                        size='c',
            #                                        color='c',
            #                                        tooltip=['a', 'b',
            #                                                 'c']).properties(
            #                                                     width=700,
            #                                                     height=500)
            # st.write(c)

        # Display a Table with the API result
        df = get_dataframe_data()
        df.drop(columns='Text_clean', inplace=True)
        st.table(df.head(10))

import json
import numpy as np


def load_R_model(filename):
    with open(filename, 'r') as j:
        data_input = json.load(j)
    data = {
        'topic_term_dists': data_input['phi'],
        'doc_topic_dists': data_input['theta'],
        'doc_lengths': data_input['doc.length'],
        'vocab': data_input['vocab'],
        'term_frequency': data_input['term.frequency']
    }
    return data


movies_model_data = load_R_model('data/movie_reviews_input.json')

import pyLDAvis

movies_vis_data = pyLDAvis.prepare(**movies_model_data)

html_string = pyLDAvis.prepared_data_to_html(movies_vis_data)
from streamlit import components
components.v1.html(html_string, width=1300, height=900, scrolling=True)
