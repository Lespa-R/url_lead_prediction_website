import streamlit as st
from PIL import Image
import pandas as pd
import requests
import re
import time
import json
import pyLDAvis

CSS = """
.reportview-container .main .block-container{
        max-width: 1226px;
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
    margin:1rem 10rem!important;
}

"""

st.set_page_config(
            page_title="Commercial 2.0",
            page_icon="🤖",
            layout="centered")

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
    """Regex for url matching, take an urls list and return number of wrong urls"""
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
    """Load file small_cleaned_data.csv for testing"""
    return pd.read_csv('data/small_cleaned_data.csv', sep=";")

def load_R_model(filename):
    """Load json file for testing will be provide by API in the futur"""
    with open(filename, 'r') as j:
        data_input = json.load(j)
    data = {
        'topic_term_dists': data_input['ratio_cat_words'], #phi
        'doc_topic_dists': data_input['ratio_url_cat'], #theta
        'doc_lengths': data_input['doc_length_url'], #doc.length
        'vocab': data_input['words'], #vocab
        'term_frequency': data_input['words_frequency'] #term.frequency
    }
    return data

urls = st.text_area("Insert one or more url(s) separated by commas:",
                    "https://www.evaliaparis.com/, https://www.yves-rocher.fr/, https://www.uneheurepoursoi.com/, https://www.calzedonia.com/, https://newjerseyparis.com/, https://www.indies.fr/, https://bylouise.fr/, https://www.placedestendances.com/, https://www.kidsaround.com/, https://www.catimini.com/, https://www.melijoe.com/fr, https://www.sergent-major.com/")
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
            param = {'urls': f'{urls_list}'}
            url = 'http://127.0.0.1:8000/get_data'
            prediction = requests.get(url, params=param)
            st.success('This is a success!')

            data = json.loads(prediction.json())
            df = pd.json_normalize(data)
            df.drop(columns='Text', inplace=True)
            df['url'] = df['url'].apply(lambda x: '<a href="{0}" target="_blank">{0}</a>'.format(x))
            df['Email'] = df['Email'].apply(lambda x: ['<a href=mailto:"{0}">{0}</a>'.format(i) for i in x] if x != 'No email' else x)
            df['Facebook'] = df['Facebook'].apply(lambda x: ['<a href="{0}" target="_blank">{0}</a>'.format(i) for i in x] if x != 'No Facebook' else x)
            df['Instagram'] = df['Instagram'].apply(lambda x: ['<a href="{0}" target="_blank">{0}</a>'.format(i) for i in x] if x != 'No Instagram' else x)
            st.write(df.to_html(escape=False, index=False), unsafe_allow_html=True)

        except:
            # Error happen when calling API
            st.error('Something goes wrong ! Try another url')

        if len(urls_list) == 1:
            # Display a graph if there is only one URL
            pass

        # Display dataviz
        movies_model_data = load_R_model('data/categories.json')
        movies_vis_data = pyLDAvis.prepare(**movies_model_data)
        html_string = pyLDAvis.prepared_data_to_html(movies_vis_data)
        from streamlit import components
        components.v1.html(html_string, width=1210, height=780, scrolling=False)
