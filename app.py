from streamlit import components
import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
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
    background-image: url(https://fr.ankorstore.com/images/stripes-background.svg);
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

h3 {
    padding:0;
    font-weight: 600;
}

.table {
    --bs-table-striped-bg:rgb(240, 242, 246)!important;
}

.block-container {
    background-color: #ffffff;
    padding: 1rem 1rem!important;
    margin: 1rem 10rem!important;
    box-shadow: 0 0 20px #0000001f;
    border-radius: 8px;
}

.streamlit-expanderContent {
    padding-left:0!important;
    padding-right:0!important;
}

a {
    text-decoration:none!important;
}

a:hover {
    text-decoration:underline!important;
}

"""

st.set_page_config(
            page_title="Commercial 2.0",
            page_icon="🤖",
            layout="centered")

st.write(f'<style>{CSS}</style>', unsafe_allow_html=True)

st.write("""<!-- CSS Bootstrap only -->
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet" integrity="sha384-1BmE4kWBq78iYhFldvKuhfTAU6auU8tT94WrHftjDbrCEXSU1oBoqyl2QvZ6jIW3" crossorigin="anonymous">
""", unsafe_allow_html=True)

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

def highlight_score(serie, color='#09ab3b33'):
    '''
    highlight the maximum in a Series or DataFrame
    '''
    attr = "<p style='background-color: {}'>".format(color)

    is_max = serie == serie[["Beauty", "Fashion", "Grocery", "Home & Living", "Kids"]].max()

    for i in range(len(serie)):
        if is_max.iloc[i]==True:
            serie.iloc[i] = f'{attr}{serie.iloc[i]}%</p>'

    return serie

# Textarea with urls
urls = st.text_area("Insert one or more url(s) separated by commas:",
                    "https://www.evaliaparis.com/, https://www.yves-rocher.fr/, https://www.uneheurepoursoi.com/, https://www.calzedonia.com/, https://newjerseyparis.com/, https://www.indies.fr/, https://bylouise.fr/, https://www.placedestendances.com/, https://www.kidsaround.com/, https://www.catimini.com/, https://www.melijoe.com/fr, https://www.sergent-major.com/")
urls_list = urls.split(",")
st.write("Number of url(s):", len(urls_list))

# Upload a CSV
st.set_option('deprecation.showfileUploaderEncoding', False)
uploaded_file = st.file_uploader("Or choose a CSV file", type="csv")
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write(data)


if st.button('Launch qualification 🎉'):
    bad_urls = regex_url(urls_list)

    if bad_urls != 0:
        st.warning('Stay positive, some urls are just maybe not well written.')
    else:
        # Add a placeholder
        bar = st.progress(0)
        latest_iteration = st.empty()

        # API URL local for testing / GCP in production
        # url = 'https://api-n75dafaksa-ew.a.run.app/get_data'
        url = 'http://127.0.0.1:8000/get_data'

        try:
            # Call API first time to get prediction
            param = {'urls': f'["{urls_list[0]}"]'}
            prediction = requests.get(url, params=param)
            data = json.loads(prediction.json())
            df = pd.json_normalize(data)

            # Update the progress bar with each iteration.
            progression = int(1/len(urls_list)*100)
            bar.progress(progression)
            latest_iteration.text(f'{progression}%')

            for i in range(1, len(urls_list)):
                # Call API to get prediction
                param = {'urls': f'["{urls_list[i]}"]'}
                prediction = requests.get(url, params=param)
                data = json.loads(prediction.json())
                df = pd.concat([df, pd.json_normalize(data)])

                # Update the progress bar with each iteration.
                progression = int((i+1)/len(urls_list)*100)
                bar.progress(progression)
                latest_iteration.text(f'{progression}%')

            st.success('This is a success!')
            df.drop(columns=['Text', 'Language'], inplace=True)
            df['url'] = df['url'].apply(lambda x: '<a href="{0}" target="_blank">{0}</a>'.format(x))
            df['Email'] = df['Email'].apply(lambda x: '' if (x == 'No email' or x == 'NaN') else ", ".join(['<a href=mailto:"{0}">email</a>'.format(i) for i in x]))
            df['Facebook'] = df['Facebook'].apply(lambda x: '' if (x == 'No Facebook' or x == 'NaN')  else ", ".join(['<a href="{0}" target="_blank">FbLink</a>'.format(i) for i in x]))
            df['Instagram'] = df['Instagram'].apply(lambda x: '' if (x == 'No Instagram' or x == 'NaN')  else ", ".join([f'<a href="{url}" target="_blank">InstaLink{i}</a>' for i, url in enumerate(x,1)]))

            df = df.apply(highlight_score, axis=1)
            df.fillna('', inplace=True)

            st.write(df.to_html(escape=False, index=False, classes=["table", "table-striped"]), unsafe_allow_html=True)
        except:
            # Error happen when calling API
            st.error('Something went wrong ! Try another url')


# @st.cache
st.markdown('# Explain our model')

# @st.cache
with st.expander("French model dataviz"):
    # @st.cache
    model_data_fr = load_R_model('data/dataviz_fr.json')
    vis_data_fr = pyLDAvis.prepare(**model_data_fr)
    html_string_fr = pyLDAvis.prepared_data_to_html(vis_data_fr)
    components.v1.html(html_string_fr, width=1210, height=780, scrolling=False)

# @st.cache
with st.expander("English model dataviz"):
    # @st.cache
    model_data_en = load_R_model('data/dataviz_en.json')
    vis_data_en = pyLDAvis.prepare(**model_data_en)
    html_string_en = pyLDAvis.prepared_data_to_html(vis_data_en)
    components.v1.html(html_string_en, width=1210, height=780, scrolling=False)

st.markdown("""#### Made with ❤️ by""")
col1, col2, col3 = st.columns(3)

col1.write("""<div style='text-align: center;'>
            <a href='https://www.linkedin.com/in/arnaud-vuacheux/' target='_blank'>
                <img src='https://media-exp1.licdn.com/dms/image/C4D03AQFKsogocxpUTA/profile-displayphoto-shrink_800_800/0/1634035549772?e=1643846400&v=beta&t=Gf-YYd1p3rkz8abPd59txkO2dmLMFQOS6jglDZhFTcg' style='width:120px;height:120px;border-radius:50%;margin-bottom: 25px;' />
                <p>Arnaud Vuacheux</p>
            </a>
            </div>""", unsafe_allow_html = True)
col2.write("""<div style='text-align: center;'>
            <a href='https://www.linkedin.com/in/emmanuel-miralles-a2604911a/' target='_blank'>
            <img src='https://media-exp1.licdn.com/dms/image/C4D03AQEpDVzZZTEwyQ/profile-displayphoto-shrink_800_800/0/1616948728275?e=1643846400&v=beta&t=st95nKH408-cpnsav0fbM0YWRHIEXphxxKxS0MowdEo' style='width:120px;height:120px;border-radius:50%;margin-bottom: 25px;' />
            <p>Emmanuel Miralles</p>
            </a>
            </div>""", unsafe_allow_html = True)
col3.write("""<div style='text-align: center;'>
            <a href='https://www.linkedin.com/in/robinbusinessdevelopment/'target='_blank'>
            <img src='https://media-exp1.licdn.com/dms/image/C4E03AQH73LCrsyPnJA/profile-displayphoto-shrink_800_800/0/1636019621502?e=1643846400&v=beta&t=bSTvj5JS8yPRnjET9Y6ybHmS8TXQdLuAo_Tstn2G1lw' style='width:120px;height:120px;border-radius:50%;margin-bottom: 25px;' />
            <p>Robin Lespagnol</p>
            </a>
            </div>""", unsafe_allow_html = True)
