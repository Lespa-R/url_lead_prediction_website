import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
import requests
import re
import time

logo_style = """
img {
    width:150px!important;
    }
"""

lewagon = Image.open('images/lewagon.png')
ankorstore = Image.open('images/ankorstore.png')
st.write(f'<style>{logo_style}</style>', unsafe_allow_html=True)

columns = st.columns(2)

logo1 = columns[0].image(lewagon, use_column_width=True)
logo2 = columns[1].image(ankorstore, use_column_width=True)

'''
# Commercial 2.0
## La qualification prédictive à partir d'un url
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

@st.cache
def get_dataframe_data():
    return pd.DataFrame(
            np.random.randn(10, 5),
            columns=('col %d' % i for i in range(5))
        )

urls = st.text_area("Inserez un ou plusieurs url(s) séparé(s) par des virgules :",
                    "www.lewagon.com, (...)")
urls_list = urls.split(",")
st.write("Nombre d'url(s):", len(urls_list))

if st.button('Qualification 🎉'):
    # print is visible in the server output, not in the page

    bad_urls = regex_url(urls_list)

    if bad_urls != 0:
        st.warning('Reste positif, certains urls ne sont peut-être pas biens écrit.')
    else:
        'Starting a long computation...'

        # Add a placeholder
        latest_iteration = st.empty()
        bar = st.progress(0)

        for i in range(100):
            # Update the progress bar with each iteration.
            latest_iteration.text(f'Iteration {i+1}')
            bar.progress(i + 1)
            time.sleep(0.1)

        '...and now we\'re done! 🎉'

        try:
            param = {'urls': urls}
            url = 'https://api-gwwhhqf6zq-uc.a.run.app/'
            prediction = requests.get(url, params=param)
            st.write(prediction.json())
            st.success('This is a success!')
        except:
            st.error('Let\'s keep positive, this might be pretty close to a success!')

        df = get_dataframe_data()
        st.table(df.head())

else:
    pass
    # st.write('I was not clicked 😞')
