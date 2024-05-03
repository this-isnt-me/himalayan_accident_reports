import streamlit as st
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import helper_functions.helper_funcs as helpf
import helper_functions.charts as chartf

st.set_page_config(
    layout="wide",
    initial_sidebar_state="expanded",
    page_title="Word Clouds")

@st.cache_data
def load_data_json(file_path):
    loaded_data = helpf.load_in_json(file_path)
    return loaded_data

@st.cache_data
def load_data_dataframe(file_path, drop_list=None):
    loaded_data = helpf.load_in_json(file_path)
    df = pd.DataFrame(loaded_data)
    if drop_list:
        df = df.drop(columns=drop_list)
    return df

injuries_list = ['respiratory problems', 'people falling', 'extreme cold', 'frostbite', 'exhaustion', 'broken bones',
                 'hypothermia', 'avalanches', 'high-altitude illness', 'icefall', 'stomach problems', 'rockfall',
                 'snowfall', 'inadequate preparation', 'glacial crevasses', 'steep rock',
                 'equipment failure', 'mental stress', 'getting lost', 'running out of resources', 'blood clot',
                 'heart problems', 'weather-related issues']


header_container = st.container()
header_container.header("Analysis of Accidents from the Himalayan Database - Word Clouds")

main_data_frame = load_data_dataframe(file_path='data/himalayan_accidents.json',
                                      drop_list=["tag_list", "countries", "NOUN_pos", "VERB_pos", "PERSON_ner",
                                                 "NORP_ner", "FAC_ner", "ORG_ner", "GPE_ner", "LOC_ner", "PRODUCT_ner",
                                                 "EVENT_ner", "DATE_ner", "TIME_ner", "accidents_embedding",
                                                 "2d_umap", "3d_umap", "bert_key_words"])

logo_path = 'img/art_deco_w_t.png'
st.sidebar.image(logo_path)
st.sidebar.title('Filters')
peak_names = st.sidebar.multiselect('Select Which Peaks You\'d Like to Analyse  - Selecting None Will Select All',
                                    placeholder='Choose Peaks',
                                    options=sorted(list(set(main_data_frame['peak_name']))),
                                    key='selected_peaks')
years_list = st.sidebar.multiselect('Select Which Years You\'d Like to Analyse  - Selecting None Will Select All',
                                    placeholder='Choose Years',
                                    options=sorted(list(set(main_data_frame['year']))),
                                    key='selected_years')
classification_list = st.sidebar.multiselect('Select Which Specific Accident Report Classifications You\'d like to '
                                             'Analyse  - Selecting None Will Select All',
                                             placeholder='Choose Classifications',
                                             options=sorted(injuries_list),
                                             key='selected_classes')

sb_col1a, sb_col1b = st.sidebar.columns([1, 1])
if sb_col1a.button('Apply Filters', use_container_width=True):
    if len(peak_names):
        main_data_frame = main_data_frame[main_data_frame['peak_name'].isin(peak_names)]
    if len(years_list):
        main_data_frame = main_data_frame[main_data_frame['year'].isin(years_list)]
    if len(classification_list):
        main_data_frame = main_data_frame[main_data_frame[classification_list].any(axis=1)]
if sb_col1b.button('Reset Data', use_container_width=True):
    main_data_frame = load_data_dataframe(file_path='data/himalayan_accidents.json',
                                          drop_list=["tag_list", "countries", "NOUN_pos", "VERB_pos", "PERSON_ner",
                                                     "NORP_ner", "FAC_ner", "ORG_ner", "GPE_ner", "LOC_ner",
                                                     "PRODUCT_ner",
                                                     "EVENT_ner", "DATE_ner", "TIME_ner", "accidents_embedding",
                                                     "2d_umap", "3d_umap", "bert_key_words"])
body_container_five = st.container()

count_fig, count_ax = plt.subplots(figsize=(9, 6))
count_freq = helpf.generate_vectoriser_cloud(text_list=list(main_data_frame['accidents']),
                                             vec_type='count')
count_wordcloud = WordCloud(background_color='black').generate_from_frequencies(count_freq)
count_ax.imshow(count_wordcloud, interpolation='bilinear')
count_ax.axis('off')

tfidf_fig, tfidf_ax = plt.subplots(figsize=(9, 6))
tfidf_freq = helpf.generate_vectoriser_cloud(text_list=list(main_data_frame['accidents']),
                                             vec_type='tfidf')
tfidf_wordcloud = WordCloud(background_color='black').generate_from_frequencies(tfidf_freq)
tfidf_ax.imshow(tfidf_wordcloud, interpolation='bilinear')
tfidf_ax.axis('off')

t_three_c1_1, t_three_c1_2 = body_container_five.tabs(["Straight Count", "Weighted Count"])
five_c1a, five_c2a, five_c3a = t_three_c1_1.columns([2, 8, 2])
five_c2a.markdown('## Straight Count Word Frequency')
five_c2a.markdown('###')
five_c2a.pyplot(count_fig, use_container_width=True)
five_c1b, five_c2b, five_c3b = t_three_c1_2.columns([2, 8, 2])
five_c2b.markdown('## Weighted(TFIDF) Count Word Frequency')
five_c2b.markdown('###')
five_c2b.pyplot(tfidf_fig, use_container_width =True)
