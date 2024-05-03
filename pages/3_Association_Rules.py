import streamlit as st
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import helper_functions.helper_funcs as helpf
import helper_functions.charts as chartf

st.set_page_config(
    layout="wide",
    initial_sidebar_state="expanded",
    page_title="Association Rules")

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
header_container.header("Analysis of Accidents from the Himalayan Database - Association Rules")

main_data_frame = load_data_dataframe(file_path='data/himalayan_accidents.json',
                                      drop_list=["tag_list", "countries", "NOUN_pos", "VERB_pos", "PERSON_ner",
                                                 "NORP_ner", "FAC_ner", "ORG_ner", "GPE_ner", "LOC_ner", "PRODUCT_ner",
                                                 "EVENT_ner", "DATE_ner", "TIME_ner", "accidents_embedding",
                                                 "2d_umap", "3d_umap", "bert_key_words"])

body_container_six = st.container()

logo_path = 'img/art_deco_w_t.png'
st.sidebar.image(logo_path)
st.sidebar.title('General Filters')
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
st.sidebar.title('Association Rules Filters')
min_support = st.sidebar.selectbox(
    "What is the Minimum Support Level",
    ("1%", "5%", "10%", "15%", "20%", "25%", "30%", "35%", "40%", "45%", "50%"),
    index=None,
    placeholder="Select Minimum Support %"
)
min_confidence = st.sidebar.selectbox(
    "What is the Minimum Confidence Level",
    ("100%", "90%", "80%", "70%", "60%", "50%", "40%", "30%", "20%", "10%"),
    index=None,
    placeholder="Select Confidence Support %"
)

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
body_container_six.markdown('### Term Definitions')
body_container_six.markdown('Support : Indicates percentage of reports that are classified as being a combination of '
                            'the antecedents and consequents')
body_container_six.markdown('Confidence : Indicates percentage of reports that are classified as being the consequents '
                            'if they are already defined as being defined as the antecedents')
body_container_six.markdown('Lift : Tells us how much more likely a report is to be classified as consequents if the '
                            'report is already classified as the antecedents')
body_container_six.markdown('###')

if not min_support:
    min_support = '10%'

if not min_confidence:
    min_confidence = '50%'

min_support = int(min_support.replace('%', '').strip())
min_confidence = int(min_confidence.replace('%', '').strip())

rules_df = helpf.market_basket_analysis(input_df=helpf.create_sub_df(main_data_frame, injuries_list),
                                        min_support=min_support/100, metric="lift", min_threshold=1,
                                        min_confidence=min_confidence)
body_container_six.markdown(f'## Association Rules for Report Classification - {rules_df.shape[0]} Rules Identified')
body_container_six.markdown(f'Min Support - {min_support}%  Min Confidence - {min_confidence}%')
body_container_six.markdown('###')

body_container_six.dataframe(rules_df,
                             hide_index=True,
                             use_container_width=True)
body_container_six.markdown('###')

one_hot_encoded_df = pd.get_dummies(main_data_frame['peak_name'])
one_hot_encoded_df = one_hot_encoded_df.astype(bool)
one_hot_encoded_df[main_data_frame.columns] = main_data_frame
one_hot_encoded_df = one_hot_encoded_df.drop(columns=['peak_name'])
rules_df = helpf.market_basket_analysis(input_df=helpf.create_sub_df(one_hot_encoded_df,
                                                                     injuries_list + main_data_frame['peak_name'].unique().tolist()
                                                                     ),
                                        min_support=min_support/100, metric="lift", min_threshold=1,
                                        min_confidence=min_confidence)
body_container_six.markdown(f'## Association Rules for Report Classification And Peak Name '
                            f'- {rules_df.shape[0]} Rules Identified')
body_container_six.markdown(f'Min Support - {min_support}%  Min Confidence - {min_confidence}%')
body_container_six.markdown('###')
body_container_six.dataframe(rules_df,
                             hide_index=True,
                             use_container_width=True)
#
# body_container_six.dataframe(rules_df,
#                              hide_index=True,
#                              use_container_width=True)
# body_container_six.markdown('###')