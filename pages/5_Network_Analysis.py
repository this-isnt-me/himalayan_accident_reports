import streamlit as st
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import helper_functions.helper_funcs as helpf
import helper_functions.charts as chartf
import streamlit.components.v1 as components
from pyvis.network import Network

import json

st.set_page_config(
    layout="wide",
    initial_sidebar_state="expanded",
    page_title="Home Page")

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
header_container.header("Analysis of Accidents from the Himalayan Database - Network Analysis")

accident_data = load_data_json(file_path='data/himalayan_accident_data.json')

body_container_one = st.container()

logo_path = 'img/art_deco_w_t.png'
st.sidebar.image(logo_path)
st.sidebar.title('General Filters')
peak_names = st.sidebar.multiselect('Select Which Peaks You\'d Like to Analyse  - Selecting None Will Select All',
                                    placeholder='Choose Peaks',
                                    options=sorted(list(helpf.extract_unique_values(accident_data, 'peak_name'))),
                                    key='selected_peaks')
years_list = st.sidebar.multiselect('Select Which Years You\'d Like to Analyse  - Selecting None Will Select All',
                                    placeholder='Choose Years',
                                    options=sorted(list(helpf.extract_unique_values(accident_data, 'year'))),
                                    key='selected_years')
season_list = st.sidebar.multiselect('Select Which Season You\'d like to Analyse  - Selecting None Will Select All',
                                     placeholder='Choose Seasons',
                                     options=helpf.extract_unique_values(accident_data, 'season'),
                                     key='selected_seasons')
ignore_list = st.sidebar.multiselect('Select Which Classes You Wish to Ignore',
                                     placeholder='Choose Classes',
                                     options=helpf.classification_list(accident_data),
                                     key='selected_classes')
if st.sidebar.button('Apply Filters', use_container_width=True):
    if len(peak_names):
        accident_data = helpf.filter_dicts_by_value(accident_data, 'peak_name', peak_names)
    if len(years_list):
        accident_data = helpf.filter_dicts_by_value(accident_data, 'year', years_list)
    if len(season_list):
        accident_data = helpf.filter_dicts_by_value(accident_data, 'season', season_list)
    if not ignore_list:
        ignore_list = []
    node_weight_list, edge_weight_list = helpf.generate_network_data(accident_data, ignore_list)
    network_graph = helpf.generate_network(node_weight_list, edge_weight_list)
    algo_dataframe, communities = helpf.centrality_detection(network_graph)
    body_container_one.markdown('### Top 10 Most Influential Nodes by Algorithm')
    body_container_one.dataframe(algo_dataframe,
                                 hide_index=True,
                                 use_container_width=True)
    body_container_one.markdown('###')
    body_container_one.markdown(f'### Top {len(communities)} Node Sub Groups')
    for index, community in enumerate(communities):
        body_container_one.markdown(f'{index + 1}) {community}')
    body_container_one.markdown('###')

    graph_net = Network(height='1000px',
                        bgcolor='#222222',
                        font_color='white')
                        # ,
                        # heading='Network Graph')
    graph_net.from_nx(network_graph)
    graph_net.repulsion(node_distance=750, central_gravity=0.1,
                        spring_length=250, spring_strength=0.10,
                        damping=0.95)
    try:
        path = 'tmp'
        graph_net.save_graph(f'{path}/pyvis_graph.html')
        HtmlFile = open(f'{path}/pyvis_graph.html', 'r', encoding='utf-8')
        # Save and read graph as HTML file (locally)
    except:
        path = 'html_files'
        graph_net.save_graph(f'{path}/pyvis_graph.html')
        HtmlFile = open(f'{path}/pyvis_graph.html', 'r', encoding='utf-8')
    with body_container_one.expander("Expand To View Network Graph"):
        components.html(HtmlFile.read(), height=1000)
