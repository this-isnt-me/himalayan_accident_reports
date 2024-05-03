import streamlit as st
import pandas as pd
import helper_functions.helper_funcs as helpf
import helper_functions.charts as chartf

st.set_page_config(
    layout="wide",
    initial_sidebar_state="expanded",
    page_title="Text Clustering")


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
header_container.header("Analysis of Accidents from the Himalayan Database - Text Clustering")

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
st.sidebar.markdown('---')
st.sidebar.title('Text Clustering Filters')
st.sidebar.subheader('LDA Filters')
lda_clusters = st.sidebar.selectbox(
    "Select The Number of LDA Clusters You Would Like To See",
    list(range(1, 21)),
    index=None,
    placeholder="Select The Number of Clusters",
)
lda_key_words = st.sidebar.selectbox(
    "Select The Number of LDA Key Words You Would Like To See",
    list(range(5, 16)),
    index=None,
    placeholder="Select The Number of Key Words",
)
st.sidebar.markdown('---')
st.sidebar.subheader('NMF Filters')
nmf_clusters = st.sidebar.selectbox(
    "Select The Number of NMF Clusters You Would Like To See",
    list(range(1, 21)),
    index=None,
    placeholder="Select The Number of Clusters",
)
nmf_key_words = st.sidebar.selectbox(
    "Select The Number of NMF Key Words You Would Like To See",
    list(range(5, 16)),
    index=None,
    placeholder="Select The Number of Key Words",
)

sb_col1a, sb_col1b = st.sidebar.columns([1, 1])
if sb_col1a.button('Apply Filters', use_container_width=True):
    if len(peak_names):
        main_data_frame = main_data_frame[main_data_frame['peak_name'].isin(peak_names)]
    if len(years_list):
        main_data_frame = main_data_frame[main_data_frame['year'].isin(years_list)]
    if len(classification_list):
        main_data_frame = main_data_frame[main_data_frame[classification_list].any(axis=1)]
    if lda_clusters and lda_key_words:
        main_data_frame = helpf.perform_clustering_analysis(input_df=main_data_frame,
                                                            text_column='accidents',
                                                            new_column_name='lda_cluster',
                                                            new_key_column='lda_cluster_key_words',
                                                            algorithm='lda',
                                                            num_topics=lda_clusters, num_top_words=lda_key_words)
    if nmf_clusters and nmf_key_words:
        main_data_frame = helpf.perform_clustering_analysis(input_df=main_data_frame,
                                                            text_column='accidents',
                                                            new_column_name='nmf_cluster',
                                                            new_key_column='nmf_cluster_key_words',
                                                            algorithm='nmf',
                                                            num_topics=nmf_clusters, num_top_words=nmf_key_words)
if sb_col1b.button('Reset Data', use_container_width=True):
    main_data_frame = load_data_dataframe(file_path='data/himalayan_accidentss.json',
                                          drop_list=["tag_list", "countries", "NOUN_pos", "VERB_pos", "PERSON_ner",
                                                     "NORP_ner", "FAC_ner", "ORG_ner", "GPE_ner", "LOC_ner",
                                                     "PRODUCT_ner",
                                                     "EVENT_ner", "DATE_ner", "TIME_ner", "accidents_embedding",
                                                     "2d_umap", "3d_umap", "bert_key_words"])

main_data_frame['nmf_cluster'] = main_data_frame['nmf_cluster'].apply(lambda x: x + 1 if x >= 0 else x)
main_data_frame['lda_cluster'] = main_data_frame['lda_cluster'].apply(lambda x: x + 1 if x >= 0 else x)
main_data_frame['bert_cluster'] = main_data_frame['bert_cluster'].apply(lambda x: x + 1 if x >= 0 else x)
body_container_one = st.container()
expander_one = body_container_one.expander('NMF Clusters Plotted', expanded=False)
body_container_two = st.container()
expander_two = body_container_two.expander('LDA Clusters Plotted', expanded=False)
body_container_three = st.container()
expander_three = body_container_two.expander('Bert Clusters Plotted', expanded=False)

chart_nmf_3d = chartf.create_scatter_chart(input_df=main_data_frame,
                                           colour_col='nmf_cluster',
                                           x_col='3d_umap_x',
                                           y_col='3d_umap_y',
                                           z_col='3d_umap_z')
chart_nmf_2d = chartf.create_scatter_chart(input_df=main_data_frame,
                                           colour_col='nmf_cluster',
                                           x_col='2d_umap_x',
                                           y_col='2d_umap_y')
one_c1, one_c2, one_c3 = expander_one.columns([24, 1, 8])
t_one_c1_1, t_one_c1_2 = one_c1.tabs(["2D Space", "3D Space"])
t_one_c1_1.markdown('## Plotting NMF Clusters in 2d Space')
t_one_c1_1.plotly_chart(chart_nmf_2d, use_container_width=True)
t_one_c1_2.markdown('## Plotting NMF Clusters in 3d Space')
t_one_c1_2.plotly_chart(chart_nmf_3d, use_container_width=True)

nmf_cluster_list = helpf.get_cluster_descriptions(input_df=main_data_frame,
                                                  sort_on_col='nmf_cluster',
                                                  subset_columns_list=['nmf_cluster',
                                                                       'nmf_cluster_key_words'])
t_one_c3_1 = one_c3.tabs(["Top Keywords in Each Cluster"])
for text in nmf_cluster_list:
    t_one_c3_1[0].markdown(text)
    # t_one_c3_1[0].markdown('#####')

chart_lda_3d = chartf.create_scatter_chart(input_df=main_data_frame,
                                           colour_col='lda_cluster',
                                           x_col='3d_umap_x',
                                           y_col='3d_umap_y',
                                           z_col='3d_umap_z')
chart_lda_2d = chartf.create_scatter_chart(input_df=main_data_frame,
                                           colour_col='lda_cluster',
                                           x_col='2d_umap_x',
                                           y_col='2d_umap_y')
two_c1, two_c2, two_c3 = expander_two.columns([24, 1, 8])
t_two_c1_1, t_two_c1_2 = two_c1.tabs(["2D Space", "3D Space"])
t_two_c1_1.markdown('## Plotting LDA Clusters in 2d Space')
t_two_c1_1.plotly_chart(chart_lda_2d, use_container_width=True)
t_two_c1_2.markdown('## Plotting LDA Clusters in 3d Space')
t_two_c1_2.plotly_chart(chart_lda_3d, use_container_width=True)

lda_cluster_list = helpf.get_cluster_descriptions(input_df=main_data_frame,
                                                  sort_on_col='lda_cluster',
                                                  subset_columns_list=['lda_cluster',
                                                                       'lda_cluster_key_words'])
t_two_c3_1 = two_c3.tabs(["Top Keywords in Each Cluster"])
for text in lda_cluster_list:
    t_two_c3_1[0].markdown(text)
    # t_two_c3_1[0].markdown('#####')

chart_bert_3d = chartf.create_scatter_chart(input_df=main_data_frame,
                                            colour_col='bert_cluster',
                                            x_col='3d_umap_x',
                                            y_col='3d_umap_y',
                                            z_col='3d_umap_z')
chart_bert_2d = chartf.create_scatter_chart(input_df=main_data_frame,
                                            colour_col='bert_cluster',
                                            x_col='2d_umap_x',
                                            y_col='2d_umap_y')
three_c1, three_c2, three_c3 = expander_three.columns([24, 1, 8])
t_three_c1_1, t_three_c1_2 = three_c1.tabs(["2D Space", "3D Space"])
t_three_c1_1.markdown('## Plotting Bert Clusters in 2d Space')
t_three_c1_1.plotly_chart(chart_bert_2d, use_container_width=True)
t_three_c1_2.markdown('## Plotting Bert Clusters in 3d Space')
t_three_c1_2.plotly_chart(chart_bert_3d, use_container_width=True)

bert_cluster_list = helpf.get_cluster_descriptions(input_df=main_data_frame,
                                                   sort_on_col='bert_cluster',
                                                   subset_columns_list=['bert_cluster',
                                                                        'bert_cluster_keywords'])
t_three_c3_1 = three_c3.tabs(["Top Keywords in Each Cluster"])
for text in bert_cluster_list:
    t_three_c3_1[0].markdown(text)
body_container_three.markdown('---')
body_container_three.markdown('###')
