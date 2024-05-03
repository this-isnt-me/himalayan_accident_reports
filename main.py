import streamlit as st
import pandas as pd
import helper_functions.helper_funcs as helpf
import helper_functions.charts as chartf

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
    df[df.columns] = df[df.columns].applymap(lambda x: x.strip() if isinstance(x, str) else x)
    return df


injuries_list = ['respiratory problems', 'people falling', 'extreme cold', 'frostbite', 'exhaustion', 'broken bones',
                 'hypothermia', 'avalanches', 'high-altitude illness', 'icefall', 'stomach problems', 'rockfall',
                 'snowfall', 'inadequate preparation', 'glacial crevasses', 'steep rock',
                 'equipment failure', 'mental stress', 'getting lost', 'running out of resources', 'blood clot',
                 'heart problems', 'weather-related issues']


header_container = st.container()
header_container.header("Analysis of Accidents from the Himalayan Database")

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

# header_container.dataframe(main_data_frame,
#                            hide_index=True,
#                            use_container_width=True)

body_container = st.container()

grouped_bar_chart = chartf.create_grouped_barchart(input_df=main_data_frame,
                                                   column_one='peak_name',
                                                   column_two='termreason')

melted_df = helpf.melt_dataframe(main_data_frame, injuries_list)
grouped_bar_chart2 = chartf.create_grouped_barchart(input_df=melted_df,
                                                    column_one='peak_name',
                                                    column_two='Class')
grouped_bar_chart3 = chartf.create_grouped_barchart(input_df=melted_df,
                                                    column_one='termreason',
                                                    column_two='Class')

term_bar_chart = chartf.create_barchart(input_df=main_data_frame, column_one='termreason')




report_bar_chart = chartf.create_injury_barchart(helpf.create_sub_df(main_data_frame, injuries_list))
melted_df = helpf.melt_dataframe(main_data_frame, injuries_list)
stacked_fig = chartf.create_stacked_line_chart(melted_df, values_filter=classification_list)

melted_df = helpf.melt_dataframe(main_data_frame[main_data_frame['season'] == 'Spring'], injuries_list)
stacked_fig_sp = chartf.create_stacked_line_chart(melted_df, values_filter=classification_list)
melted_df = helpf.melt_dataframe(main_data_frame[main_data_frame['season'] == 'Summer'], injuries_list)
stacked_fig_su = chartf.create_stacked_line_chart(melted_df, values_filter=classification_list)
melted_df = helpf.melt_dataframe(main_data_frame[main_data_frame['season'] == 'Autumn'], injuries_list)
stacked_fig_au = chartf.create_stacked_line_chart(melted_df, values_filter=classification_list)
melted_df = helpf.melt_dataframe(main_data_frame[main_data_frame['season'] == 'Winter'], injuries_list)
stacked_fig_wi = chartf.create_stacked_line_chart(melted_df, values_filter=classification_list)

t_four_c1_1, t_four_c1_2, t_four_c1_3, t_four_c1_4 = body_container.tabs(
    [grouped_bar_chart['layout']['title']['text'],
     term_bar_chart['layout']['title']['text'],
     report_bar_chart['layout']['title']['text'],
     stacked_fig['layout']['title']['text']]
)
t_four_c1_1.plotly_chart(grouped_bar_chart, use_container_width=True)
t_four_c1_1.markdown('###')
t_four_c1_1.plotly_chart(grouped_bar_chart2, use_container_width=True)
t_four_c1_1.markdown('###')
t_four_c1_1.plotly_chart(grouped_bar_chart3, use_container_width=True)
t_four_c1_2.plotly_chart(term_bar_chart, use_container_width=True)
t_four_c1_3.plotly_chart(report_bar_chart,use_container_width=True)
t_four_c1_4.markdown('Whole Year')
t_four_c1_4.plotly_chart(stacked_fig, use_container_width=True)
t_four_c1_4.markdown('Spring')
t_four_c1_4.plotly_chart(stacked_fig_sp, use_container_width=True)
t_four_c1_4.markdown('Summer')
t_four_c1_4.plotly_chart(stacked_fig_su, use_container_width=True)
t_four_c1_4.markdown('Autumn')
t_four_c1_4.plotly_chart(stacked_fig_au, use_container_width=True)
t_four_c1_4.markdown('Winter')
t_four_c1_4.plotly_chart(stacked_fig_wi, use_container_width=True)
body_container.markdown('###')
body_container.markdown('---')


file_path = 'data/himalayan_accident_data.json'  # Replace with your JSON file path
with open(file_path, 'r') as file:
    data = json.load(file)

# Loop through the dictionaries and update the 'season' attribute
for entry in data:
    if entry.get('season') == 'Aumtum':
        entry['season'] = 'Autumn'

# Save the updated list of dictionaries back to the same file
with open(file_path, 'w') as file:
    json.dump(data, file)
