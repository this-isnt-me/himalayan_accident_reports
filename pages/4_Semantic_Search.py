import streamlit as st
import pandas as pd
import helper_functions.helper_funcs as helpf

st.set_page_config(
    layout="wide",
    initial_sidebar_state="expanded",
    page_title="Semantic Search")


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

if "sem_search_result" not in st.session_state:
    st.session_state["sem_search_result"] = []

injuries_list = ['respiratory problems', 'people falling', 'extreme cold', 'frostbite', 'exhaustion', 'broken bones',
                 'hypothermia', 'avalanches', 'high-altitude illness', 'icefall', 'stomach problems', 'rockfall',
                 'snowfall', 'inadequate preparation', 'glacial crevasses', 'steep rock',
                 'equipment failure', 'mental stress', 'getting lost', 'running out of resources', 'blood clot',
                 'heart problems', 'weather-related issues']

header_container = st.container()
header_container.header("Analysis of Accidents from the Himalayan Database - Semantic Search")

accident_data = load_data_json(file_path='data/himalayan_accident_data.json')

body_container = st.container()

logo_path = 'img/art_deco_w_t.png'
st.sidebar.image(logo_path)
st.sidebar.title('Filters')
query_string = st.sidebar.text_area('What Do You Want Do Know About?', 'Query')
doc_count = st.sidebar.selectbox(
    "Select The Number of Reports You Would Like To See",
    list(range(5, 21)),
    index=None,
    placeholder="Select The Number of Reports",
)
st.sidebar.markdown('###')
if st.sidebar.button('Search', use_container_width=True):
    if not doc_count:
        doc_count = 10
    query_string = helpf.clean_string(input_text=query_string)
    query_embedding = helpf.generate_embedding(input_text=query_string)
    similar_docs = helpf.find_most_similar(input_vector=query_embedding[0],
                                           data=accident_data,
                                           embedding_key="accidents_embedding",
                                           topx=doc_count)
sem_search_results = st.session_state["sem_search_result"]
if sem_search_results and len(sem_search_results) > 0:
    body_container.markdown('---')
    colh_1, colh_2, colh_3, colh_4, colh_5, colh_6 = body_container.columns([1, 1, 6, 1, 1, 1])
    colh_1.markdown('##### Expedition ID')
    colh_2.markdown('##### Accident ID')
    colh_3.markdown('##### Accident Report')
    colh_4.markdown('##### Report Classifications')
    colh_5.markdown('##### Similarity Score')
    colh_6.markdown('')
    body_container.markdown('---')
    for entry in sem_search_results:
        col_1, col_2, col_3, col_4, col_5, col_6 = body_container.columns([1, 1, 6, 1, 1, 1])
        class_string = helpf.concat_tuples_list(entry['data']['tag_list'])
        col_1.markdown(entry['data']['expidition_id'])
        col_2.markdown(entry['data']['accident_id'])
        accident_string = entry['data']['accidents']
        element_list = [(entry['data']['PERSON_ner'], "PER", "#4169e1"),
                        (entry['data']['NORP_ner'], "NORP", "#dc143c"),
                        (entry['data']['FAC_ner'], "FAC", "#20c997"),
                        (entry['data']['ORG_ner'], "ORG", "#228b22"),
                        (entry['data']['GPE_ner'], "GPE", "#6f42c1"),
                        (entry['data']['LOC_ner'], "LOC", "#fd7e14"),
                        (entry['data']['PRODUCT_ner'], "PROD", "#800000"),
                        (entry['data']['EVENT_ner'], "EVE", "#000080"),
                        (entry['data']['DATE_ner'], "DATE", "#d2691e"),
                        (entry['data']['TIME_ner'], "TIME", "#ffd700")
                        ]
        for element in element_list:
            if len(element[0]) < 0:
                continue
            for ent in element[0]:
                accident_string = accident_string.replace(ent, f'<span style="background:{element[2]};'
                                                               f'border-radius: 5px;"> <strong>{ent}</strong>'
                                                               f'({element[1]}) </span>')
        col_3.markdown(f'<span style="font-size: 20px;">{accident_string}</span>', unsafe_allow_html=True)
        col_4.markdown(class_string)
        col_5.markdown(f'{(entry["score"] * 100):.2f}%')
        col_6.button('More Like This',
                     key=entry['data']['expidition_id'] + entry['data']['accident_id'],
                     on_click=helpf.find_most_similar,
                     args=[entry['data']['accidents_embedding'],
                           accident_data,
                           "accidents_embedding",
                           doc_count],
                     use_container_width=True)
        body_container.markdown('---')
