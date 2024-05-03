import requests
import time
import json
import streamlit as st
import numpy as np
import pandas as pd
import unicodedata
import networkx as nx
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.feature_extraction import text
from sklearn.metrics.pairwise import cosine_similarity
from mlxtend.frequent_patterns import apriori, association_rules
from collections import defaultdict
import helper_functions.charts as chartf
import plotly.express as px


def create_sub_df(input_df, keep_columns_list):
    return input_df[keep_columns_list]


def generate_vectoriser_cloud(text_list,
                              vec_type):
    if not vec_type:
        return None
    elif vec_type.lower() == 'tfidf':
        vectorizer = TfidfVectorizer(stop_words=list(text.ENGLISH_STOP_WORDS))
        # path = 'img/tfidf_wordcloud.png'
    elif vec_type.lower() == 'count':
        vectorizer = CountVectorizer(stop_words=list(text.ENGLISH_STOP_WORDS))
        # path = 'img/count_wordcloud.png'
    matrix = vectorizer.fit_transform(text_list)
    feature_names = vectorizer.get_feature_names_out()
    count_sum = matrix.sum(axis=0)
    freq = dict(zip(feature_names, count_sum.tolist()[0]))
    freq = dict(sorted(freq.items(), key=lambda item: item[1], reverse=True)[:50])
    return freq


def generate_embedding(input_text=None,
                       model_name='multi-qa-distilbert-cos-v1',
                       api_key='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'):
    if isinstance(input_text, str):
        input_list = [input_text]
    elif isinstance(input_text, list):
        input_list = input_text
    else:
        return None
    if input_text:
        api_url = f'https://api-inference.huggingface.co/pipeline/feature-extraction/' \
                  f'sentence-transformers/{model_name}'
        headers = {'Authorization': f'Bearer {api_key}'}
        payload = {
            "inputs": input_list
        }
        retry = True
        count = 0
        while retry and count < 5:
            response = requests.post(api_url, headers=headers, json=payload)
            response = response.json()
            if isinstance(response, dict) and response.get('error', None):
                wait_time = response.get('estimated_time', 20)
                time.sleep(wait_time)
                count += 1
            elif isinstance(response, list):
                retry = False
                return response
    return None


def find_most_similar(input_vector, data, embedding_key, topx=10):
    # Extract embeddings from data
    embeddings = np.array([entry[embedding_key] for entry in data])
    # Compute cosine similarity with all embeddings
    similarities = cosine_similarity([input_vector], embeddings)[0]
    # Create list of (entry, similarity) pairs
    similarity_pairs = [(entry, similarity) for entry, similarity in zip(data, similarities)]
    # Sort the similarity pairs in descending order
    similarity_pairs.sort(key=lambda x: x[1], reverse=True)
    # Filter out the exact same vectors and select the top 10 most similar
    top_similarities = [sim for sim in similarity_pairs if sim[1] < 1][:topx]
    # Extract the data from dictionaries and add a 'score' attribute
    # results = [{'data': entry[0], 'score': entry[1]} for entry in top_similarities]
    st.session_state["sem_search_result"] = [{'data': entry[0], 'score': entry[1]} for entry in top_similarities]
    # return results


def perform_clustering_analysis(input_df=None,
                                text_column=None,
                                new_column_name='cluster',
                                new_key_column='cluster_key_words',
                                algorithm='nmf',
                                num_topics=10, num_top_words=10):
    if input_df is None or text_column is None:
        return None
    # Extract text data
    texts = input_df[text_column].tolist()

    if algorithm.lower() == 'nmf':
        vectorizer = TfidfVectorizer(stop_words=list(text.ENGLISH_STOP_WORDS))
        matrix = vectorizer.fit_transform(texts)

        # Perform LDA analysis
        model = NMF(n_components=num_topics, random_state=42)
        model.fit(matrix)
    elif algorithm.lower() == 'lda':
        # Convert text data into document-term matrix
        vectorizer = CountVectorizer(stop_words=list(text.ENGLISH_STOP_WORDS))
        matrix = vectorizer.fit_transform(texts)

        # Perform LDA analysis
        model = LatentDirichletAllocation(n_components=num_topics, random_state=42)
        model.fit(matrix)
    # Assign topics to each text
    clusters = model.transform(matrix).argmax(axis=1)
    input_df[new_column_name] = clusters

    # Get top words for each cluster
    feature_names = vectorizer.get_feature_names_out()
    cluster_top_words = {}
    for i, topic in enumerate(model.components_):
        top_words_idx = topic.argsort()[:-num_top_words - 1:-1]
        cluster_top_words[i] = '; '.join([feature_names[idx] for idx in top_words_idx])

    # Add cluster key words to the DataFrame
    input_df[new_key_column] = input_df[new_column_name].map(cluster_top_words)
    return input_df


def load_in_json(file_path):
    with open(file_path, 'r') as json_file:
        data_list = json.load(json_file)
    return data_list


def drop_duplicates(input_df,
                    subset_columns_list):
    return input_df.drop_duplicates(subset=subset_columns_list)


def get_cluster_descriptions(input_df,
                             sort_on_col,
                             subset_columns_list):
    input_df = drop_duplicates(input_df,
                               subset_columns_list)
    input_df = input_df.sort_values(by=sort_on_col)
    input_df[sort_on_col] = 'cluster ' + input_df[sort_on_col].astype(str)
    return [str(a) + " : " + str(b) for a, b in zip(input_df[subset_columns_list[0]], input_df[subset_columns_list[1]])]


def market_basket_analysis(input_df, min_support=0.05, metric="lift", min_threshold=1, min_confidence=None):
    frequent_item_sets = apriori(input_df,
                                 min_support=min_support,
                                 use_colnames=True)
    rules = association_rules(frequent_item_sets, metric=metric, min_threshold=min_threshold)
    if rules.empty:
        return pd.DataFrame(columns=['antecedents', 'consequents', 'support', 'confidence', 'lift'])
    rules = rules.sort_values(['confidence', 'lift'], ascending=[False, False])
    rules = rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']]
    rules['lift'] = rules['lift'].round(1)
    rules['confidence'] = (rules['confidence'] * 100).round(2)
    if min_confidence:
        rules = rules[rules['confidence'] > min_confidence]
    rules['support'] = (rules['support'] * 100).round(2)
    rules['antecedents'] = rules['antecedents'].apply(lambda x: ', '.join(x))
    rules['consequents'] = rules['consequents'].apply(lambda x: ', '.join(x))
    return rules


def melt_dataframe(input_df, boolean_columns):
    # Melt the DataFrame to convert specified boolean columns into a single column
    melted_df = pd.melt(input_df, id_vars=input_df.columns.difference(boolean_columns),
                        value_vars=boolean_columns, var_name='Class', value_name='Value')
    # Filter out rows where the value is False
    melted_df = melted_df[melted_df['Value']]
    # Drop the 'Value' column as it's no longer needed
    melted_df.drop(columns=['Value'], inplace=True)
    return melted_df


def clean_string(input_text=None):
    text_string = ''.join(cat for cat in unicodedata.normalize('NFKD', input_text) if unicodedata.category(cat) != 'Mn')
    string_list = text_string.split()
    text_string = " ".join(string_list)
    for entry in [(",,",","), (" ,",","), ("  "," ")]:
        text_string = text_string.replace(entry[0], entry[1])
    return text_string.strip()


def concat_tuples_list(list_of_tuples, element_numer=0):
    if not list_of_tuples:
        return ""
    if len(list_of_tuples) == 1:
        return list_of_tuples[0][0]
    elements = list(set(element[element_numer] for element in list_of_tuples))
    joined_string = "; ".join(elements[:-1])
    last_element = elements[-1]
    return joined_string + " & " + last_element


def filter_dicts_by_value(list_of_dicts, key, values_to_check):
    return [d for d in list_of_dicts if d.get(key) in values_to_check]


def extract_unique_values(list_of_dicts, attribute):
    unique_values = set()
    for d in list_of_dicts:
        if attribute in d:
            unique_values.add(d[attribute])
    return list(unique_values)


def custom_join(strings):
    if len(strings) <= 1:
        return strings[0]
    else:
        return ', '.join(strings[:-1]) + ' & ' + strings[-1]


def generate_network(node_list, edge_list):
    # Create an empty graph
    G = nx.Graph()
    # Add nodes to the graph with node size as a node attribute
    color_scale = list(px.colors.sequential.Plasma + px.colors.sequential.Viridis + px.colors.sequential.Magma)
    for index, node in enumerate(node_list):
        G.add_node(node[0], size=node[1], label=f'{node[0]} : {node[1]}', color=color_scale[index])
    # Add edges to the graph with weight as an edge attribute
    for start_node, end_node, weight in edge_list:
        G.add_edge(start_node, end_node, weight=weight)
    return G


def centrality_detection(network_graph):
    centrality = nx.eigenvector_centrality(network_graph)
    centrality_list = sorted([v for v, c in centrality.items()], key=lambda x: x[1], reverse=True)[:10]
    page_rank = nx.pagerank(network_graph)
    page_rank_list = sorted([v for v, c in page_rank.items()], key=lambda x: x[1], reverse=True)[:10]
    betweenness = nx.betweenness_centrality(network_graph)
    betweenness_list = sorted([v for v, c in betweenness.items()], key=lambda x: x[1], reverse=True)[:10]
    communities = nx.community.greedy_modularity_communities(network_graph, weight='weight', cutoff=3)#, best_n=5)v
    for index, community in enumerate(communities[:5]):
        communities[index] = custom_join(list(community))
    ranking_list = [str(i) for i in range(1, 11)]
    algo_dataframe = pd.DataFrame(zip(ranking_list, centrality_list, page_rank_list, betweenness_list),
                                  columns=['Rank', 'Centrality Ranking', 'PageRank Ranking', 'Betweenness Ranking'])
    return algo_dataframe, communities


def generate_network_data(list_of_dicts, ignore_list):
    node_list = []
    # Loop through the list of dictionaries
    for document in list_of_dicts:
        # Extract the first element from each tuple in the 'tag_list' and add it to the master list
        sub_list = [tag_tuple[0] for tag_tuple in document['tag_list'] if tag_tuple[0] not in ignore_list]
        if len(sub_list) > 1:
            node_list.append(sorted(sub_list))
    node_counts = {}
    edge_counts = defaultdict(int)
    # Iterate over each sublist
    for sublist in node_list:
        sublist = sorted(sublist)
        for string in sublist:
            # Update the count for the string
            node_counts[string] = node_counts.get(string, 0) + 1
        for i in range(len(sublist)):
            for j in range(i + 1, len(sublist)):
                # Increment the count for the pair of strings
                edge_counts[(sublist[i], sublist[j])] += 1

    # Convert the dictionary to a list of tuples
    node_weight_list = [(string, count) for string, count in node_counts.items()]
    edge_weight_list = [(pair[0], pair[1], count) for pair, count in edge_counts.items()]
    return node_weight_list, edge_weight_list


def classification_list(list_of_dicts):
    node_list = []
    for document in list_of_dicts:
        sub_list = [tag_tuple[0] for tag_tuple in document['tag_list']]
        if len(sub_list) >= 1:
            node_list.extend(sub_list)
    return sorted(list(set(node_list)))

