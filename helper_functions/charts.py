import plotly.express as px
import copy
from wordcloud import WordCloud
import matplotlib.pyplot as plt


color_scale = px.colors.qualitative.Plotly[:20]

label_lookup = {
    'peak_name': 'Peak Name',
    'termreason': 'Reason Terminated',
    'location': 'Location',
    'region': 'Region',
    'year': 'Year',
    'season': 'Season',
    'host': 'Host',
    'lda_cluster': 'LDA Cluster',
    'nmf_cluster': 'NMF Cluster',
    'bert_cluster': 'Bert Cluster'
}


def generate_wordcloud(freq):
    wordcloud = WordCloud(width=800,
                          height=800,
                          mode='RGBA',
                          background_color=None).generate_from_frequencies(freq)
    fig = plt.subplots(6,6)
    plt.imshow(wordcloud,
              interpolation='bilinear')
    plt.axis('off')
    return fig


def create_scatter_chart(input_df,
                         colour_col,
                         x_col,
                         y_col,
                         z_col=None):
    df_copy = copy.deepcopy(input_df)
    df_copy = df_copy.sort_values(by=colour_col)
    df_copy[colour_col] = 'Cluster ' + df_copy[colour_col].astype(str)
    if z_col:
        fig = px.scatter_3d(df_copy, x=x_col, y=y_col, z=z_col,
                            color=colour_col, height=800,
                            color_discrete_sequence=color_scale,
                            hover_data={'3d_umap_x': False,
                                        '3d_umap_y': False,
                                        '3d_umap_z': False,
                                        'accidents': True,
                                        'peak_name': True,
                                        'expidition_id': True,
                                        'termreason': True
                                        }
                            )
        fig.update_layout(scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        ))
        fig.update_traces(marker=dict(size=15,
                                      line=dict(
                                          color='black',
                                          width=1
                                      )))
        return fig
    else:
        fig = px.scatter(df_copy, x=x_col, y=y_col,
                         color=colour_col, height=800,
                         color_discrete_sequence=color_scale,
                         # hover_data=["accidents"]
                         hover_data={'2d_umap_x': False,
                                     '2d_umap_y': False,
                                     'accidents': True,
                                     'peak_name': True,
                                     'expidition_id': True,
                                     'termreason': True
                                     }
                         )
        fig.update_layout(scene=dict(
            xaxis_title='X',
            yaxis_title='Y'
        ))
        fig.update_traces(marker=dict(size=15,
                                      line=dict(
                                          color='black',
                                          width=1
                                      )))
        return fig


def create_grouped_barchart(input_df,
                            column_one,
                            column_two):
    input_df = input_df.groupby([column_one,
                                 column_two]).size().reset_index(name='count')
    col_one = label_lookup.get(column_one, column_one)
    col_two = label_lookup.get(column_two, column_two)
    input_df = input_df.rename(columns={column_one: col_one, column_two: col_two})
    input_df = input_df.sort_values(by=[col_one, col_two], ignore_index=True, ascending=[True,True])
    fig = px.bar(input_df, x=col_one, y='count', color=col_two, barmode='group', height=800)
    # Update layout
    fig.update_layout(
        xaxis_title=col_one,
        yaxis_title='Count',
        title=f'Grouped Bar Chart split by {col_one} and {col_two}'
    )
    return fig


def create_injury_barchart(input_df):
    true_counts = input_df.sum()
    true_counts = true_counts.sort_values(ascending=False)
    # Create a bar chart using Plotly Express
    fig = px.bar(x=true_counts.index, y=true_counts.values,
                 labels={'x': 'Injury Report Type', 'y': 'Count'}, height=800)
    fig.update_layout(title='Occurrence of Each Injury Report Type')
    return fig


def create_barchart(input_df,
                    column_one):
    input_df = input_df[column_one].value_counts().reset_index()
    col_one = label_lookup[column_one]
    input_df.columns = [col_one, 'count']
    input_df = input_df.sort_values(by='count',
                                    ascending=False)

    fig = px.bar(input_df, x=col_one, y='count', height=800)
    # Update layout
    fig.update_layout(
        xaxis_title=col_one,
        yaxis_title='Count',
        title=f'Bar Chart Count of {col_one}'
    )
    return fig


def create_stacked_line_chart(input_df, values_filter=None):
    class_counts_df = input_df.groupby(['year', 'Class']).size().reset_index(name='Count')
    if values_filter and len(values_filter) > 0 :
        class_counts_df = class_counts_df[class_counts_df['Class'].isin(values_filter)]
    fig = px.area(class_counts_df, x='year', y='Count', color='Class', line_group='Class',
                  labels={'Count': 'Number of Occurrences'},
                  title='Number of Occurrences of Each Class by Year',
                  # category_orders={'Class': ['A', 'B']},
                  height= 800,
                  color_discrete_sequence=px.colors.qualitative.Set1)
    # fig.update_layout(yaxis_range=[0, 180])
    fig.update_yaxes(tick0=0,
                     dtick=10,
                     range=[0, 170]
                     )
    return fig

