import streamlit as st
import json
import plotly.graph_objects as go
import networkx as nx
import csv
import math
import matplotlib.colors as mcolors
import numpy as np

color_names = []
for name in mcolors.cnames:
    color_names.append(str(name))

def input_dialogue():
    G = nx.Graph()

    solution_idx = []
    solution_projected_x = []
    solution_projected_y = []
    solution_cluster_id = []
    solution_scale = []
    solution_description = []
    with open('classification_result_projected.csv', newline = '', encoding='utf-8') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        index = 0
        for row in spamreader:
            solution_idx.append(int(row[0]))
            solution_projected_x.append(float(row[1]))
            solution_projected_y.append(float(row[2]))
            solution_cluster_id.append(int(row[3]))
            solution_scale.append(int(row[4]))
            solution_description.append(row[5])

    cluster_x_min = []
    cluster_x_max = []
    cluster_y_min = []
    cluster_y_max = []
    for i in range(max(solution_cluster_id) + 1):
        x_in_cluster = []
        y_in_cluster = []
        for j in range(len(solution_idx)):
            if solution_cluster_id[j] == i:
                x_in_cluster.append(solution_projected_x[j])
                y_in_cluster.append(solution_projected_y[j])
        cluster_x_min.append(min(x_in_cluster))
        cluster_x_max.append(max(x_in_cluster))
        cluster_y_min.append(min(y_in_cluster))
        cluster_y_max.append(max(y_in_cluster))

    length = len(solution_idx)
    G.add_nodes_from(range(length))
    # add edges using edge_list [0] and [1]
    for i in range(length):
        G.nodes[i]['label'] = solution_description[i]
        G.nodes[i]['pos_x'] = solution_projected_x[i]
        G.nodes[i]['pos_y'] = solution_projected_y[i]
        G.nodes[i]['cluster_id'] = solution_cluster_id[i]
        G.nodes[i]['scale'] = solution_scale[i]
        G.nodes[i]['idx'] = solution_idx[i]


    # draw the nodes,nodes have different shapes according to their labels 0 or 1
    node_traces = []
    node_x = []
    node_y = []
    node0_x = []
    node0_y = []
    node0_color = []
    node0_cluster_name = []
    node0_size = []
    node1_x = []
    node1_y = []

    for node in G.nodes():
        x = G.nodes[node]['pos_x']
        y = G.nodes[node]['pos_y']
        node_x.append(x)
        node_y.append(y)

        node0_x.append(x)
        node0_y.append(y)
        node0_color.append(color_names[G.nodes[node]['cluster_id']])
        node0_size.append(G.nodes[node]['scale'] * 4 + 10)

    node_trace0 = go.Scatter(
        legendgroup="Nodes",
        legendgrouptitle_text="Node Importance",
        name="not defined",
        x=node0_x, y=node0_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=False,
            color=node0_color,
            size=node0_size,
            line_width=1,
            symbol='circle'
        ))
    # node_trace1 = go.Scatter(
    #     legendgroup="Nodes",
    #     name="1",
    #     x=node1_x, y=node1_y,
    #     mode='markers',
    #     hoverinfo='text',
    #     marker=dict(
    #         showscale=False,
    #         color='orangered',
    #         size=6,
    #         line_width=1,
    #         symbol='star'
    #     ))
    node_trace0.text = [G.nodes[node]['label'] for node in G.nodes()]
    # node_trace1.text = [dialogue_json[node]['speaker'] + ": " + dialogue_json[node]['text'] for node in G.nodes() if
    #                     G.nodes[node]['label'] == 1]
    node_traces.append(node_trace0)
    # node_traces.append(node_trace1)

    # draw the index number of the nodes
    # node_index_trace = go.Scatter(
    #     name="Node Index",
    #     x=node_x, y=node_y,
    #     mode='text',
    #     hoverinfo='text',
    #     text=[str(G.nodes[node]['idx']) for node in G.nodes()],
    #     textposition="top center",
    #     textfont=dict(
    #         size=14,
    #         color="black"
    #     )
    # )

    # draw the graph, legend is the 16 different relations without any nodes
    fig = go.Figure(data=[*node_traces],
                    layout=go.Layout(
                        title='<br>Graph of PCA Projection',
                        titlefont_size=16,
                        showlegend=True,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        annotations=[dict(
                            text="",
                            showarrow=False,
                            xref="paper", yref="paper",
                            x=0.005, y=-0.002)],
                        xaxis=dict(showgrid=True, zeroline=True, showticklabels=True),
                        yaxis=dict(showgrid=True, zeroline=True, showticklabels=True))
                    )
    
    for i in range(len(cluster_x_max)):
        fig.add_shape(type="rect",
            xref="x", yref="y",
            x0=cluster_x_min[i], y0=cluster_y_min[i], x1=cluster_x_max[i], y1=cluster_y_max[i],
            fillcolor=color_names[i],
            opacity=0.4,
            line_width=0.5
        )

    st.plotly_chart(fig)


st.title("Graph")
input_dialogue()
