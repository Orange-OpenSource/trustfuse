"""
Software Name : TrustFuse
SPDX-FileCopyrightText: Copyright (c) Orange SA
SPDX-License-Identifier: MIT
 
This software is distributed under the MIT License,
see the "LICENSE.txt" file for more details or https://spdx.org/licenses/MIT.html
 
Authors: Lucas Jarnac, Yoan Chabot, and Miguel Couceiro
Software description: TrustFuse is a testbed that supports experimentation with data fusion models,
their evaluation, and the visualization of datasets as graphs or tables within a unified user interface.
"""

import os
import shutil

import networkx as nx
from pyvis.network import Network
import pandas as pd


def set_colors_map(attributes):
    """Assign a color to each attribute in attributes

    Args:
        attributes (dict): attributes for each bucket

    Returns:
        dict: hexadecimal code color for each attribute
    """
    distinct_attributes = set()
    for bid in attributes:
        distinct_attributes |= set(attributes[bid])

    colors_map = generate_random_colors(len(distinct_attributes))

    return {attr: colors_map[i] for i, attr in enumerate(distinct_attributes)}


def generate_random_colors(n):
    """Generate a random color"""
    palette = ["#FF7900", "#4BB4E6", "#50BE87", "#FFB4E6", "#A885D8",
               "#FFD200", "#B5E8F7", "#B8EBD6", "#FFE8F7", "#D9C2F0",
               "#FFF6B6", "#527EDB", "#32C832", "#FFCC00", "#CD3C14"]
    nb_colors = len(palette)
    colors = (n // nb_colors) * palette + (n % nb_colors) * palette
    return colors


def visualize(dataset,
              attributes,
              graph="Output",
              metrics=None,
              bid=0,
              gradio=False) -> None:
    
    attribute_colors = dataset.colors_map

    df_mapping = {
        "Output": dataset.fmt_fused_data,
        "Input": dataset.data_pp, # dataset.seed_data
        "Ground Truth": dataset.seed_gt_data
    }

    if graph not in df_mapping:
        possible_values = "\n"
        for idx, value in enumerate(df_mapping, 1):
            possible_values += f"{idx}. {value}\n"
        raise ValueError(f"Graph {graph} cannot be displayed, " \
                         "choose one of these values: {possible_values}")

    df_to_visualize = df_mapping[graph]

    distinct_entities = set(df_to_visualize[bid][dataset.entity_col_name])
    distinct_attributes = set(df_to_visualize[bid].columns)
    entity_id_mapping = {ent: i+1 for i, ent in enumerate(distinct_entities)}
    attribute_id_mapping = {attr: i+1 for i, attr in enumerate(distinct_attributes)}


    g = nx.DiGraph()
    for idx, row in df_to_visualize[bid].iterrows():
        entity = row[dataset.entity_col_name]
        if entity is not None:
            if not g.has_node(entity):
                g.add_node(entity, color="black") # title= ...
            for attribute, obj in row.items():
                if attribute not in [dataset.entity_col_name, "Source"] \
                    and attribute in attributes[bid]:
                    add_attribute = True
                    if isinstance(obj, list):
                        if len(obj) == 0 or obj[0] is None:
                            add_attribute = False
                    elif obj is None or pd.isna(obj):
                        add_attribute = False
                    if add_attribute:
                        shape = "dot"
                        if (attribute in dataset.attribute_types and
                            dataset.attribute_types[attribute] in ["quantity",
                                                                    "time", 
                                                                    "coordinates", 
                                                                    "string"]):
                            shape = "box"
                        obj_color = attribute_colors[attribute]
                        if shape == "box":
                            obj_color = "#D6D6D6"
                        node_id = int(str(entity_id_mapping[entity])
                                    + str(attribute_id_mapping[attribute]))
                        g.add_node(node_id, label=attribute, color=obj_color, shape=shape)
                        g.add_edge(entity, node_id, label="", attribute=attribute)
                        attribute_name = attribute
                        if isinstance(obj, list):
                            for val in obj:
                                if val is not None:
                                    val_str = str(val)
                                    if not g.has_node(val_str):
                                        g.add_node(val_str, label=val_str, color=obj_color, shape=shape)
                                    g.add_edge(node_id, val_str, title=attribute_name, attribute=attribute)
                        elif not pd.isna(obj) and obj is not None:
                            obj_name = str(obj)
                            obj_color = attribute_colors[attribute]
                            if shape == "box":
                                obj_color = "#D6D6D6"
                            if not g.has_node(obj_name):
                                g.add_node(obj_name, color=obj_color, shape=shape)
                            g.add_edge(node_id, obj_name, title=attribute_name, attribute=attribute)
    net = Network(notebook=True, directed=True, height="750px", width="100%", bgcolor="white",
                    font_color="black", select_menu=True, cdn_resources='remote', filter_menu=True)

    if len(g) > 2000:
        net.repulsion()
        net.toggle_physics(False)

    net.from_nx(g)
    if gradio:
        return net
    else:
        # Saving path
        dir_path = os.path.join(os.getcwd(), "web_app/static")
        os.makedirs(dir_path, exist_ok=True)
        record_path = os.path.join(dir_path, graph)
        if os.path.exists(record_path):
            shutil.rmtree(record_path)

        os.makedirs(record_path, exist_ok=True)
        net.show(os.path.join(record_path, f"{graph}_{bid}.html"))


def visualize_partial_orders(dataset, bid):
    nt = Network(notebook=True, directed=True, cdn_resources='remote', width="100%", height="300px")
    if hasattr(dataset, "partial_orders") and len(dataset.partial_orders[bid].values()) > 0:
        graph_merged = nx.compose_all(dataset.partial_orders[bid].values())
        nt.from_nx(graph_merged)
        return nt
    return None
