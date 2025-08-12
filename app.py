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

import pickle
import os
import re
import json
import logging
import inspect

import gradio as gr
import pandas as pd

from trustfuse.conflicting_dataset.dataset import (DynamicDataset,
                                                   StaticDataset)
from trustfuse.evaluation import evaluation
from trustfuse.visualization import visualization
import settings
import utils
from trustfuse.conflicting_dataset.preprocessing import (DATA_PREPROCESSING_FUNCTIONS,
                                                         METADATA_PREPROCESSING_FUNCTIONS)


PREPROCESSING_FUNCTIONS = (list(DATA_PREPROCESSING_FUNCTIONS.keys())
                           + list(METADATA_PREPROCESSING_FUNCTIONS.keys()))

THEME = gr.themes.Default(primary_hue="orange",
                          secondary_hue="blue")

INITIAL_COMPONENT_STATES = {
    "dataset_global": gr.update(value=None),
    "colors_map": gr.update(value=None),
    "metrics_global": gr.update(value=None),
    "preprocessing_list": {
        "modify_structure": {},
        "modify_data": {}
    },
    "dataset_selector": gr.update(label="Select the pickle file",
                                  value=None,
                                  interactive=True,
                                  file_count="single",
                                  file_types=[".pkl"],
                                  height=140),
    "data_selector": gr.update(label="📂 Select conflicting data folder",
                               file_count="directory",
                               value=None,
                               interactive=True,
                               height=140,
                               visible=False),
    "gt_selector": gr.update(label="📂 Select GT data folder",
                             file_count="directory",
                             value=None,
                             interactive=True,
                             height=140,
                             visible=False),
    "parameters_selector": gr.update(label="Select the dataset parameters",
                                     file_count="single",
                                     value=None,
                                     interactive=True,
                                     height=140,
                                     visible=False),
    "type_selector": gr.update(label="Select attribute/datatatype mapping file",
                               file_count="single",
                               value=None,
                               interactive=True,
                               height=140,
                               visible=False),
    "preprocessing_file_selector": gr.update(label="You can choose a preprocessing file",
                                             interactive=True,
                                             value=None,
                                             file_count="single",
                                             file_types=[".json"],
                                             height=140),
    "function_dropdown": gr.update(PREPROCESSING_FUNCTIONS,
                                   value=None,
                                   label="Preprocessing functions"),
    "property_selector": gr.update(choices=[""],
                                   value=None,
                                   label="Concerned properties",
                                   multiselect=True),
    "preprocessing_list_display": gr.update(choices=[],
                                            value=None,
                                            multiselect=True,
                                            label="Preprocessing to be performed"),
    "preprocessing_information": gr.update(label="Selected preprocessing",
                                           value=None),
    "preprocessing_state": gr.update(label="Preprocessing state",
                                     value=None),
    "metric_state": "",
    "model_information": gr.update(label="Model state",
                                   interactive=False,
                                   value=None),
    "bucket_state": gr.update(value=0),
    "available_buckets": [0],
    "file_output": gr.update(label="📄 Dataset",
                             interactive=False,
                             value=None,),
    "bucket_id": gr.update(label="Current bucket",
                           choices=["Bucket #0"],
                           interactive=False,
                           value=""),
    "prev_button": gr.update(interactive=False),
    "next_button": gr.update(interactive=False),
    "toggle": gr.update(["Input", "Ground Truth"],
                        label="Display",
                        value="Input"),
    "partial_order": gr.update(value=""),
    "graph_output": gr.update(value=""),
    "table_output": gr.update(label="📋 Data loaded",
                              value=None),
    "n_sources": gr.update(value="",
                           label="Enter the Top N of desired sources",
                           visible=False),
    "top_n_sources": gr.update(headers=["Source", "Score"],
                               visible=False,
                               value=None),
    "is_reset": gr.update(value=False),
    "preprocessing_tab": gr.update(interactive=False),
    "model_tab": gr.update(interactive=False),
    "tabs": gr.update(selected=0)
}

# Absolute paths to load available datasets
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_DIRECTORY = 'data/input_trustfuse/wikiconflict/'

MODEL_PAPERS = {
    "CRH": "[CRH paper](https://dl.acm.org/doi/pdf/10.1145/2588555.2610509)",
    "CATD": "[CATD paper](https://dl.acm.org/doi/pdf/10.14778/2735496.2735505)",
    "KDEm": "[KDEm paper](https://dl.acm.org/doi/pdf/10.1145/2939672.2939837)",
    "GTM": "[GTM paper](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=2ffe1157df90ce94cb91f28074b43b58135cedac)",
    "TruthFinder": "[TruthFinder paper](https://dl.acm.org/doi/pdf/10.1145/1281192.1281309)",
    "TKGC": "[TKGC paper]()",
    "SLIMFAST": "[SLiMFast paper](https://dl.acm.org/doi/pdf/10.1145/3035918.3035951)",
    "LTM": "[LTM paper](https://dl.acm.org/doi/pdf/10.14778/2168651.2168656)",
    "ACCU": "[ACCU paper](https://dl.acm.org/doi/pdf/10.14778/1687627.1687690)",
}

# Existing datasets in the app
DYNAMIC_DATASETS_AVAILABLE = {
    "Paris monuments (WikiConflict)": os.path.join(BASE_DIR,
                                                   "data",
                                                   "input_trustfuse",
                                                   "wikiconflict",
                                                   "monuments_in_paris",
                                                   "data.pkl")
}
STATIC_DATASETS_AVAILABLE = {
    "Flight": {
        "attr_type_path": os.path.join(BASE_DIR, "data",
                                       "configurations",
                                       "truthfinder", "flight",
                                       "types.json"),
        "params_path": os.path.join(BASE_DIR, "data",
                                    "input_trustfuse",
                                    "flight", "dataset_parameters.json"),
        "data_folder": os.path.join(BASE_DIR, "data",
                                    "input_trustfuse",
                                    "flight", "conflicting_data"),
        "gt_folder": os.path.join(BASE_DIR, "data",
                                  "input_trustfuse",
                                  "flight", "ground_truth"),
        "preprocess": os.path.join(BASE_DIR, "data",
                                   "configurations",
                                   "truthfinder", "flight",
                                   "preprocess_configuration.json")
    },
    "Stock": {
        "attr_type_path": os.path.join(BASE_DIR, "data",
                                       "configurations",
                                       "truthfinder", "stock",
                                       "types.json"),
        "params_path": os.path.join(BASE_DIR, "data",
                                    "input_trustfuse",
                                    "stock", "dataset_parameters.json"),
        "data_folder": os.path.join(BASE_DIR, "data",
                                    "input_trustfuse",
                                    "stock", "conflicting_data"),
        "gt_folder": os.path.join(BASE_DIR, "data",
                                  "input_trustfuse",
                                  "stock", "ground_truth"),
        "preprocess": os.path.join(BASE_DIR, "data",
                                   "configurations",
                                   "truthfinder", "stock",
                                   "preprocess_configuration.json")
    },
    "Book": {
        "attr_type_path": os.path.join(BASE_DIR, "data",
                                       "configurations",
                                       "truthfinder", "book",
                                       "types.json"),
        "params_path": os.path.join(BASE_DIR, "data",
                                    "input_trustfuse",
                                    "book", "dataset_parameters.json"),
        "data_folder": os.path.join(BASE_DIR, "data",
                                    "input_trustfuse",
                                    "book", "conflicting_data"),
        "gt_folder": os.path.join(BASE_DIR, "data",
                                  "input_trustfuse",
                                  "book", "ground_truth"),
        "preprocess": os.path.join(BASE_DIR, "data",
                                   "configurations",
                                   "truthfinder", "book",
                                   "preprocess_configuration.json")
    },
}


### Dataset loading functions
def load_dataset(dataset, mode, reset, cleaned):
    """Load dataset in pickle format as described in the doc"""

    if reset and not cleaned:
        components = ["table_output",
                      "file_output",
                      "graph_output",
                      "colors_map",
                      "partial_order",
                      "bucket_id",
                      "dataset_global",
                      "preprocessing_tab",
                      "model_tab",
                      "prev_button",
                      "next_button",
                      "bucket_id"]
        return [INITIAL_COMPONENT_STATES[component]
                for component in components] + [True, True]


    loaded_dataset = None
    if dataset is None:
        return None, "📂 No dataset selected.", "", ""
    dataset_name = dataset.name


    with open(os.path.join(BASE_DIR, "data", "configurations", "crh",
                           "wikiconflict", "property_types.pkl"), 'rb') as f:
        attr_types = pickle.load(f)
        # Define types not in configuration file
        attr_types["label_en"] = "string"
        attr_types["label_fr"] = "string"
        attr_types["description_en"] = "string"
        attr_types["description_fr"] = "string"

    try:
        if dataset_name.endswith('.csv'):
            df = pd.read_csv(dataset_name)

        elif dataset_name.endswith('.pkl'):

            loaded_dataset = DynamicDataset(dataset_name,
                                            attribute_types=attr_types,
                                            **settings.DATASET_PARAMETERS["WikiConflict"],
                                            entity_as="string")
            loaded_dataset.make_post_preprocess_copy()

            df = loaded_dataset.data[0]

        else:
            return (None,
                    "⚠️ Format not supported.",
                    "<p style='color:red'>⚠️ No dataset selected, the graph cannot be generated.</p>",
                    None,
                    "",
                    "?",
                    "",
                    loaded_dataset)
        interactive = False
        nb_buckets = len(loaded_dataset.data)
        if nb_buckets > 1:
            interactive = True
        return (df,
                "✅ Dataset successfully loaded.",
                *generate_graph(0, mode, loaded_dataset),
                generate_partial_order_graph(0, loaded_dataset),
                gr.update(value="Bucket #0",
                          choices=[f"Bucket #{key}" for key in loaded_dataset.data]),
                loaded_dataset,
                gr.update(interactive=True),
                gr.update(interactive=True),
                gr.update(interactive=False),
                gr.update(interactive=interactive),
                gr.update(interactive=True),
                False,
                False)
    except Exception as e:
        return (None,
                f"❌ Error : {str(e)}",
                "<p style='color:red'>⚠️ No dataset selected, the graph cannot be generated.</p>",
                None,
                "",
                "?",
                "",
                loaded_dataset)


### Dataset loading functions
def load_available_dataset(dataset_name, mode, reset, cleaned):
    """Load dataset in pickle format as described in the doc"""
    if reset and not cleaned:
        components = ["table_output",
                      "file_output",
                      "graph_output",
                      "colors_map",
                      "partial_order",
                      "bucket_id",
                      "dataset_global",
                      "preprocessing_tab",
                      "model_tab",
                      "prev_button",
                      "next_button",
                      "bucket_id"]
        return [INITIAL_COMPONENT_STATES[component]
                for component in components] + [True, True]

    loaded_dataset = None
    if dataset_name in DYNAMIC_DATASETS_AVAILABLE:

        with open(os.path.join(BASE_DIR, "data", "configurations", "crh", "wikiconflict", "property_types.pkl"), 'rb') as f:
            attr_types = pickle.load(f)
            # Define types not in configuration file
            attr_types["label_en"] = "string"
            attr_types["label_fr"] = "string"
            attr_types["description_en"] = "string"
            attr_types["description_fr"] = "string"

        try:
            loaded_dataset = DynamicDataset(DYNAMIC_DATASETS_AVAILABLE[dataset_name],
                                            attribute_types=attr_types,
                                            **settings.DATASET_PARAMETERS["WikiConflict"],
                                            entity_as="string")
            loaded_dataset.make_post_preprocess_copy()

            df = loaded_dataset.data[0]

            interactive = False
            nb_buckets = len(loaded_dataset.data)
            if nb_buckets > 1:
                interactive = True
            return (df,
                    "✅ Dataset successfully loaded.",
                    *generate_graph(0, mode, loaded_dataset),
                    generate_partial_order_graph(0, loaded_dataset),
                    gr.update(value="Bucket #0",
                            choices=[f"Bucket #{key}" for key in loaded_dataset.data]),
                    loaded_dataset,
                    gr.update(interactive=True),
                    gr.update(interactive=True),
                    gr.update(interactive=False),
                    gr.update(interactive=interactive),
                    gr.update(interactive=True),
                    False,
                    False)

        except Exception as e:
            print(e)
            return f"❌ Error : {str(e)}", loaded_dataset

    if dataset_name in STATIC_DATASETS_AVAILABLE:
        file_path = STATIC_DATASETS_AVAILABLE[dataset_name]["attr_type_path"]
        with open(file_path, encoding="utf-8") as f:
            attr_types = json.load(f)

        try:
            data_folder_path = STATIC_DATASETS_AVAILABLE[dataset_name]["data_folder"]
            gt_folder_path = STATIC_DATASETS_AVAILABLE[dataset_name]["gt_folder"]
            data_folder = [os.path.join(data_folder_path, bucket)
                           for bucket in os.listdir(data_folder_path)]
            gt_folder = [os.path.join(gt_folder_path, bucket)
                         for bucket in os.listdir(gt_folder_path)]

            loaded_dataset = StaticDataset([data_folder, gt_folder],
                                attribute_types=attr_types,
                                gradio=True,
                                **settings.DATASET_PARAMETERS[dataset_name]) # **parameters
            loaded_dataset.make_post_preprocess_copy()
            df = loaded_dataset.data_pp[0]
            interactive = False
            nb_buckets = len(loaded_dataset.data)
            if nb_buckets > 1:
                interactive = True

            return (df,
                    "✅ Dataset successfully loaded.",
                    *generate_graph(0, mode, loaded_dataset),
                    generate_partial_order_graph(0, loaded_dataset),
                    gr.update(value="Bucket #0",
                            choices=[f"Bucket #{key}" for key in loaded_dataset.data]),
                    loaded_dataset,
                    gr.update(interactive=True),
                    gr.update(interactive=True),
                    gr.update(interactive=False),
                    gr.update(interactive=interactive),
                    gr.update(interactive=True),
                    False,
                    False)

        except Exception as e:
            print(e)
            return (None,
                    f"❌ Error : {str(e)}",
                    "<p style='color:red'>⚠️ No dataset selected, the graph cannot be generated.</p>",
                    None,
                    "?",
                    "",
                    loaded_dataset)
        

def display_dataset(mode, dataset, reset, cleaned):
    if reset and not cleaned:
        components = ["table_output",
                      "graph_output",
                      "colors_map",
                      "partial_order",
                      "bucket_id"]
        return [INITIAL_COMPONENT_STATES[component]
                for component in components]
    df = dataset.data_pp[0]
    return (df,
            *generate_graph(0, mode, dataset),
            generate_partial_order_graph(0, dataset),
            gr.update(value="Bucket #0",
                      choices=[f"Bucket #{key}" for key in dataset.data]))


def load_dataset_from_folders(data_folder, gt_folder, type_mapping,
                              dataset_parameters, mode, reset, cleaned):
    """Load datasets from two folders containing a list of buckets"""
    if reset and not cleaned:
        components = ["table_output",
                      "file_output",
                      "graph_output",
                      "colors_map",
                      "partial_order",
                      "bucket_id",
                      "dataset_global",
                      "preprocessing_tab",
                      "model_tab",
                      "prev_button",
                      "next_button",
                      "bucket_id"]
        return [INITIAL_COMPONENT_STATES[component]
                for component in components] + [True, True]

    loaded_dataset = None
    if data_folder is None or gt_folder is None:
        return None, "📂 No dataset selected.", "", ""

    with open(type_mapping, encoding="utf-8") as f:
        attr_types = json.load(f)

    with open(dataset_parameters, encoding="utf-8") as f:
        parameters = json.load(f)

    try:
        loaded_dataset = StaticDataset([data_folder, gt_folder],
                            attribute_types=attr_types,
                            gradio=True,
                            **parameters)
        loaded_dataset.make_post_preprocess_copy()

        df = loaded_dataset.data[0]

        interactive = False
        nb_buckets = len(loaded_dataset.data)
        if nb_buckets > 1:
            interactive = True

        return (df,
                "✅ Dataset successfully loaded.",
                *generate_graph(0, mode, loaded_dataset),
                generate_partial_order_graph(0, loaded_dataset),
                gr.update(value="Bucket #0",
                          choices=[f"Bucket #{key}" for key in loaded_dataset.data]),
                loaded_dataset,
                gr.update(interactive=True),
                gr.update(interactive=True),
                gr.update(interactive=False),
                gr.update(interactive=interactive),
                gr.update(interactive=True),
                False,
                False)
    except Exception as e:
        print(e)
        return (None,
                f"❌ Error : {str(e)}",
                "<p style='color:red'>⚠️ No dataset selected, the graph cannot be generated.</p>",
                None,
                "?",
                "",
                loaded_dataset)


def generate_graph(bid, mode, dataset):
    """Generate the graph visualization for the bucket bid"""
    colors_map_initialization = None
    if colors_map_initialization is None:
        colors_map_initialization = visualization.set_colors_map(dataset.attributes)
    net = visualization.visualize(dataset,
                            dataset.attributes,
                            graph=mode,   # mode = Input or Ouptput
                            gradio=True,
                            bid = bid)
    
    graph_html = net.generate_html()
    graph_html = utils.fix_html(graph_html)
    return (""" <iframe style="width: 100%; height: 600px; """
            """ margin:0 auto" frameborder="0" """
            f""" srcdoc='{graph_html}'></iframe> """,
            colors_map_initialization)


def generate_partial_order_graph(bid, dataset):
    """Generate the partial order graph of the bucket bid"""
    net = visualization.visualize_partial_orders(dataset, bid)
    if net is not None:
        graph_html = net.generate_html()
        fixed_html = graph_html.replace("'", "\"")
        return (""" <iframe style="width: 100%; """
                """ height: 300px; margin:0 auto" """
                f"""frameborder="0" srcdoc='{fixed_html}'></iframe>""")
    return ("<p style='color:#FF7900'>"
            "⚠️ No partial orders for the current bucket.</p>")


def update_dataframe(index, mode, sources, dataset, reset):
    """Update the bucket"""

    if reset:
        components = ["bucket_id",
                      "table_output",
                      "graph_output",
                      "colors_map",
                      "partial_order",
                      "top_n_sources"]
        return [INITIAL_COMPONENT_STATES[component]
                for component in components]

    top_sources = gr.update()
    if len(dataset.weights_dict) > 0:
        top_sources = gr.update(visible=True,
                                  value=top_n(sources, index, dataset, reset))
    return (f"Bucket #{index}", dataset.data_pp[index],
            *generate_graph(index, mode, dataset),
            generate_partial_order_graph(index, dataset),
            top_sources)


def update_dataframe_from_dropdown(index, mode, sources, dataset, reset):
    """Load the bucket from the selected one and update the visualization"""
    if reset:
        components = ["bucket_state",
                      "table_output",
                      "graph_output",
                      "partial_order",
                      "top_n_sources"]
        return [INITIAL_COMPONENT_STATES[component]
                for component in components]
    index = int(re.search(r"Bucket #(\d+)", index).group(1))
    dataset_mapping = {
        "Input": dataset.data_pp,
        "Output": dataset.fmt_fused_data,
        "Ground Truth": dataset.seed_gt_data
    }
    prev_interactive = True
    next_interactive = True
    if index + 1 == len(dataset_mapping[mode]):
        next_interactive = False
    if index == 0:
        prev_interactive = False
    top_sources = gr.update()
    if len(dataset.weights_dict) > 0:
        top_sources = gr.update(visible=True,
                                  value=top_n(sources, index,
                                              dataset, reset))
    return (index, dataset_mapping[mode][index],
            *generate_graph(index, mode, dataset),
            generate_partial_order_graph(index, dataset),
            top_sources, gr.update(interactive=prev_interactive),
            gr.update(interactive=next_interactive))


def prev_dataframe(index, mode, sources, dataset, reset):
    """Load the previous bucket and update the visualization"""
    if reset:
        components = ["prev_button",
                      "bucket_state",
                      "bucket_id",
                      "table_output",
                      "graph_output",
                      "colors_map",
                      "partial_order",
                      "top_n_sources"]
        return [INITIAL_COMPONENT_STATES[component]
            for component in components]
    index -= 1
    interactive = True
    if index == 0:
        interactive = False
    return (gr.update(interactive=interactive),
            gr.update(interactive=True),
            index,
            *update_dataframe(index, mode, sources, dataset, reset))


def next_dataframe(index, mode, sources, dataset, reset):
    """Load the next bucket and update the visualization"""
    if reset:
        components = ["next_button"
                      "bucket_state",
                      "bucket_id",
                      "table_output",
                      "graph_output",
                      "colors_map",
                      "partial_order",
                      "top_n_sources"]
        return [INITIAL_COMPONENT_STATES[component]
            for component in components]
    index += 1
    interactive = True
    if index == len(dataset.data_pp) - 1:
        interactive = False
    return (gr.update(interactive=interactive),
            gr.update(interactive=True),
            index,
            *update_dataframe(index, mode, sources, dataset, reset))


def toggle_display(choice, bid, dataset, reset):
    """Change the the visualization choice among Input/Output"""
    if reset:
        components = ["table_output",
                      "graph_output",
                      "colors_map"]
        return [INITIAL_COMPONENT_STATES[component]
            for component in components]

    if dataset is None:
        return None, "", None
    dataset_mapping = {
        "Input": dataset.data_pp,
        "Output": dataset.fmt_fused_data,
        "Ground Truth": dataset.seed_gt_data
    }
    return dataset_mapping[choice][bid], *generate_graph(bid, choice, dataset)


def apply_model(df, model_name, dataset, parameters, progress=gr.Progress()):
    """Run the fusion model on the loaded dataset"""
    if df is None:
        return "⚠️ No dataset selected."
    computed_metrics = None
    logging.info("Apply %s", model_name)
    model = settings.MODEL_MAP[model_name](dataset,
                                           progress=progress,
                                           **parameters)
    for bid, inputs in progress.tqdm(model.model_input.items(),
                                     desc="Fusion"):
        results = model.fuse(dataset, bid, inputs, progress=progress)
        logging.info("Performing reverse mapping")
        dataset.reverse_mapping(results, bid, progress)
        logging.info("Metrics computation")
        _, metrics = evaluation.get_metrics(dataset,
                                            dataset.attributes,
                                            mode="positive",
                                            progress=progress)
        logging.info("Metrics computed")
        metrics = metrics.style.applymap(utils.color_gradient, subset=metrics.columns[1:])
        metrics_html = utils.display_metrics(metrics, model_name)
        computed_metrics = metrics_html
        yield ("BID",
               [f"Bucket #{key}" for key in dataset.fmt_fused_data],
               metrics_html,
               dataset,
               computed_metrics)


def update_bucket_id(dataset, reset):
    if reset:
        return INITIAL_COMPONENT_STATES["bucket_id"]
    return gr.update(choices=[f"Bucket #{key}"
                               for key in dataset.fmt_fused_data])


def display_model_output(reset):
    if reset:
        components = ["toggle", "n_sources"]
        return [INITIAL_COMPONENT_STATES[component]
            for component in components]
    return (gr.update(choices=["Input", "Ground Truth", "Output"], value="Output"),
            gr.update(visible=True))


def display_metrics_gr(metrics):
    return metrics


def update_visibility(choice, reset):
    """Updates the component display according to the choice of the user."""
    if reset:
        components = ["dataset_selector",
                      "data_selector",
                      "gt_selector"]
        return [INITIAL_COMPONENT_STATES[component]
                for component in components]
    if choice == "Single file (Pickle)":
        return (gr.update(visible=True),
                gr.update(visible=False),
                gr.update(visible=False))
    else:
        return (gr.update(visible=False),
                gr.update(visible=True),
                gr.update(visible=False))


def update_selector_visibility(reset):
    """Change the visibility of File selector component"""
    if reset:
        return gr.update(visible=False)
    return gr.update(visible=True)


def top_n(n, bucket, dataset, reset):

    if reset:
        return INITIAL_COMPONENT_STATES["top_n_sources"]

    sorted_scores = sorted(dataset.weights_dict[bucket].items(),
                           key=lambda x: x[1],
                           reverse=True)

    max_n = min(10, len(sorted_scores))

    try:
        n = int(n) if n else max_n
        n = max(1, min(n, len(sorted_scores)))
    except ValueError:
        return "⚠️ Enter a valid number"

    return sorted_scores[:n]

def init_preprocessing_list(preprocessing_file, reset):
    if reset:
        components = ["preprocessing_information", "preprocessing_list"]
        return [INITIAL_COMPONENT_STATES[component]
                for component in components]
    loaded_preprocessing_file = None
    with open(preprocessing_file, encoding="utf-8") as f:
        loaded_preprocessing_file = json.load(f)
    return loaded_preprocessing_file, loaded_preprocessing_file


def update_property_selection_mode(mode, dataset):
    """ Update the Dropdown options related to the selected mode """
    property_names = list(dataset.data[len(dataset.data)-1].keys())
    if mode == "Property type":
        property_types = {
            dataset.attribute_types[k]
            for k in property_names
            if k in dataset.attribute_types
        }
        return gr.update(
            choices=property_types,
            value=[None]
            )
    if mode == "Property name":
        return gr.update(
            choices=property_names,
            value=[None]
            )
    return gr.update(choices=["All"], value="All")


def add_preprocessing(selected_function, mode, selected_attributes, preprocessing):
    """ Add a preprocessing function to the list and update the content displayed """
    if not selected_function:
        return gr.update(), {}

    if mode == "Property type":
        action = "type_selection"
    elif mode == "Property name":
        action = "custom"
    else:
        action = "all"

    if selected_function not in preprocessing:
        if selected_function == "extract_authors":
            preprocessing["modify_structure"][selected_function] = {
                "attributes": selected_attributes,
                "action": action
                }
        else:
            preprocessing["modify_data"][selected_function] = {
                "attributes": selected_attributes,
                "action": action
                }

    # Update the list of preprocessing functions to be applied
    choices = (list(preprocessing["modify_structure"].keys())
               + list(preprocessing["modify_data"].keys()))

    # Display preprocessing functions
    return (gr.update(choices=choices, value=choices if choices else None),
            preprocessing,
            preprocessing)


def remove_preprocessing(selection, preprocessing):
    """ Remove a preprocessing function and update the display """
    preprocessing = {
        "modify_structure": {
            k: preprocessing["modify_structure"][k]
            for k in selection
            if k in preprocessing["modify_structure"]
            },
        "modify_data": {
            k: preprocessing["modify_data"][k]
            for k in selection
            if k in preprocessing["modify_data"]
            },
        }
    choices = (list(preprocessing["modify_structure"].keys())
               + list(preprocessing["modify_data"].keys()))
    return (gr.update(choices=choices, value=choices if choices else None),
            preprocessing,
            preprocessing)


def apply_preprocessing(bid, mode, dataset,
                        preprocessing, progress=gr.Progress()):

    logging.info("DATA PREPROCESSING")
    dataset.apply_data_preprocessing(preprocessing,
                                            progress=progress)
    logging.info("METADATA PREPROCESSING")
    dataset.apply_metadata_preprocessing(preprocessing,
                                                progress=progress)
    return (run_button,
            *generate_graph(bid, mode, dataset),
            dataset.data[bid],
            dataset)


def reset_all():
    components = ["dataset_global",
            "colors_map",
            "metrics_global",
            "preprocessing_list",
            "dataset_selector",
            "data_selector",
            "gt_selector",
            "parameters_selector",
            "type_selector",
            "preprocessing_file_selector",
            "function_dropdown",
            "property_selector",
            "preprocessing_list_display",
            "preprocessing_information",
            "preprocessing_state",
            "metric_state",
            "model_information",
            "bucket_state",
            "available_buckets",
            "file_output",
            "bucket_id",
            "toggle",
            "partial_order",
            "graph_output",
            "n_sources",
            "top_n_sources",
            "preprocessing_tab",
            "model_tab",
            "tabs",
            "bucket_id",
            "prev_button",
            "next_button"]
    return [INITIAL_COMPONENT_STATES[component]
            for component in components] + [True, False]


with gr.Blocks(theme=THEME, fill_height=True) as trustfuse_demo:
    gr.Markdown("## 🌋 TrustFuse", elem_id="title")

    dataset_global = gr.State(None)
    colors_map = gr.State(None)
    metrics_global = gr.State(None)
    preprocessing_list = gr.State({
        "modify_structure": {},
        "modify_data": {}
    })
    is_reset = gr.State(False)
    cleaning_done = gr.State(False)

    with gr.Row():
        with gr.Column(scale=1):
            # Dataset selectors
            with gr.Column(scale=1):
                with gr.Tabs() as tabs:
                    with gr.Tab("Load Dataset", id=0) as dataset_tab:
                        with gr.Column():
                            choice_mode = gr.Radio(
                                ["Single file (Pickle)", "Two separate folders (Data + GT)"],
                                label="Select the dataset format to be loaded",
                                value="Single file (Pickle)"
                            )
                            available_datasets = gr.Dropdown(
                                choices=(list(STATIC_DATASETS_AVAILABLE)
                                        + list(DYNAMIC_DATASETS_AVAILABLE)),
                                label="or select an available dataset",
                                value="",
                                interactive=True
                            )
                        dataset_selector = gr.File(label="Select the pickle file",
                                                interactive=True,
                                                file_count="single",
                                                file_types=[".pkl"],
                                                height=140)

                        with gr.Row():
                            data_selector = gr.File(label="📂 Select conflicting data folder",
                                                file_count="directory",
                                                interactive=True, height=140, visible=False)
                            gt_selector = gr.File(label="📂 Select GT data folder",
                                                file_count="directory",
                                                interactive=True, height=140, visible=False)

                        parameters_selector = gr.File(label="Select the dataset parameters",
                                                file_count="single",
                                                interactive=True, height=140, visible=False)

                        type_selector = gr.File(label="Select attribute/datatatype mapping file",
                                                file_count="single",
                                                interactive=True, height=140, visible=False)
                    with gr.Tab("Preprocessing", id=1, interactive=False) as preprocessing_tab:
                        with gr.Column():
                            preprocessing_file_selector = gr.File(
                                label="You can choose a preprocessing file",
                                interactive=True,
                                file_count="single",
                                file_types=[".json"],
                                height=140)
                            function_dropdown = gr.Dropdown(PREPROCESSING_FUNCTIONS,
                                                            label="Preprocessing functions")
                            mode_selector = gr.Radio(["Property type", "Property name", "All"],
                                                    label="Property selection mode")
                            property_selector = gr.Dropdown(choices=[""],
                                                            label="Concerned properties",
                                                            multiselect=True)
                            add_button = gr.Button("Add")
                        preprocessing_list_display = gr.Dropdown(choices=[],
                                                                multiselect=True,
                                                                label="Preprocessing to be performed")
                        
                        preprocessing_information = gr.JSON(label="Selected preprocessing")
                        preprocessing_button = gr.Button("Apply preprocessing")
                        preprocessing_state = gr.Textbox(label="Preprocessing state")

                    with gr.Tab("Model", id=2, interactive=False) as model_tab:
                        metric_state = gr.State("")
                        selected_model = gr.State("CRH")
                        model_selector = gr.Dropdown(["CRH", "CATD", "KDEm", "GTM",
                                "TruthFinder", "SLIMFAST", 
                                "LTM", "ACCU"], label="🌋 Choose a fusion model")
                        model_parameters = gr.State({})
                        @gr.render(inputs=[model_selector, model_parameters])
                        def show_model_parameters(model_selector, model_params):
                            model_params.clear()
                            with gr.Accordion(label="Model parameters"):
                                gr.Markdown(f'<span style="font-size:10px;">for more details on the parameters, please refer to the {MODEL_PAPERS[model_selector]}.</span>')
                                with gr.Row():
                                    sig = inspect.signature(settings.MODEL_MAP[model_selector].__init__)
                                    for param_name, param in sig.parameters.items():

                                        def update_model_param(param, model_params=model_params, param_name=param_name):
                                            model_params[param_name] = param
                                        if isinstance(param.default, (float, int)):
                                            model_param = gr.Number(label=param_name,
                                                                    value=param.default,
                                                                    interactive=True)
                                            
                                            model_param.change(update_model_param,
                                                               inputs=[model_param],
                                                                       outputs=None)

                                        if isinstance(param.default, str):
                                            model_param = gr.Textbox(label=param_name,
                                                                     value=param.default,
                                                                     interactive=True)
        
                                            model_param.change(update_model_param,
                                                               inputs=[model_param],
                                                                       outputs=None)

                        model_information = gr.Textbox(label="Model state", interactive=False)
                        run_button = gr.Button("🚀 Run", interactive=True)

                    bucket_state = gr.State(0)
                    available_buckets = gr.State([0])
                    file_output = gr.Textbox(label="📄 Dataset",
                                            interactive=False)
                    bucket_id = gr.Dropdown(label="Current bucket",
                                            choices=["Bucket #0"],
                                            interactive=False, value="")
                
                    toggle = gr.Radio(["Input", "Ground Truth"],
                                    label="Display", value="Input")
                    with gr.Column():
                        partial = gr.Markdown("#### 🎯 Partial orders")
                        partial_order = gr.HTML()
                    reset_button = gr.Button("Reset")
        with gr.Column(scale=2) as vis_column:
            with gr.Tabs():
                with gr.Tab("Graph"):
                    graph_output = gr.HTML()
                with gr.Tab("Table"):
                    table_output = gr.Dataframe(label="📋 Data loaded")
                with gr.Tab("Results"):
                    metric_output = gr.HTML()
                with gr.Tab("Sources"):
                    n_sources = gr.Textbox(value="",
                                        label="Enter the Top N of desired sources",
                                        visible=False)
                    top_n_sources = gr.Dataframe(headers=["Source", "Score"], visible=False)

            with gr.Row():
                prev_button = gr.Button("⬅ Previous bucket", interactive=False)
                next_button = gr.Button("Next bucket ➡", interactive=False)

        reset_button.click(reset_all, inputs=None, outputs=[
            dataset_global,
            colors_map,
            metrics_global,
            preprocessing_list,
            dataset_selector,
            data_selector,
            gt_selector,
            parameters_selector,
            type_selector,
            preprocessing_file_selector,
            function_dropdown,
            property_selector,
            preprocessing_list_display,
            preprocessing_information,
            preprocessing_state,
            metric_state,
            model_information,
            bucket_state,
            available_buckets,
            file_output,
            bucket_id,
            toggle,
            partial_order,
            graph_output,
            n_sources,
            top_n_sources,
            preprocessing_tab,
            model_tab,
            tabs,
            bucket_id,
            prev_button,
            next_button,
            is_reset,
            cleaning_done
        ])

        preprocessing_file_selector.change(init_preprocessing_list,
                                           inputs=[preprocessing_file_selector, is_reset],
                                           outputs=[preprocessing_information, preprocessing_list])

        mode_selector.change(update_property_selection_mode,
                             inputs=[mode_selector, dataset_global],
                             outputs=[property_selector])
        add_button.click(add_preprocessing,
                         inputs=[function_dropdown,
                                 mode_selector,
                                 property_selector,
                                 preprocessing_list],
                         outputs=[preprocessing_list_display,
                                  preprocessing_information,
                                  preprocessing_list])

        # To remove un preprocessing function
        preprocessing_list_display.input(remove_preprocessing,
                                        inputs=[preprocessing_list_display,
                                                preprocessing_list],
                                        outputs=[preprocessing_list_display,
                                                 preprocessing_information,
                                                 preprocessing_list])

        preprocessing_button.click(apply_preprocessing,
                                   inputs=[bucket_state,
                                           toggle,
                                           dataset_global,
                                           preprocessing_list],
                                   outputs=[preprocessing_state,
                                            graph_output,
                                            colors_map,
                                            table_output,
                                            dataset_global])

        n_sources.change(fn=top_n,
                         inputs=[n_sources, bucket_state, dataset_global, is_reset],
                         outputs=[top_n_sources])
        prev_button.click(prev_dataframe,
                          inputs=[bucket_state,
                                  toggle,
                                  n_sources,
                                  dataset_global,
                                  is_reset],
                          outputs=[prev_button,
                                   next_button,
                                   bucket_state,
                                   bucket_id,
                                   table_output,
                                   graph_output,
                                   colors_map,
                                   partial_order,
                                   top_n_sources])

        next_button.click(next_dataframe,
                          inputs=[bucket_state,
                                  toggle,
                                  n_sources,
                                  dataset_global,
                                  is_reset],
                          outputs=[next_button,
                                   prev_button,
                                   bucket_state,
                                   bucket_id,
                                   table_output,
                                   graph_output,
                                   colors_map,
                                   partial_order,
                                   top_n_sources])

        toggle.change(toggle_display,
                      inputs=[toggle,
                              bucket_state,
                              dataset_global,
                              is_reset],
                      outputs=[table_output,
                               graph_output,
                               colors_map])

        available_datasets.change(load_available_dataset,
                                  inputs=[available_datasets, toggle, is_reset, cleaning_done],
                                  outputs=[table_output,
                                         file_output,
                                         graph_output,
                                         colors_map,
                                         partial_order,
                                         bucket_id,
                                         dataset_global,
                                         preprocessing_tab,
                                         model_tab,
                                         prev_button,
                                         next_button,
                                         bucket_id,
                                         is_reset,
                                         cleaning_done])

        file_output.change(display_dataset,
                           inputs=[toggle, dataset_global, is_reset, cleaning_done],
                           outputs=[table_output,
                                   graph_output,
                                   colors_map,
                                   partial_order,
                                   bucket_id])

        dataset_selector.change(load_dataset,
                                inputs=[dataset_selector,
                                        toggle,
                                        is_reset,
                                        cleaning_done],
                                outputs=[table_output,
                                         file_output,
                                         graph_output,
                                         colors_map,
                                         partial_order,
                                         bucket_id,
                                         dataset_global,
                                         preprocessing_tab,
                                         model_tab,
                                         prev_button,
                                         next_button,
                                         bucket_id,
                                         is_reset,
                                         cleaning_done])

        model_information.change(display_model_output,
                                 inputs=[is_reset],
                                 outputs=[toggle,
                                  n_sources])

        metric_state.change(display_metrics_gr,
                                 inputs=[metric_state],
                                 outputs=[metric_output])
        # Model events
        run_button.click(apply_model,
                         inputs=[table_output,
                                 model_selector,
                                 dataset_global,
                                 model_parameters],
                         outputs=[model_information,
                                  available_buckets,
                                  metric_state,
                                  dataset_global,
                                  metrics_global])

        available_buckets.change(update_bucket_id,
                                 inputs=[dataset_global, is_reset],
                                 outputs=[bucket_id])

        bucket_id.input(update_dataframe_from_dropdown,
                        inputs=[bucket_id,
                                toggle,
                                n_sources,
                                dataset_global,
                                is_reset],
                        outputs=[bucket_state,
                                 table_output,
                                 graph_output,
                                 colors_map,
                                 partial_order,
                                 top_n_sources,
                                 prev_button,
                                 next_button])

        choice_mode.change(update_visibility,
                           inputs=[choice_mode, is_reset],
                           outputs=[dataset_selector,
                            data_selector,
                            gt_selector])
        
        data_selector.change(update_selector_visibility,
                             inputs=[is_reset],
                             outputs=[gt_selector])
        gt_selector.change(update_selector_visibility,
                           inputs=[is_reset],
                           outputs=[parameters_selector])
        gt_selector.change(update_selector_visibility,
                           inputs=[is_reset],
                           outputs=[type_selector])

        type_selector.change(load_dataset_from_folders,
                             inputs=[data_selector,
                                     gt_selector,
                                     type_selector,
                                     parameters_selector,
                                     toggle,
                                     is_reset,
                                     cleaning_done],
                            outputs=[table_output,
                                     file_output,
                                     graph_output,
                                     colors_map,
                                     partial_order,
                                     bucket_id,
                                     dataset_global,
                                     preprocessing_tab,
                                     model_tab,
                                     prev_button,
                                     next_button,
                                     bucket_id,
                                     is_reset,
                                     cleaning_done])

trustfuse_demo.queue().launch()
