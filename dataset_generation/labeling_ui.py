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

import copy
import os
import pickle
import re

import gradio as gr
import pandas as pd


def is_qid(val):
    """Check if the val is a Wikidata QID"""
    return bool(re.fullmatch(r'Q\d+', str(val)))


def convert_coordinate_selection_v0(selection):
    """Convert back the coordinate values of the Gradio format"""
    coordinates = [tuple(map(float, s.split(","))) for s in selection]
    return coordinates


def convert_coordinate_selection(selection):
    """Convert back the coordinate values of the Gradio format"""
    coordinates = []
    for s in selection:
        s_clean = s.strip("()")
        lat, lon = map(float, s_clean.split(","))
        coordinates.append(((lat, lon),)*2) # same format as at the beginning
    return coordinates

# Relative path to open static files
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Categories to be labeled
PATH_MAPPING = {
    "Monuments of Paris": os.path.join(BASE_DIR, "files", "buckets_by_qid_paris.pkl"),
    "Football cup 2022": os.path.join(BASE_DIR, "files", "buckets_by_qid_foot.pkl"),
    "Machine Learning": os.path.join(BASE_DIR, "files", "buckets_by_qid_ml.pkl"),
    "filters": os.path.join(BASE_DIR, "files", "filters.pkl"),
    "auto_labeled_data": os.path.join(BASE_DIR, "files", "auto_labeled_data.pkl"),
    "latest_values": os.path.join(BASE_DIR, "files", "latest_values.pkl"),
    "constrained_properties": os.path.join(BASE_DIR, "files", "constrained_properties.pkl"),
    "attribute_types": os.path.join(BASE_DIR, "files", "attribute_types.pkl")
}

buckets_by_qid = None
latest_values = None
constrained_properties = None
attribute_types = None
auto_labeled_data = None
evolving_props = []
static_props = []
filters = []
partial_ordering = {}
# To facilitate labeling, we save all possible values for each property
values_by_prop = {}
buckets = None
pass_state = 1
buckets_intervals = {}
# Dictionary containing all values already seen for each property
# depending on the bucket, bucket i includes all the values in bucket i-1
seen_values = None

props_mapping = None
qid_name_mapping = {}

"""
for each QID/prop:
- feature = [static, evolving, filter]
- values = [val1, val2, ...]
- partial_orders = [[...]]
"""
# Record all inputs from Gradio selections
props_information = {}
props_mapping_second_pass = {}

### Dataset loading functions
def load_category(category):
    """Load dataset in pickle format as described in the doc"""

    global buckets_by_qid
    global latest_values
    global constrained_properties
    global attribute_types
    global filters
    global auto_labeled_data


    with open(PATH_MAPPING[category], "rb") as f:
        buckets_by_qid = pickle.load(f)
    # Get latest values to support labeling decision making
    with open(PATH_MAPPING["latest_values"], 'rb') as f:
        latest_values = pickle.load(f)
    # Support labeling by indicating if a property is multi-valued
    with open(PATH_MAPPING["constrained_properties"], 'rb') as f:
        constrained_properties = pickle.load(f)
    # To display or not the QID of entities as values
    with open(PATH_MAPPING["attribute_types"], 'rb') as f:
        attribute_types = pickle.load(f)
    # To filter the properties (media properties, st. images, audio)
    filters = []
    try:
        with open(PATH_MAPPING["filters"], 'rb') as f:
            filters =  pickle.load(f)
    except FileNotFoundError:
        print("The file does not exist.")

    with open(PATH_MAPPING["auto_labeled_data"], "rb") as f:
        auto_labeled_data = pickle.load(f)

    for qid, properties in auto_labeled_data.items():
        props_information[qid] = {}
        for prop, values in properties.items():
            props_information[qid][prop]= {
                "values": values,
                "feature": "",
                "orders": [],
                "buckets": {}
            }


def prepare_data(qid):
    """Prepare data to be labeled once the qid is selected

    Args:
        qid (str): Wikidata QID

    Returns:
        Gradio components
    """

    global partial_ordering
    global seen_values
    global values_by_prop
    global props_mapping
    global buckets
    global buckets_intervals

    # Initalize partial orders
    partial_ordering[qid] = {
        "qid": {},
        "value": {}
    }

    # Load data to be labeled (i.e., the buckets)
    buckets = buckets_by_qid[qid]['buckets']
    # All seen values by buckets B#i contains values of B#i-1
    seen_values = {i: {} for i in range(len(buckets))}
    for i, bucket in enumerate(buckets):
        buckets_intervals[i] = [bucket["start_time"], bucket["end_time"]]
        if i-1 in seen_values:
            seen_values[i].update(copy.deepcopy(seen_values[i-1]))
        for contrib in bucket['only-bucket']:
            prop = contrib[1]
            value = contrib[2]
            value_id = contrib[3]
            if value_id not in qid_name_mapping:
                qid_name_mapping[value_id] = value
            # will be manipulated at the input preparation for fusion models
            if prop not in filters:
                if prop not in seen_values[i]:
                    seen_values[i][prop] = set()
                seen_values[i][prop].add((value, value_id))
                if prop not in values_by_prop:
                    values_by_prop[prop] = set([(value, value_id)])
                else:
                    values_by_prop[prop].add((value, value_id))
    # Create an index for each property
    props_mapping = dict(enumerate(list(values_by_prop.keys())))
    # Select all the possible values for the current property
    choices = [(v[0], v[1]) for v in values_by_prop[props_mapping[0]]]
    # Prefill the choices with the current selected values or the values from the auto labeling
    values = props_information[qid][props_mapping[0]]["values"]
    # Adapt the format if the data are coordinate
    if props_mapping[0] in attribute_types and attribute_types[props_mapping[0]] == "Coordinate":
        # v[0][0] == lat; v[0][1] = long; v[0] = (lat, long)
        choices = [f"({v[0][0]}, {v[0][1]})" for v in values_by_prop[props_mapping[0]]]
        values = [f"({v[0][0]}, {v[0][1]})" for v in values]
    elif props_mapping[0] in attribute_types and attribute_types[props_mapping[0]] == "Entity":
        choices = [(f"{v[0]} ({v[1]})", v[1]) for v in values_by_prop[props_mapping[0]]]
        values = [v[1] for v in values]
    else:
        values = [v[1] for v in values]
    links = {
        v[0]: f"[{v[0]} ({v[1]})](<https://www.wikidata.org/wiki/{v[1]}>)"
        for v in values_by_prop[props_mapping[0]] if is_qid(v[1])
    }

    return (gr.update(label=f"1 / {len(props_mapping)}", value=props_mapping[0]),
            gr.update(choices=choices, # correct_values
                      value=values),
            "### Wikidata entities<br>" + "<br>" \
            .join([links[c] for c in choices if c in links]))


def change_correct_values(correct_values, qid, prop, index):
    # pass_state == 1 --> Static properties
    # pass_state == 2 --> Evolving properties
    if pass_state == 1:
        if props_mapping[index] in attribute_types \
            and attribute_types[props_mapping[index]] == "Coordinate":
            props_information[qid][prop]["values"] = \
                convert_coordinate_selection(correct_values)
        else:
            props_information[qid][prop]["values"] = [(qid_name_mapping[val], val)
                                                      for val in correct_values]
    if pass_state == 2:
        bid = props_mapping_second_pass[index][0]
        props_information[qid][prop]["buckets"][bid]["values"] = correct_values
        if prop in attribute_types and attribute_types[prop] == "Coordinate":
            props_information[qid][prop]["buckets"][bid]["values"] = \
                convert_coordinate_selection(correct_values)
        else:
            props_information[qid][prop]["buckets"][bid]["values"] = [(qid_name_mapping[val], val)
                                                                      for val in correct_values]


def update_prop(index, qid):
    """Update Gradio components when the property changes

    Args:
        index (int): index of the property
        qid (str): Wikidata QID

    Returns:
        Gradio components
    """
    global props_mapping
    global values_by_prop
    visible = False
    # If static properties pass
    if pass_state == 1:
        choices = [v[0] for v in values_by_prop[props_mapping[index]]]
        values = props_information[qid][props_mapping[index]]["values"]
        if props_mapping[index] in attribute_types \
            and attribute_types[props_mapping[index]] == "Coordinate":
            choices = [f"({v[0][0]}, {v[0][1]})" \
                       for v in values_by_prop[props_mapping[index]]]
            values = [f"({v[0][0]}, {v[0][1]})" for v in values]
        elif props_mapping[index] in attribute_types \
            and attribute_types[props_mapping[index]] == "Entity":
            choices = [(f"{v[0]} ({v[1]})", v[1]) \
                       for v in values_by_prop[props_mapping[index]]]
            values = [v[1] for v in values]
        else:
            values = [v[1] for v in values]

        feature = props_information[qid][props_mapping[index]]["feature"]
        if feature == "Static":
            visible = True
        links = {
            v[0]: f"[{v[0]} ({v[1]})](<https://www.wikidata.org/wiki/{v[1]}>)"
            for v in values_by_prop[props_mapping[index]] if is_qid(v[1])
        }
        return (gr.update(label=f"{index+1} / {len(props_mapping)}",
                          value=props_mapping[index]), # prop_name
                gr.update(choices=choices, # correct_values
                        value=values,
                        visible=visible,
                        interactive=True),
                gr.update(value=feature), # prop_feature
                gr.update(value="### Wikidata entities<br>" + "<br>" \
                        .join(links.values()), visible=visible), # wikidata_links
                props_information[qid][props_mapping[index]]["orders"], # orders
                gr.update(visible=visible), # add_partial_order_button
                gr.update(visible=visible), # partial_order_text
                gr.update(visible=False) # time_intervals
                )

    # If evolving properties pass
    if pass_state == 2:
        bid = props_mapping_second_pass[index][0]
        prop = props_mapping_second_pass[index][1]
        choices = [sv[0] for sv in seen_values[bid][prop]]
        values = props_information[qid][prop]["buckets"][bid]["values"]
        if prop in attribute_types and attribute_types[prop] == "Coordinate":
            choices = [f"({sv[0][0]}, {sv[0][1]})" for sv in seen_values[bid][prop]]
            values = [f"({sv[0][0]}, {sv[0][1]})" for sv in values]
        elif prop in attribute_types and attribute_types[prop] == "Entity":
            choices = [(f"{sv[0]} ({sv[1]})", sv[1]) for sv in seen_values[bid][prop]]
            values = [sv[1] for sv in values]
        else:
            values = [v[1] for v in values]
        links = {
            v[0]: f"[{v[0]} ({v[1]})](<https://www.wikidata.org/wiki/{v[1]}>)"
            for v in seen_values[bid][prop] if is_qid(v[1])
        }
        return (gr.update(label=f"{index+1} / {len(props_mapping_second_pass)}", value=prop), # prop_name
                gr.update(choices=choices, # correct_values
                        value=values,
                        visible=True,
                        interactive=True),
                gr.update(visible=False), # prop_feature
                gr.update(value="### Wikidata entities<br>" + "<br>" \
                        .join(links.values()), visible=True), # wikidata_links
                props_information[qid][prop]["buckets"][bid]["orders"], # orders
                gr.update(visible=True), # add_partial_order_button
                gr.update(visible=True), # partial_order_text
                gr.update(value=(f"### Start time = {buckets_intervals[bid][0]}"
                                 f" - End time = {buckets_intervals[bid][1]}"),
                                 visible=True) # time_intervals
                )


def prev_prop(index, qid):
    """Back to the previous property

    Args:
        index (int): index of the property
        qid (str): Wikidata QID

    Returns:
        Gradio components
    """
    #global props_mapping
    if pass_state == 1:
        index = (index - 1) % len(props_mapping)
        return index, *update_prop(index, qid)
    if pass_state == 2:
        index = (index - 1) % len(props_mapping_second_pass)
        return index, *update_prop(index, qid)


def next_prop(index, qid, orders):
    """Go to the next property

    Args:
        index (int): index of the property
        qid (str): Wikidata QID

    Returns:
        Gradio components
    """
    #global props_mapping
    if pass_state == 1:
        props_information[qid][props_mapping[index]]["orders"] = copy.deepcopy(orders)
        index = (index + 1) % len(props_mapping)
        return index, *update_prop(index, qid)
    if pass_state == 2:
        bid = props_mapping_second_pass[index][0]
        prop = props_mapping_second_pass[index][1]
        props_information[qid][prop]["buckets"][bid]["orders"] = copy.deepcopy(orders)
        index = (index + 1) % len(props_mapping_second_pass)
        return index, *update_prop(index, qid)


def add_order(orders):
    """To add a partial order"""
    return orders + [[{"values": []}, {"values": []}]]


def classify_prop(feature, prop, qid, index):
    """Classify prop only for the first labeling pass"""
    if feature == "Static":
        props_information[qid][prop]["feature"] = "Static"
        static_props.append(prop)
        if prop in filters:
            filters.remove(prop)
        if prop in evolving_props:
            evolving_props.remove(prop)
        return (index,
                prop,
                gr.update(visible=True), # correct_values
                gr.update(), # prop_feature
                gr.update(visible=True), # partial_order_text
                gr.update(visible=True), # add_partial_order_button
                gr.update(visible=True)) # links
    if feature == "Evolving":
        props_information[qid][prop]["feature"] = "Evolving"
        evolving_props.append(prop)
        if prop in filters:
            filters.remove(prop)
        if prop in static_props:
            static_props.remove(prop)
        return (index,
                prop,
                gr.update(visible=False),
                gr.update(),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False)) # links
    if feature == "Filter":
        props_information[qid][prop]["feature"] = "Filter"
        if prop in evolving_props:
            evolving_props.remove(prop)
        if prop in static_props:
            static_props.remove(prop)
        filters.append(prop)
        return (index,
                prop,
                gr.update(visible=False), # correct_values
                gr.update(), # prop_feature
                gr.update(visible=False), # partial_order_text
                gr.update(visible=False), # add_partial_order_button
                gr.update(visible=False)) # links


def valid_first_pass(qid):
    """Prepare data for the 2nd pass of the labeling (i.e., evolving properties)

    Args:
        qid (str): Wikidata QID

    Returns:
        Gradio components
    """
    global props_mapping_second_pass
    global pass_state
    global evolving_props
    pass_state = 2

    # Sort the evolving properties in a list
    evolving_props = []
    for p in props_information[qid]:
        if props_information[qid][p]["feature"] == "Evolving":
            evolving_props.append(p)

    if evolving_props:
        mapping_index = 0
        for prop in evolving_props:
            for i in seen_values:
                if prop in seen_values[i]:
                    # i : BID
                    props_mapping_second_pass[mapping_index] = [i, prop]
                    mapping_index += 1

        bid = props_mapping_second_pass[0][0]
        prop = props_mapping_second_pass[0][1]
        # Prefilled values for the buckets in props_information
        for _, bid_prop in props_mapping_second_pass.items():
            if bid_prop[0] not in props_information[qid][bid_prop[1]]["buckets"]:
                props_information[qid][bid_prop[1]]["buckets"][bid_prop[0]] = {
                    "values": [],
                    "orders": []
                }
            prefilled_values = []
            for val in props_information[qid][props_mapping[0]]["values"]:
                if val in seen_values[bid_prop[0]][bid_prop[1]]:
                    prefilled_values.append(val)
            props_information[qid][bid_prop[1]]["buckets"][bid_prop[0]]["values"] = prefilled_values

        choices = [sv[0] for sv in seen_values[bid][prop]]
        values = props_information[qid][prop]["buckets"][bid]["values"]
        if prop in attribute_types and attribute_types[prop] == "Coordinate":
            choices = [f"({sv[0][0]}, {sv[0][1]})" for sv in seen_values[bid][prop]]
            values = [f"({sv[0][0]}, {sv[0][1]})" for sv in values]
        elif prop in attribute_types and attribute_types[prop] == "Entity":
            choices = [(f"{v[0]} ({v[1]})", v[1]) for v in seen_values[bid][prop]]
            values = [v[1] for v in values]
        else:
            values = [v[1] for v in values]

        links = {
            v[0]: f"[{v[0]} ({v[1]})](<https://www.wikidata.org/wiki/{v[1]}>)"
            for v in seen_values[bid][prop] if is_qid(v[1])
        }

        return (
                gr.update(visible=False), # valid_button
                gr.update(visible=False), # prop_feature
                gr.update(label=f"1 / {len(props_mapping_second_pass)}", # prop_name
                        value=props_mapping_second_pass[0][1]),
                gr.update(choices=choices, value=values, visible=True), # correct_values
                gr.update(value="### Wikidata entities<br>" + "<br>" \
                                 .join(links.values()),
                                 visible=True), # Wikidata links
                props_information[qid][prop]["buckets"][bid]["orders"], # orders
                gr.update(visible=True), # add_partial_order_button
                gr.update(visible=True), # partial_order_text
                gr.update(value=(f"### Start time = {buckets_intervals[bid][0]}"
                                 f" - End time = {buckets_intervals[bid][1]}"),
                                 visible=True), # time_intervals
                0,
                gr.update(visible=True) # save_button
                )
    else:
        return (
            "Finish", # valid_button
            gr.update(visible=False), # prop_feature
            gr.update(visible=False), # prop_name
            gr.update(visible=False), # correct_values
            gr.update(visible=False), # wikidata links
            gr.update(visible=False), # orders
            gr.update(visible=False), # add_partial_order_button
            gr.update(visible=False), # partial_order_text
            gr.update(visible=False), # time_intervals
            0, # index_state
            gr.update(visible=True,
                      interactive=True)
        )


def save_labeling(qid):
    # We define a dictionary that will be used to create all the
    # dataframes representing GT for buckets
    static_props_per_bucket = {}
    for i, bucket in enumerate(buckets):
        static_props_per_bucket[i] = set()
        for triple in bucket["only-bucket"]:
            if triple[1] in static_props:
                static_props_per_bucket[i].add(triple[1])

    buckets_df = {}
    for i in range(len(buckets)):
        buckets_df[i] = {}
        bucket_claims = {}
        bucket_qids = {}
        for prop in static_props_per_bucket[i]:
            gt_values = []
            gt_qids = []
            labeled_values = props_information[qid][prop]["values"]
            if labeled_values is not None:
                gt_values = [[val[0] for val in labeled_values \
                              if val in seen_values[i][prop]]]
                gt_qids = [[val[1] for val in labeled_values \
                              if val in seen_values[i][prop]]]
                if len(gt_values) == 0:
                    gt_values = [[None]]
                    gt_qids = [[None]]
            else:
                gt_values = [[None]]
                gt_qids = [[None]]
            bucket_claims[prop] = gt_values
            bucket_qids[prop] = gt_qids
        for prop in evolving_props:
            if i in props_information[qid][prop]["buckets"]:
                values = props_information[qid][prop]["buckets"][i]["values"]
                if values is not None:
                    bucket_claims[prop] = [[val[0] for val in values]]
                    bucket_qids[prop] = [[val[1] for val in values]]
                else:
                    bucket_claims[prop] = [[None]]
                    bucket_qids[prop] = [[None]]

        bucket_claims["Entity"] = [qid]
        bucket_qids["Entity"] = [qid]

        # Claims
        bucket_claims_df = pd.DataFrame(bucket_claims)
        bucket_qids_df = pd.DataFrame(bucket_qids)
        buckets_df[i]["GT"] = {}
        buckets_df[i]["GT"]["value"] = bucket_claims_df
        buckets_df[i]["GT"]["qid"] = bucket_qids_df

        partial_order_per_prop = {"qid": {}, "value": {}}
        for prop in props_information[qid]:
            partial_order_per_prop["qid"][prop] = []
            partial_order_per_prop["value"][prop] = []
            bucket_orders = []
            if i in props_information[qid][prop]["buckets"]:
                bucket_orders = props_information[qid][prop]["buckets"][i]["orders"]
            else:
                bucket_orders = props_information[qid][prop]["orders"]
            for order in bucket_orders:
                preprocessed_order_qid = []
                preprocessed_order_val = []
                for level in order:
                    preprocessed_lvl_qid = []
                    preprocessed_lvl_val = []
                    for val in level["values"]:
                        preprocessed_lvl_qid.append(val[1])
                        preprocessed_lvl_val.append(val[0])
                    preprocessed_order_qid.append(preprocessed_lvl_qid)
                    preprocessed_order_val.append(preprocessed_lvl_val)
                partial_order_per_prop["qid"][prop].append(preprocessed_order_qid)
                partial_order_per_prop["value"][prop].append(preprocessed_order_val)

        bucket_partial_ordering_value = partial_order_per_prop["value"]
        bucket_partial_ordering_value.update(partial_ordering[qid]["value"])
        bucket_partial_ordering_qid = partial_order_per_prop["qid"]
        bucket_partial_ordering_qid.update(partial_ordering[qid]["qid"])
        buckets_df[i]["GT"]["value_order"] = bucket_partial_ordering_value
        buckets_df[i]["GT"]["qid_order"] = bucket_partial_ordering_qid

        index = i
        bucket_mapping = buckets[index]['only-bucket']
        distinct_properties = set()
        # Dict containing contributions from each source
        # in the form of attribute/value pairs
        claims_by_source = {}
        # We iterate on the claims
        for claim in bucket_mapping:
            distinct_properties.add(claim[1])
            if claim[0] not in claims_by_source:
                claims_by_source[claim[0]] = {}
            if claim[1] not in claims_by_source[claim[0]]:
                claims_by_source[claim[0]][claim[1]] = []
            # claim[2] = label ; claim[3] = QID
            claims_by_source[claim[0]][claim[1]].append((claim[2], claim[3]))
        # Pre-fill the dataframe template with column names
        bucket_claims_value = {prop: [] for prop in distinct_properties}
        bucket_claims_qid = {prop: [] for prop in distinct_properties}
        bucket_claims_value['Source'] = []
        bucket_claims_qid['Source'] = []
        for source, claims in claims_by_source.items():
            max_attribute_values = max(len(v) for v in claims.values())
            bucket_claims_value['Source'].extend([source] * max_attribute_values)
            bucket_claims_qid['Source'].extend([source] * max_attribute_values)
            for prop in distinct_properties:
                if prop in claims:
                    for val in claims[prop]:
                        bucket_claims_value[prop].append(val[0])
                        bucket_claims_qid[prop].append(val[1])
                    # Complete with None to create the Dataframe correctly
                    bucket_claims_value[prop].extend(
                        [None for _ in range(max_attribute_values \
                                                - len(claims[prop]))]
                        )
                    bucket_claims_qid[prop].extend(
                        [None for _ in range(max_attribute_values \
                                                - len(claims[prop]))]
                        )
                else:
                    # We fill the prop with None if the source has provided no value for it
                    bucket_claims_value[prop].extend([None for _ in range(max_attribute_values)])
                    bucket_claims_qid[prop].extend([None for _ in range(max_attribute_values)])
        bucket_claims_value['Entity'] = [qid for _ in bucket_claims_value['Source']]
        bucket_claims_qid['Entity'] = [qid for _ in bucket_claims_value['Source']]
        buckets_df[i]["data"] = {}
        buckets_df[i]["data"]["value"] = pd.DataFrame(bucket_claims_value)
        buckets_df[i]["data"]["qid"] = pd.DataFrame(bucket_claims_qid)

    with open(os.path.join(BASE_DIR, "data", f"{qid}_gt.pkl"), "wb") as f:
        pickle.dump(buckets_df, f)
    with open(PATH_MAPPING["filters"], "wb") as f:
        pickle.dump(filters, f)


css = """
#specificity {max-width: 50px}
"""

with gr.Blocks(fill_height=True, css=css) as demo:
    gr.Markdown("## Labeling", elem_id="title")

    with gr.Column(scale=1):
        # To browse properties
        index_state = gr.State(0)
        category_selector = gr.Dropdown(label="Select the category",
                                        choices=["Monuments of Paris",
                                                 "Football cup 2022",
                                                 "Machine Learning"],
                                        value="")
        qid_to_label = gr.Textbox(label="QID", interactive=True)
        prop_name = gr.Textbox(label="", interactive=False)
        prop_feature = gr.Radio(["Static", "Evolving", "Filter"],
                                value=None, container=False)
        time_intervals = gr.Markdown(visible=False)

        with gr.Row():
            correct_values = gr.CheckboxGroup(choices=[],
                                            label="Select correct values",
                                            visible=False)
            wikidata_links = gr.Markdown(visible=False)

        partial_order_text= gr.HTML(("<h3>Specify partial orders "
                                     "of specificity (SPO) if needed</h3>"),
                visible=False)
        orders = gr.State([])
        add_partial_order_button = gr.Button("Add SPO", visible=False)

        prop_feature.input(classify_prop,
                            inputs=[prop_feature, prop_name, qid_to_label, index_state],
                            outputs=[index_state, prop_name,
                                     correct_values, prop_feature,
                                     partial_order_text,
                                     add_partial_order_button,
                                     wikidata_links])

        # Partial orders component
        @gr.render(inputs=[orders, correct_values, prop_name])
        def orders_display(orders_state, selected_values, prop):
            """Orders display"""
            for order in orders_state:
                with gr.Row():
                    choices = selected_values
                    if prop in attribute_types and attribute_types[prop] == "Entity":
                        choices = [
                            (f"{qid_name_mapping[selected_val]} ({selected_val})", selected_val)
                            for selected_val in selected_values
                            ]
                    for nb_lvl, level in enumerate(order):
                        if prop in attribute_types \
                            and attribute_types[prop] == "Coordinate":
                                values = [f"({v[0][0]}, {v[0][1]})" for v in level["values"]]
                        elif prop in attribute_types \
                            and attribute_types[prop] == "Entity":
                            values = [v[1] for v in level["values"]]
                        else:
                            values = [v[1] for v in level["values"]]

                        level_dropdown = gr.Dropdown(value=values,
                                                        choices=choices,
                                                        multiselect=True,
                                                        container=None)
                        if nb_lvl < len(order) - 1:
                            gr.HTML("<span style='display: flex; justify-content: center; text-align: center; font-size: 25px'>&gt;</span>",
                                    elem_id="specificity")
                        def update_level_values(level_values, level=level):
                            # Post-preprocessing of selected values
                            if prop in attribute_types \
                                and attribute_types[prop] == "Coordinate":
                                level["values"] = \
                                    convert_coordinate_selection(level_values)
                            else:
                                level["values"] = [(qid_name_mapping[val], val)
                                                                        for val in level_values]
                        level_dropdown.change(update_level_values,
                                              inputs=[level_dropdown],
                                              outputs=None)
                    with gr.Column():
                        order_remove_button = gr.Button(value="Remove order", variant="stop")
                        add_level_button = gr.Button(value="Add level", variant="huggingface")
                        def add_level(order=order):
                            order.append({"values": []})
                            return orders_state
                        add_level_button.click(add_level, inputs=None, outputs=[orders])
                        def delete_order(order=order):
                            orders_state.remove(order)
                            return orders_state
                        order_remove_button.click(delete_order, inputs=None, outputs=[orders])

        with gr.Row():
            prev_btn = gr.Button("⬅️ Previous")
            next_btn = gr.Button("Next ➡️")
        valid_button = gr.Button("Valid the first pass")
        save_button = gr.Button("Save this entity", interactive=True, visible=False)

    save_button.click(save_labeling, inputs=[qid_to_label], outputs=[])

    valid_button.click(valid_first_pass,
                       inputs=[qid_to_label],
                       outputs=[valid_button, prop_feature, prop_name,
                                correct_values, wikidata_links, orders,
                                add_partial_order_button, partial_order_text,
                                time_intervals, index_state, save_button])

    prev_btn.click(prev_prop,
                   inputs=[index_state, qid_to_label],
                   outputs=[index_state, prop_name, correct_values, prop_feature,
                            wikidata_links, orders, add_partial_order_button,
                            partial_order_text, time_intervals])

    next_btn.click(next_prop,
                   inputs=[index_state, qid_to_label, orders],
                   outputs=[index_state, prop_name, correct_values, prop_feature,
                            wikidata_links, orders, add_partial_order_button,
                            partial_order_text, time_intervals])

    category_selector.change(load_category,
                                inputs=[category_selector],
                                outputs=[])
    qid_to_label.submit(prepare_data,
                        inputs=[qid_to_label],
                        outputs=[prop_name, correct_values, wikidata_links])

    correct_values.input(change_correct_values,
                          inputs=[correct_values, qid_to_label, prop_name, index_state],
                          outputs=[])

    add_partial_order_button.click(add_order,
                                   inputs=[orders],
                                   outputs=[orders])

demo.launch()
