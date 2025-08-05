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

import argparse
import pickle
import os
import ast

import pandas as pd
import tqdm

import utils


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",
                        help="Dataset containing conflicting data",
                        required=True)
    parser.add_argument("--fusion-data",
                        dest="fusion_data",
                        help="Folder path that will contain data for fusion models",
                        required=True)
    parser.add_argument("--revisions-folder",
                        dest="revisions_folder",
                        required=True)
    parser.add_argument("--property-labels",
                        dest="property_labels",
                        required=True)
    parser.add_argument("--constrained-properties",
                        dest="constrained_properties",
                        required=True)
    args = parser.parse_args()

    property_labels = pickle.load(open(args.property_labels, 'rb'))
    constrained_properties = {}

    for constraint_file in tqdm.tqdm(os.listdir(args.constrained_properties)):
        file_path = os.path.join(args.constrained_properties, constraint_file)
        constrained_properties_df = pd.read_csv(file_path)
        constrained_properties[constraint_file.replace('.csv', '')] = \
        [property_labels[prop_url.replace('http://www.wikidata.org/entity/', '')] \
         for prop_url in constrained_properties_df['pid'].values \
            if prop_url.replace('http://www.wikidata.org/entity/', '') in property_labels]

    # "Dynamic" dictionary, since it will be filled and emptied when buckets will be generated
    dynamic_modifications = {}
    # Columns to be filtered for experimentation
    columns_to_filter = ['label', 'description', 'image', 'ID', 'URL', 'Unnamed', 
                         'source', 'place name sign', 'signature', 'audio', 'view',
                         'banner', 'Identifier', 'identifier', 'video', 'article',
                         '3D model', 'commons', 'Common', 'Commons', 'common',
                         'category', 'Category']
    latest_values = {}
    # For stats purpose
    distinct_sources = set()
    # Read the folder of CSV files containing the revisions for each entity in a Wikipedia category
    files = os.listdir(args.revisions_folder)
    # Iterate files one by one
    for file in tqdm.tqdm(files):
        # Stats files are ignored
        if ('Q243.csv' in file or 'Q64436.csv' in file or 'Q517.csv' in file or 'Q15155995.csv') and 'stats' not in file:
            modifications_by_timestamp = {}
            # Concatenate path to file
            file_path = os.path.join(args.revisions_folder, file)
            qid_df = pd.read_csv(file_path, index_col=False)
            # Extract the QID from the name of the file
            qid = file_path.replace(".csv", "")
            qid = qid.replace(args.revisions_folder, "")
            latest_values[qid] = {}
            # Modifications to the “QID” entity are initialized with an empty dictionary
            dynamic_modifications[qid] = {}
            # List of entity property labels
            prop_names = list(qid_df.columns)
            # remove the irrelevant columns/properties
            prop_filtering = prop_names.copy()
            for col in prop_filtering:
                for col_to_filter in columns_to_filter:
                    if col_to_filter in col and col in prop_names:
                        # Attributes added temporally
                        if col not in ['label_en', 'label_fr', 'description_en', 'description_fr']:
                            prop_names.remove(col)
            for prop_name in prop_names:
                # A modification dictionary is used to temporally reproduce the enrichment of a Wikidata entity.
                dynamic_modifications[qid][prop_name] = {}
                # Retrieves all non-zero values of the property concerned
                values = qid_df[pd.notnull(qid_df[prop_name])]
                # Iterate on these retrieved values
                for _, row in values.iterrows():
                    # For stats purpose
                    distinct_sources.add(row['source'])
                    # "literal_eval" useful for transforming a string into a python object
                    for value in ast.literal_eval(row[prop_name]):
                        # We transform the date into an integer which will be used to order the dictionary of modifications.
                        converted_time = utils.date2seconds(value[1])
                        # Next, we add the modification to the time dictionary
                        if converted_time not in modifications_by_timestamp:
                            modifications_by_timestamp[converted_time] = {}
                            modifications_by_timestamp[converted_time][prop_name] = [value[0]]
                        else: 
                            if prop_name not in modifications_by_timestamp[converted_time]:
                                modifications_by_timestamp[converted_time][prop_name] = [value[0]]
                            else:
                                modifications_by_timestamp[converted_time][prop_name].append(value[0])

            ordered_timestamp_keys = list(modifications_by_timestamp.keys())
            ordered_timestamp_keys.sort()

            for timestamp in ordered_timestamp_keys:
                for prop, values in modifications_by_timestamp[timestamp].items():
                    latest_values[qid][prop] = values

    pickle.dump(latest_values, open('latest_values.pkl', 'wb'))
    pickle.dump(constrained_properties, open('constrained_properties.pkl', 'wb'))


if __name__ == "__main__":
    main()
