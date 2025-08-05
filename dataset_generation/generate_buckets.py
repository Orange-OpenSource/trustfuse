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

import tqdm
import pandas as pd

import utils


def main():

    parser = argparse.ArgumentParser(prog="generate_buckets",
                                     description="Generate buckets after the revision history collection step")
    parser.add_argument("--dataset",
                        help="Dataset containing conflicting data",
                        required=True)
    parser.add_argument("--fusion-data",
                        dest="fusion_data",
                        help="Folder path that will contain data for fusion models",
                        required=True)
    parser.add_argument("--revisions-folder",
                        dest="revisions_folder",
                        help=("Folder path where are stored the revisions"
                              "collected with generate_conflicting_data.py"),
                        required=True)
    parser.add_argument("--property-labels",
                        dest="property_labels",
                        help="Pickle file containing property PID/label mapping",
                        required=True)
    parser.add_argument("--constrained-properties",
                        dest="constrained_properties",
                        help=("Folder that contains three CSV files for three"
                              "diffrent constraints in Wikidata and its"
                              "associated properties"),
                        required=True)
    parser.add_argument("--media-properties",
                        dest="media_properties",
                        help=("CSV file containing media properties such as"
                              "(MP4, PNG, and others)"),
                        required=True)
    args = parser.parse_args()

    media_properties = pd.read_csv(args.media_properties)
    media_properties = list(media_properties['propertyLabel'])
    constrained_properties = {}

    for constraint_file in tqdm.tqdm(os.listdir(args.constrained_properties)):
        file_path = os.path.join(args.constrained_properties, constraint_file)
        constrained_properties_df = pd.read_csv(file_path)
        constrained_properties[constraint_file.replace('.csv', '')] = \
        [prop_url.replace('http://www.wikidata.org/entity/', '') \
         for prop_url in constrained_properties_df['pid'].values]
    
    # Construction parameters
    # alpha and delta in senconds
    parameters = {
        'delta': 63_072_000, # 2 years
        'alpha': 864_000, # 10 day
    }

    # dictionary containing modifications by timestamp
    modifications_by_timestamp = {}
    # "Dynamic" dictionary, since it will be filled and emptied when buckets will be generated
    dynamic_modifications = {}
    # Columns to be filtered for experimentation
    columns_to_filter = ['label', 
                         'description', 
                         'image', 
                         'ID', 
                         'URL', 
                         'Unnamed', 
                         'source', 
                         'place name sign',
                         'signature',
                         'audio',
                         'view',
                         'banner',
                         'Identifier',
                         'identifier',
                         'video',
                         'article',
                         '3D model',
                         'commons',
                         'Common',
                         'Commons',
                         'common',
                         'category',
                         'Category',
                         'code',
                         'Code'
                         ]
    columns_to_filter.extend(media_properties)
    # For stats purpose
    distinct_sources = set()
    # Read the folder of CSV files containing the revisions for each entity in a Wikipedia category
    files = os.listdir(args.revisions_folder)
    # Iterate files one by one
    for file in tqdm.tqdm(files):
        # Stats files are ignored
        if file.endswith('.csv') and 'stats' not in file:
            # Concatenate path to file
            file_path = os.path.join(args.revisions_folder, file)
            qid_df = pd.read_csv(file_path, index_col=False)
            # Extract the QID from the name of the file
            qid = file_path.replace(".csv", "")
            qid = qid.replace(args.revisions_folder, "")
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
                            modifications_by_timestamp[converted_time] = \
                                  [(qid, prop_name, value[0], row['source'], value[2], value[1])] # value[2] = QID ; value[1] == timestamp
                        else: 
                            modifications_by_timestamp[converted_time] \
                                .append((qid, prop_name, value[0], row['source'], value[2], value[1]))
                                                            
    # The dictionary is ordered by time before buckets are generated
    ordered_timestamp_keys = list(modifications_by_timestamp.keys())
    ordered_timestamp_keys.sort()
    # Create a list containing all the buckets of the category
    buckets = []
    # Dict containing all buckets by QID
    buckets_by_qid = {}
    # list of distinct data sources 
    distinct_sources = list(distinct_sources)
    # dictionary containing all modifications in the form of pairs (prop, value) for each source
    source_modifications = {s: set() for s in distinct_sources}
    # Time dict for each QID to compute delta
    time_by_qid = {q: -1 for q in dynamic_modifications.keys()}
    # Initialize bucket creation start time
    start_time = 0
    # Modifications are iterated in temporal order to create buckets (1st dimension for buckets)
    for timestamp in tqdm.tqdm(ordered_timestamp_keys):
        if start_time == 0:
            start_time = timestamp
            # We iterate over all modifications made at the same date
        for modification in modifications_by_timestamp[timestamp]: 
            qid = modification[0]
            prop = modification[1]
            value = modification[2]
            source = modification[3]
            value_id = modification[4]
            claimed_value_time = modification[5]
            # Check whether the source has already provided the (prop, value) pair for the same entity
            # First checking to avoid similar modifications by the same data source
            if (f"{qid}_{prop}", value) not in source_modifications[source]:
                # Time initialization
                if time_by_qid[qid] < 0:
                    time_by_qid[qid] = timestamp
                # Add the pair to the dictionary that groups source modifications
                source_modifications[source].add((f"{qid}_{prop}", value))
                # Check that the source has not already provided a value for this property in the current bucket 
                if source not in dynamic_modifications[qid][prop] and (timestamp-time_by_qid[qid]) < parameters['delta']:
                    # If it is not the case, we add the modification to the dynamic dictionary
                    dynamic_modifications[qid][prop][source] = [(value, timestamp, value_id, claimed_value_time)]
                elif source in dynamic_modifications[qid][prop] and (timestamp-time_by_qid[qid]) < parameters['delta'] and \
                (timestamp - dynamic_modifications[qid][prop][source][-1][1]) == 0:
                    dynamic_modifications[qid][prop][source].append((value, timestamp, value_id, claimed_value_time))
                else:
                    # If the source has modified its own value in a time lapse shorter than "modification_delta" 
                    # then this modification is not a criterion for closing the bucket, and the old value is replaced by the most recent
                    if (timestamp-time_by_qid[qid]) < parameters['delta'] and \
                        (timestamp - dynamic_modifications[qid][prop][source][-1][1]) <= parameters['alpha'] and \
                            (timestamp - dynamic_modifications[qid][prop][source][-1][1]) > 0:
                        dynamic_modifications[qid][prop][source] = [(value, timestamp, value_id, claimed_value_time)]

                    # Otherwise a bucket is generated
                    else:
                        # Initialization of the bucket
                        bucket = {
                            'only-bucket': []
                            }

                        # We retrieve all property/value pairs of the QID
                        for qid_prop, _ in dynamic_modifications[qid].items():

                            # list of sources involved for the "qid_prop" property
                            sources = list(dynamic_modifications[qid][qid_prop].keys())

                            for involved_source, source_values in dynamic_modifications[qid][qid_prop].items():
                                for source_value in source_values:
                                    bucket['only-bucket'].append((involved_source, qid_prop, source_value[0], source_value[2], source_value[3]))

                            # Empty the dynamic modification dictionary after bucket construction and add the new contribution
                            dynamic_modifications[qid][qid_prop] = {}
                        # We put the new value that triggered the bucket's closure into the dynamic dictionary previously emptied.
                        dynamic_modifications[qid][prop][source] = [(value, timestamp, value_id, claimed_value_time)]
                        # We add some information to the generated bucket
                        bucket['temporal_size_readable'] = utils.convert_seconds_to_readable_time(timestamp - start_time)
                        bucket['temporal_size'] = timestamp - start_time

                        bucket['start_time'] = utils.seconds2date(start_time)
                        bucket['end_time'] = utils.seconds2date(timestamp)
                        # The time of the previous bucket is assigned to the start time of the next bucket
                        start_time = timestamp
                        time_by_qid[qid] = timestamp
                        # Then add the bucket to the list of all buckets
                        buckets.append(bucket)
                        if qid in buckets_by_qid:
                            buckets_by_qid[qid]['buckets'].append(bucket)
                        else:
                            buckets_by_qid[qid] = {
                                'buckets': [bucket]
                            }

    # Completion of buckets with remaining modifications 
    # when the same source has not provided
    # a value for the same (entity, property) pair.
    
    bar = tqdm.tqdm(total=len(dynamic_modifications))
    for qid, properties in dynamic_modifications.items():
        bucket = {
            'only-bucket': []
            }
        timestamps = []
        for prop, sources in properties.items():
            for source, values in sources.items():
                for value in values:
                    # QID prefix added to distinguish properties
                    bucket['only-bucket'].append((source, prop, value[0], value[2], value[3]))
                    timestamps.append(value[1])
        if len(bucket) > 0:
            bucket['start_time'] = utils.seconds2date(min(timestamps)) # utils.convert_seconds_to_readable_time(min(timestamps))
            bucket['end_time'] = utils.seconds2date(max(timestamps)) # utils.convert_seconds_to_readable_time(max(timestamps))
            buckets.append(bucket)
            if qid in buckets_by_qid:
                buckets_by_qid[qid]['buckets'].append(bucket)
            else:
                buckets_by_qid[qid] = {
                    'buckets': [bucket]
                    }
        bar.update(1)
    
    # Save buckets in fusion data folder
    dir_path = os.path.join(os.getcwd(), args.fusion_data)
    os.makedirs(dir_path, exist_ok=True)
    # Children folder for conflicting data
    conflicting_data_path = os.path.join(os.getcwd(), os.path.join(args.fusion_data, 'conflicting_data'))
    os.makedirs(conflicting_data_path, exist_ok=True)
    # Children folder for ground truth data
    dir_path = os.path.join(os.getcwd(), os.path.join(args.fusion_data, 'ground_truth'))
    os.makedirs(dir_path, exist_ok=True)

    # Save each bucket as a CSV file
    bucket_id = 0
    for bucket in tqdm.tqdm(buckets):
        bucket_id += 1
        distinct_properties = set()
        claims_by_source = {}
        for claim in bucket['only-bucket']:
            distinct_properties.add(claim[1])
            if claim[0] not in claims_by_source:
                claims_by_source[claim[0]] = {claim[1]: claim[2]}
            else:
                claims_by_source[claim[0]][claim[1]] = claim[2]
        bucket_claims = {prop: [] for prop in distinct_properties}
        bucket_claims['source'] = []
        for source, claims in claims_by_source.items():
            bucket_claims['source'].append(source)
            for prop in distinct_properties:
                if prop in claims:
                    bucket_claims[prop].append(claims[prop])
                else:
                    bucket_claims[prop].append(None)

    # Serialize the buckets
    pickle.dump(buckets_by_qid, open(args.dataset, 'wb'))
    if args.source_names is not None:
        pickle.dump(distinct_sources, open(args.source_names, 'wb'))


if __name__ == "__main__":
    main()
