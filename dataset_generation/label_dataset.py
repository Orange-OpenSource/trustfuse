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
import os
import re
import pickle

import pandas as pd
import tqdm
import copy


def length_without_ansi(text):
    ansi_escape = re.compile(r'\x1B[@-_][0-?]*[ -/]*[@-~]')
    return len(ansi_escape.sub('', text))


def text_boxing(text):
    lines = text.split('\n')
    max_length = max(length_without_ansi(line) for line in lines)
    
    upper_border = '┏' + '━' * (max_length + 2) + '┓'
    lower_border = '┗' + '━' * (max_length + 2) + '┛'
    
    print(upper_border)
    for line in lines:
        espace_padding = ' ' * (max_length - length_without_ansi(line))
        print(f'┃ {line}{espace_padding} ┃')
    print(lower_border)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--buckets',
                        required=True)
    parser.add_argument('--latest-values',
                        dest='latest_values',
                        required=True)
    parser.add_argument("--fusion-data",
                        dest="fusion_data",
                        help="Folder path that will contain data for fusion models",
                        required=True)
    parser.add_argument("--constrained-properties",
                        dest="constrained_properties",
                        required=True)
    parser.add_argument("--attribute-types",
                        dest="attribute_types",
                        required=True)
    parser.add_argument("--filters",
                        required=True)
    args = parser.parse_args()


    # Get latest values to support labeling decision making
    with open(args.latest_values, 'rb') as f:
        latest_values = pickle.load(f)
    # Support labeling by indicating if a property is multi-valued
    with open(args.constrained_properties, 'rb'):
        constrained_properties = pickle.load(f)
    # To display or not the QID of entities as values
    with open(args.attribute_types, 'rb'):
        attribute_types = pickle.load(f)
    # QIDs to be labeled
    qids_to_label = ['Q243'] # 'Q64436', 'Q517'
    #qids_to_label = ['Q15155995'] # QID de test

    # Load output of buckets generation script
    with open(args.buckets, 'rb') as f:
        buckets_by_qid = pickle.load(f)

    # Partial ordering of values
    partial_ordering = {}

    # To filter the properties (media properties, st. images, audio)
    filters = []
    try:
        with open(args.filters, 'rb') as f:
            filters =  pickle.load(f)
    except FileNotFoundError:
        print("The file does not exist.")


    for qid in qids_to_label:
        print('\n************************************************')
        print(f"You are starting the labeling of entity \033[32m{qid}\033[0m")
        print('************************************************')

        partial_ordering[qid] = {
            "qid": {},
            "value": {}
        }

        buckets = buckets_by_qid[qid]['buckets']
        # To facilitate labeling, we save all possible values for each property
        values_by_prop = {}

        # Dictionary containing all values already seen for each property
        # depending on the bucket, bucket i includes all the values in bucket i-1
        seen_values = {i: {} for i in range(len(buckets))}

        for i, bucket in enumerate(buckets):
            if i-1 in seen_values:
                seen_values[i].update(copy.deepcopy(seen_values[i-1]))
            for contrib in bucket['only-bucket']:
                #source = contrib[0]
                prop = contrib[1]
                value = contrib[2]
                value_id = contrib[3]

                # will be manipulated at the input preparation for fusion models
                if prop not in filters:
                    if prop not in seen_values[i]:
                        seen_values[i][prop] = set()
                    seen_values[i][prop].add((value, value_id))
                    if prop not in values_by_prop:
                        values_by_prop[prop] = set([(value, value_id)])
                    else:
                        values_by_prop[prop].add((value, value_id))
        # First Pass
        # We apply a first labeling pass that to speed up labeling
        # by separating static and dynamic properties
        print(f'First pass of labeling!')

        # Dictionary containing property values for the first pass
        props_first_pass = {}
        for prop, values in tqdm.tqdm(values_by_prop.items()):

            # values_mapping allows the annotator to specify
            # numbers instead of strings
            values_mapping = {str(i): val for i, val in enumerate(values)}

            # For display purpose
            choose_values = ''
            for key, val in values_mapping.items():
                if prop in attribute_types:
                    if attribute_types[prop] == 'Entity':
                        choose_values += f'\033[32m{key}\033[0m : {val[0]} ({val[1]})\n'
                    else:
                        choose_values += f'\033[32m{key}\033[0m : {val[0]}\n'
                else:
                    choose_values += f'\033[32m{key}\033[0m : {val[0]}\n'

            response = ''
            while response not in ['n', 'y', 'f']:
                print("\n")
                text_boxing(f"Is the value of the property \033[32m{prop}\033[0m "
                             f"likely to change over time? (\033[32my\033[0m or \033[32mn\033[0m) or \033[32mf (filtrer)\033[0m ?")
                response = input(f"\n\033[33mEntrée : \033[0m")
                
                if response not in ['n', 'y', 'f']:
                    print(f"\033[31mThe answer you gave is not valid.\033[0m\n")
            # If it's a property to be excluded from labeling, enter -1 and
            # will be excluded for all remaining entities
            if response == 'f':
                filters.append(prop)
            
            # If the property cannot evolve over time
            # If the property is not expected to change over time, the annotator
            # can set its value now, instead of proceeding by bucket
            elif response == 'n':
                # We check whether the property expects strictly a single value to inform the annotator
                constraint = '\033[32mthe correct value(s)\033[0m by separating them with a comma (val1,val2).'
                if prop in constrained_properties['single-best_value'] or prop in constrained_properties['single-value']:
                    constraint = '\033[32mONLY SINGLE VALUE\033[0m'

                current_value = '' 
                for val in latest_values[qid][prop]:
                    current_value += f'\033[35m{val}\033[0m\n'

                stop_criterion = True
                while stop_criterion:
                    stop_criterion = False
                    print("\n")
                    text_boxing(f"Select {constraint} \n" 
                                f"for this property from among these (if none, enter \033[32m-1\033[0m) :\nCurrent values : {current_value} \n{choose_values}")
                    response = input(f"\n\033[33mInput : \033[0m")
                    
                    if response == '-1':
                        props_first_pass[prop] = None
                    else:
                        chosen_values = response.split(',')
                        for chosen_val in chosen_values:
                            if chosen_val not in values_mapping:
                                stop_criterion = True
                                print(f"\033[31mThe answer you gave is not valid.\033[0m\n")
                                break
                        # We store the values entered in the first pass dictionary
                        if stop_criterion == False:
                            props_first_pass[prop] = [values_mapping[val] for val in chosen_values]

                # Partial ordering
                stop_criterion = True
                while stop_criterion:
                    stop_criterion = False
                    print("\n")
                    text_boxing(f"If other values are correct but less specific, provide "
                                 f"a partial order using the character \033[32m{'<'}\033[0m\n"
                                 f"If you want to provide several orders, separate them by \033[32m{';'}\033[0m\n."
                                 f"for example: \033[32m{'5=4;3<2=1<0'}\033[0m (Paris=Nanterre=Chatillon<Ile-de-France<France;Chine<Asia)"
                                 f" else provide \033[32m{'-1'}\033[0m :\n{choose_values}")
                    response = input(f"\n\033[33mInput : \033[0m")

                    if response == '-1':
                        stop_criterion = False
                        partial_ordering[qid]["qid"][prop] = []
                        partial_ordering[qid]["value"][prop] = []

                    else:
                        partial_orders = response.split(';')
                        all_elements_qid = []
                        all_elements_value = []
                        for partial_order in partial_orders:
                            elements = partial_order.split('<')
                            elements = [elt.split('=') for elt in elements]
                            elements_mapping_qid = []
                            elements_mapping_value = []
                            for m_elt in elements:
                                m_elt_mapping_qid = []
                                m_elt_mapping_value = []
                                for elt in m_elt:
                                    if elt not in values_mapping:
                                        stop_criterion = True
                                        print(f"\033[31mThe answer you gave is not valid.\033[0m\n") 
                                        break
                                    else:
                                        m_elt_mapping_qid.append(values_mapping[elt][1])
                                        m_elt_mapping_value.append(values_mapping[elt][0])
                                elements_mapping_qid.append(m_elt_mapping_qid)
                                elements_mapping_value.append(m_elt_mapping_value)
                            all_elements_qid.append(elements_mapping_qid)
                            all_elements_value.append(elements_mapping_value)

                        # We do the same by storing the partial order in the
                        # dictionary dedicated to the first labeling pass
                        if stop_criterion == False:
                            partial_ordering[qid]["qid"][prop] = all_elements_qid
                            partial_ordering[qid]["value"][prop] = all_elements_value

                    
        # Filter out labeled properties before labeling buckets
        # We are now interested in the remaining properties, but this time we will go through the buckets one by one
        # First of all, we create a static_props dictionary which will store for each bucket
        # identified by its index in the i list, the static properties whose GT was given during the
        # first labeling pass
        remaining_buckets = []
        static_props = {}
        for i, bucket in enumerate(buckets):
            static_props[i] = set()
            filtered_bucket = bucket.copy()
            new_bucket = []
            for triple in filtered_bucket['only-bucket']:
                if triple[1] not in props_first_pass and triple[1] not in filters:
                    new_bucket.append(triple)
                else:
                    if triple[1] not in filters:
                        static_props[i].add(triple[1])
            filtered_bucket['only-bucket'] = new_bucket
            remaining_buckets.append(filtered_bucket)

        # Second labeling pass
        print(f'Seconde passe de labellisation !')
        # We define the same elements as above
        props_second_pass = {}
        partial_ordering_second_pass = {}
        # Iterate over the remaining buckets
        for i, bucket in tqdm.tqdm(enumerate(remaining_buckets)):
            # Initialization of partial orders and GT props for
            # the bucket for the second labeling pass
            partial_ordering_second_pass[i] = {
                "value": {},
                "qid": {}
            }
            props_second_pass[i] = {}
            # Distinct values per prop for each bucket
            b_values_by_prop = {}
            for triple in bucket['only-bucket']:
                prop = triple[1]
                value = triple[2]
                value_id = triple[3]
                if prop not in b_values_by_prop:
                    b_values_by_prop[prop] = set([(value, value_id)])
                else:
                    b_values_by_prop[prop].add((value, value_id))
            # Start labeling
            for prop in tqdm.tqdm(b_values_by_prop):
                # All values already seen until the bucket "i"
                values = seen_values[i][prop]
                # Mapping dict
                values_mapping = {str(i): val for i, val in enumerate(values)}
                # For display purpose
                choose_values = ''
                for key, val in values_mapping.items():
                    if prop in attribute_types:
                        if attribute_types[prop] == 'Entity':
                            choose_values += f'\033[32m{key}\033[0m : {val[0]} ({val[1]})\n'
                        else:
                            choose_values += f'\033[32m{key}\033[0m : {val[0]}\n'
                    else:
                        choose_values += f'\033[32m{key}\033[0m : {val[0]}\n'
                # We add the constraint, if any, to help with labeling
                constraint = '\033[32mla (les) bonne(s) valeur(s)\033[0m en les séparant par une virgule (val1,val2)'
                if prop in constrained_properties['single-best_value'] or prop in constrained_properties['single-value']:
                    constraint = '\033[32mONLY SINGLE VALUE\033[0m'
                # We display the current value on Wikidata to help the annotator
                current_value = '' 
                for val in latest_values[qid][prop]:
                    current_value += f'\033[35m{val}\033[0m\n'
                # While an answer is invalid, we continue to return the question
                stop_criterion = True
                while stop_criterion:
                    stop_criterion = False
                    print("\n")
                    text_boxing(f"Bucket time interval = [{bucket['start_time']}, {bucket['end_time']}]\n"
                                 f"Select {constraint} \n"
                                 f"for the property \033[32m{prop}\033[0m among these (if none provide \033[32m-1\033[0m)"
                                 f" :\nCurrent values : {current_value} \n{choose_values}")
                    response = input(f"\n\033[33mInput : \033[0m")
                    # If none of the values is correct, GT is labelled as None and a good fusion model
                    # should provide no value (or a suggestion in the best case)
                    if response == '-1':
                        props_second_pass[i][prop] = None
                    else:
                        chosen_values = response.split(',')
                        for chosen_val in chosen_values:
                            if chosen_val not in values_mapping:
                                stop_criterion = True
                                print(f"\033[31mThe answer you gave is not valid.\033[0m\n")
                                break
                        if stop_criterion == False:
                            # Reverse mapping is used to save values rather than mapping integers
                            props_second_pass[i][prop] = [values_mapping[val] for val in chosen_values]

                # Partial ordering
                # We are now interested in partial orders if several values coexist
                stop_criterion = True
                while stop_criterion:
                    stop_criterion = False
                    print("\n")
                    text_boxing(f"If other values are correct but less specific, provide "
                                f"a partial order using the character \033[32m{'<'}\033[0m\n"
                                f"If you want to provide several orders, separate them by \033[32m{';'}\033[0m\n."
                                f"for example: \033[32m{'5=4;3<2=1<0'}\033[0m (Paris=Nanterre=Chatillon<Ile-de-France<France;Chine<Asia)"
                                f" else provide \033[32m{'-1'}\033[0m :\n{choose_values}")
                    response = input(f"\n\033[33mInput : \033[0m")

                    if response == '-1':
                        stop_criterion = False
                        partial_ordering_second_pass[i]["qid"][prop] = []
                        partial_ordering_second_pass[i]["value"][prop] = []

                    else:
                        partial_orders = response.split(';')
                        all_elements_qid = []
                        all_elements_value = []
                        for partial_order in partial_orders:
                            elements = partial_order.split('<')
                            elements = [elt.split('=') for elt in elements]
                            elements_mapping_qid = []
                            elements_mapping_value = []
                            for m_elt in elements:
                                m_elt_mapping_qid = []
                                m_elt_mapping_value = []
                                for elt in m_elt:
                                    if elt not in values_mapping:
                                        stop_criterion = True
                                        print(f"\033[31mLThe answer you gave is not valid.\033[0m\n") 
                                        break
                                    else:
                                        m_elt_mapping_qid.append(values_mapping[elt][1])
                                        m_elt_mapping_value.append(values_mapping[elt][0])
                                    elements_mapping_qid.append(m_elt_mapping_qid)
                                    elements_mapping_value.append(m_elt_mapping_value)
                            all_elements_qid.append(elements_mapping_qid)
                            all_elements_value.append(elements_mapping_value)

                        if stop_criterion == False:
                            # Reverse mapping and saving partial orders
                            partial_ordering_second_pass[i]["qid"][prop] = all_elements_qid
                            partial_ordering_second_pass[i]["value"][prop] = all_elements_value

        # We define a dictionary that will be used to create all the
        # dataframes representing GT for buckets
        buckets_df = {}
        bucket_qids = {}
        for i in range(len(buckets)):
            buckets_df[i] = {}
            # dict --> dataframe for index bucket i
            bucket_claims = {}
            bucket_qids = {}
            for prop in static_props[i]:
                gt_values = []
                gt_qids = []
                if props_first_pass[prop] is not None:
                    gt_values = [[val[0] for val in props_first_pass[prop] 
                                if val in seen_values[i][prop]]] # and val is not None
                    gt_qids = [[val[1] for val in props_first_pass[prop]
                              if val in seen_values[i][prop]]] # and val is not None
                    if len(gt_values) == 0:
                        gt_values = [[None]]
                        gt_qids = [[None]]
                else:
                    gt_values = [[None]]
                    gt_qids = [[None]]
                bucket_claims[prop] = gt_values
                bucket_qids[prop] = gt_qids

            for prop, values in props_second_pass[i].items():
                if values is not None:
                    bucket_claims[prop] = [[val[0] for val in values]] # if val is not None
                    bucket_qids[prop] = [[val[1] for val in values]] # if val is not None
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

            # Orders
            bucket_partial_ordering_value = partial_ordering_second_pass[i]["value"]
            bucket_partial_ordering_value.update(partial_ordering[qid]["value"])
            bucket_partial_ordering_qid = partial_ordering_second_pass[i]["qid"]
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
                claims_by_source[claim[0]][claim[1]].append((claim[2], claim[3]))
            # Pre-fill the dataframe template with column names
            bucket_claims_value = {prop: [] for prop in distinct_properties}
            bucket_claims_qid = {prop: [] for prop in distinct_properties}
            bucket_claims_value['Source'] = []
            bucket_claims_qid['Source'] = []
            for source, _ in claims_by_source.items():
                max_attribute_values = max(len(v) for v in claims_by_source[source].values())
                bucket_claims_value['Source'].extend([source] * max_attribute_values)
                bucket_claims_qid['Source'].extend([source] * max_attribute_values)
                for prop in distinct_properties:
                    if prop in claims_by_source[source]:
                        for val in claims_by_source[source][prop]:
                            bucket_claims_value[prop].append(val[0])
                            bucket_claims_qid[prop].append(val[1])
                        # Complete with None to create the Dataframe correctly
                        bucket_claims_value[prop].extend(
                            [None for _ in range(max_attribute_values \
                                                 - len(claims_by_source[source][prop]))]
                            )
                        bucket_claims_qid[prop].extend(
                            [None for _ in range(max_attribute_values \
                                                 - len(claims_by_source[source][prop]))]
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

        pickle.dump(buckets_df, open(os.path.join(args.fusion_data, f'{qid}_gt_buckets.pkl'), 'wb'))
        pickle.dump(filters, open(args.filters, 'wb'))


if __name__ == '__main__':
    main()
