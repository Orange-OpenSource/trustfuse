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
import json
import pickle
import os
import time
import concurrent.futures

from typing import List, Dict

import requests
import pandas
import tqdm
import lmdb


def get_contributions_count(username):
    """ Retrieves the number of contributions made by a Wikidata user based on his username """

    url = f"https://www.wikidata.org/w/api.php?action=query&list=users&ususers={username}&usprop=editcount&format=json"
    try:
        response = requests.get(url)
        data = response.json()
        
        if 'editcount' in data['query']['users'][0]:
            contributions_count = data['query']['users'][0]['editcount']
            return contributions_count
        else:
            return "User not found or no contribution."
    
    except Exception as e:
        return f"Data recovery error : {str(e)}"


def get_label(txn, qid):
    """ Get english label of an entity """
    label = None
    try:
        obj = txn.get(qid.encode("ascii"))

        label = "no label"
        if obj:
            entity = pickle.loads(obj)

            if 'labels' in entity:
                if 'en' in entity['labels']:
                    label = entity['labels']['en']["value"]
                elif 'fr' in entity['labels']:
                    label = entity['labels']['fr']["value"]
        else:
            label = None
        if isinstance(label, list):
            label = label[0]

        if label is not None:
            label = label.replace(",", " ")
            label = label.replace(";", " ")

        return label
    except:
        return label


def get_label_prop(element_id):
    """ To obtain the english/french label of an entity or property """
    url = f"https://www.wikidata.org/w/api.php?action=wbgetentities&format=json&ids={element_id}"
    response = requests.get(url)
    data = response.json()
    if "error" not in data:
        if element_id in data["entities"]:
            if "labels" in data['entities'][element_id]:
                if "en" in data['entities'][element_id]['labels']:
                    label = data['entities'][element_id]['labels']['en']['value']
                    return label
                elif "fr" in data['entities'][element_id]:
                    label = data['entities'][element_id]['labels']['fr']['value']
                    return label
        else:
            return element_id
    else:
        return element_id


def get_entity_revisions(entity_id,
                         units,
                         property_labels,
                         global_stats,
                         all_modifications,
                         distinct_values,
                         distinct_users,
                         txn,
                         dir_path,
                         prop_types):
    """ Requests the Wikidata API, retrieves revisions of entity_id
    and saves them as a CSV file with the following columns: properties
    and for each row a revision (additions / modifications) and the revision source """

    revisions = {}
    revision_nb = 0
    url = 'https://www.wikidata.org/w/api.php'

    params = {
        'action': 'query',
        'prop': 'revisions',
        'format': 'json',
        # Retrieve revision content, user name and ID, and revision timestamp
        'rvprop': 'content|user|userid|timestamp',
        # Limit the number of revisions on a page
        'rvlimit': 500,
        'titles': f'Item:{entity_id}'
    }

    while True:
        response = None
        # Loop to ensure API response
        while response == None:
            # Query the Wikidata API with the above query parameters
            try:
                response = requests.get(url, params=params)
            except:
                time.sleep(1)
                response = requests.get(url, params=params)
        # Formalizing query results
        data = response.json()
        current_revisions = data['query']['pages']
        for _, page_data in current_revisions.items():
            if 'revisions' in page_data:
                for rev in page_data['revisions']:
                    if "*" in rev:
                        rev_modified = rev
                        # Convert revision content into JSON objects
                        rev_modified["*"] = json.loads(rev["*"])
                        # Saves the revision in a dictionary with its revision number
                        revisions[revision_nb] = rev
                        revision_nb += 1
        # If there are still revisions, continue to collect them.
        if 'continue' in data:
            rvcontinue = data['continue']['rvcontinue']
            params["rvcontinue"] = rvcontinue
        else:
            break

    # Statistics
    entity_stats = {
    "modifications": [0],
    "additions": [0],
    "creations": [0]
    }

    # We query the Wikidata API and save the revisions in a dictionary
    """
    total_revisions += len(revisions)
    """
    # Compute the differences between n and n-1 to find the modifications made by a user
    # modifications := dictionary(key = user ID, dictionary(key = PID, list(modifications)))
    modifications = {}

    # To save modified entity properties
    entity_properties = []
    # Detects changes between each pair of revisions
    # Because each revision contains all the current (properties, values) of the entity
    find_modifications(modifications,
                       revisions,
                       units,
                       entity_properties,
                       entity_stats,
                       property_labels,
                       prop_types)
    # Stats
    for key, val in entity_stats.items():
        if key in global_stats:
            global_stats[key][0] += val[0]
        else:
            global_stats[key] = val
    # Save each entity modifications in a global dict to find correlations between contributors
    all_modifications[entity_id] = modifications

    dataset = {prop: [] for prop in entity_properties}
    dataset["source"] = []

    # Dictionary (QID: {PID: {val1, val2, ..., valn}}) which contains all
    # distinct values for each property and for each entity (QID)
    distinct_values[entity_id] = {prop: set() for prop in entity_properties}

    for source, contributions in modifications.items():
        username = contributions["user"]
        if username in data:
            data[username][entity_id] = set(contributions.keys())
        else:
            data[username] = {entity_id: set(contributions.keys())}
        distinct_users.add(source)
        dataset["source"].append(source)

        for prop in entity_properties:
            if prop in contributions:
                contribution_list = []
                for contrib in contributions[prop]:
                    contrib_label = get_label(txn, contrib[0])
                    if contrib_label is not None and contrib_label != "no label":
                        contribution_list.append((contrib_label, contrib[1], contrib[0]))
                    else:
                        contribution_list.append(contrib + (contrib[0],))
                dataset[prop].append(contribution_list)

            else:
                dataset[prop].append(None)


    qid_file_path = os.path.join(dir_path, f"{entity_id}.csv")
    serialized_qid_file_path = os.path.join(dir_path, f"{entity_id}.csv")
    qid_df = pandas.DataFrame(dataset)
    # Serialize conflicting data into pickle file format
    pickle.dump(qid_df, open(serialized_qid_file_path, "wb"))
    qid_df.to_csv(qid_file_path)
    qid_stats = pandas.DataFrame(entity_stats)
    qid_stats.to_csv(os.path.join(dir_path, f"{entity_id}_stats.csv"))

    return {entity_id: revisions}


def get_values(objects, units, prop, prop_types):
    values = set()
    for value in objects:
        if "datavalue" in value["mainsnak"]:
            try:
                # quantity
                if value["mainsnak"]["datavalue"]["type"] == "quantity":
                    unit = value["mainsnak"]["datavalue"]["value"]["unit"]
                    if unit not in units:
                        units[unit] = get_label_prop(unit.replace("http://www.wikidata.org/entity/", ""))
                        unit = units[unit]
                    else:
                        unit = units[unit]
                    values.add(value["mainsnak"]["datavalue"]["value"]["amount"] + 
                                                                unit)
                    # Save property type for knowledge fusion purpose
                    if prop not in prop_types:
                        prop_types[prop] = 'Numerical'

                # string
                if value["mainsnak"]["datavalue"]["type"] == "string":
                    values.add(value["mainsnak"]["datavalue"]["value"])
                    if prop not in prop_types:
                        prop_types[prop] = 'String'

                # time
                if value["mainsnak"]["datavalue"]["type"] == "time":
                    time_value = value["mainsnak"]["datavalue"]["value"]["time"].replace("+", "")
                    time_value = time_value.replace("T00:00:00Z", "")
                    values.add(time_value)
                    if prop not in prop_types:
                        prop_types[prop] = 'Time'

                # globecoordinate
                if value["mainsnak"]["datavalue"]["type"] == "globecoordinate":
                    values.add((value["mainsnak"]["datavalue"]["value"]["latitude"], 
                                value["mainsnak"]["datavalue"]["value"]["longitude"]))
                    if prop not in prop_types:
                        prop_types[prop] = 'Coordinate'

                # monolingualtext
                if value["mainsnak"]["datavalue"]["type"] == "monolingualtext":
                    values.add(value["mainsnak"]["datavalue"]["value"]["text"])
                    if prop not in prop_types:
                        prop_types[prop] = 'String'

                # wikibase-entityid
                if value["mainsnak"]["datavalue"]["type"] == "wikibase-entityid":
                    values.add(value["mainsnak"]["datavalue"]["value"]["id"])
                    if prop not in prop_types:
                        prop_types[prop] = 'Entity'
            except:
                continue

    return values


def compare_revisions(old_revision: Dict,
                      current_revision: Dict,
                      modifications: Dict,
                      units: Dict,
                      entity_properties: List,
                      entity_stats,
                      property_labels: Dict,
                      prop_types: Dict):
    """ Detects additions and modifications by comparing each pair of consecutive revisions.
    Args: 
        - old_revision (dict): entity at time t-1 (old)
        - current_revision (dict): entity at time t (recent)
        - modifications (dict): dictionary to record all modifications made to an entity
        - units (dict): dictionary to record labels already requested (for all QIDs)
        - entity_properties (list): list containing all modified entity properties
        - entity_stats: stats
        - property_labels (dict): contains property labels (for all QIDs)
    """

   # if the user's account has been deleted, the revision is ignored
    # because no source to evaluate for fusion methods
    if "userhidden" not in current_revision and "suppressed" not in current_revision:
        # If the user had not yet contributed to the entity, then we add the user to the dictionary
        if current_revision["user"] not in modifications:
            modifications[current_revision["user"]] = {
                "userid": current_revision["userid"], 
                "user": current_revision["user"]
                }
        # Extract pairs (property, value) as a dictionary
        old_claims = old_revision["*"].get('claims', {})
        if old_claims == []:
            old_claims = {}
        current_claims = current_revision["*"].get('claims', {})
        if current_claims == []:
            current_claims = {}
        
        # Identifies added, deleted and modified properties
        added_properties = set(current_claims.keys()) - set(old_claims.keys())
        modified_properties = set(old_claims.keys()) & set(current_claims.keys())
        
        # Identifies value changes for modified properties
        for prop in (modified_properties | added_properties):
            # Registers property label if not already requested
            if prop not in property_labels:
                prop_label = get_label_prop(prop)
                property_labels[prop] = prop_label
                property_labels[prop_label] = prop
            if property_labels[prop] not in entity_properties:
                # adds the property to the list of entity properties that will be useful
                # to save the entity dataset as a CSV file
                entity_properties.append(property_labels[prop])
                # For stats purpose
                entity_stats["creations"][0] += 1
                entity_stats[f"{property_labels[prop]} modifications"] = [0]
                entity_stats[f"{property_labels[prop]} additions"] = [0]

            old_values = set()
            if prop in old_claims:
                old_values = get_values(old_claims[prop], units, prop, prop_types)
            current_values = get_values(current_claims[prop], units, prop, prop_types)
            added_values = current_values - old_values
            if added_values:
                # For statistics purpose
                if len(old_values & current_values) < len(old_values):
                    entity_stats["modifications"][0] += 1
                    entity_stats[f"{property_labels[prop]} modifications"][0] += 1
                else:
                    entity_stats["additions"][0] += 1
                    entity_stats[f"{property_labels[prop]} additions"][0] += 1
                # Save changes
                if property_labels[prop] not in modifications[current_revision["user"]]:
                    modifications[current_revision["user"]][property_labels[prop]] = \
                        {(val, current_revision["timestamp"]) for val in current_values}
                else:
                    modifications[current_revision["user"]][property_labels[prop]] |= \
                        {(val, current_revision["timestamp"]) for val in current_values}

        # properties of type "description"
        old_claims = old_revision["*"].get('descriptions', {})
        if old_claims == []:
            old_claims = {}
        current_claims = current_revision["*"].get('descriptions', {})
        if current_claims == []:
            current_claims = {}
        # Identify added, deleted and modified properties
        added_properties = set(current_claims.keys()) - set(old_claims.keys())
        modified_properties = set(old_claims.keys()) & set(current_claims.keys())
        
        # Identify value changes for modified properties
        for prop in (modified_properties | added_properties):
            if f"description_{prop}" not in entity_properties:
                entity_properties.append(f"description_{prop}")
                entity_stats["creations"][0] += 1
                entity_stats[f"description_{prop} modifications"] = [0]
                entity_stats[f"description_{prop} additions"] = [0]
            old_values = set()
            if prop in old_claims:
                old_values = {old_claims[prop]["value"]}
            current_values = {current_claims[prop]["value"]}
            added_values = current_values - old_values
            if added_values:
                if len(old_values & current_values) < len(old_values):
                    entity_stats["modifications"][0] += 1
                    entity_stats[f"description_{prop} modifications"][0] += 1
                else:
                    entity_stats["additions"][0] += 1
                    entity_stats[f"description_{prop} additions"][0] += 1
                if f"description_{prop}" not in modifications[current_revision["user"]]:
                    modifications[current_revision["user"]][f"description_{prop}"] = \
                        {(val, current_revision["timestamp"]) for val in added_values}
                else:
                    modifications[current_revision["user"]][f"description_{prop}"] |= \
                        {(val, current_revision["timestamp"]) for val in added_values}

        # properties of type "label"
        old_claims = old_revision["*"].get('labels', {})
        if old_claims == []:
            old_claims = {}
        current_claims = current_revision["*"].get('labels', {})
        if current_claims == []:
            current_claims = {}

        # Identify added, deleted and modified properties
        added_properties = set(current_claims.keys()) - set(old_claims.keys())
        #deleted_properties = set(old_claims.keys()) - set(current_claims.keys())
        modified_properties = set(old_claims.keys()) & set(current_claims.keys())

        # Identify value changes for modified properties
        for prop in (modified_properties | added_properties):
            if f"label_{prop}" not in entity_properties:
                entity_properties.append(f"label_{prop}")
                entity_stats["creations"][0] += 1
                entity_stats[f"label_{prop} modifications"] = [0]
                entity_stats[f"label_{prop} additions"] = [0]
            old_values = set()
            if prop in old_claims:
                old_values = {old_claims[prop]["value"]}
            current_values = {current_claims[prop]["value"]}
            added_values = current_values - old_values
            if added_values:
                if len(old_values & current_values) < len(old_values):
                    entity_stats["modifications"][0] += 1
                    entity_stats[f"label_{prop} modifications"][0] += 1
                else:
                    entity_stats["additions"][0] += 1
                    entity_stats[f"label_{prop} additions"][0] += 1
                if f"label_{prop}" not in modifications[current_revision["user"]]:
                    modifications[current_revision["user"]][f"label_{prop}" ] = \
                        {(val, current_revision["timestamp"]) for val in added_values}
                else:
                    modifications[current_revision["user"]][f"label_{prop}" ] |= \
                        {(val, current_revision["timestamp"]) for val in added_values}


def find_modifications(modifications,
                       revisions,
                       units,
                       entity_properties,
                       entity_stats,
                       property_labels,
                       prop_types: Dict):
    """ Function that identifies changes made between each revision of a Wikidata entity:
    Args:
    - modifications (dict) : dictionary that will contain these modifications with the user ID as key 
    and as a value a dictionary that will associate each property with the modifications made by this user.
    - revisions (dict): the output of the Wikidata API request key=revision ID and value=Wikidata entity
    """

    # The revision keys are sorted in descending order
    reversed_keys = sorted(revisions.keys(), reverse=True)
    # Create an empty revision to represent the starting point for building the Wikidata entity with none (property, value).
    reversed_keys.insert(0, reversed_keys[0]+1)
    revisions[reversed_keys[0]] = {"*": {}}
    # Once the keys have been sorted, we iterate over the revisions.
    for i in range(len(reversed_keys)-1):
        # Compute differences
        compare_revisions(revisions[reversed_keys[i]],    # older version
                          revisions[reversed_keys[i+1]],  # newer version
                          modifications,                  # dictionary to save changes
                          units,                          # dictionary to store unit labels
                          entity_properties,              # list to save all modified properties for an entity
                          entity_stats,                   # stats
                          property_labels,                # dict to save property labels
                          prop_types)


def main():

    parser = argparse.ArgumentParser(prog="generate_conflicting_dataset",
                                     description="Conflicting dataset generator")
    parser.add_argument("--entities",
                        help="JSON file containing the list of entities of interest",
                        required=True)
    parser.add_argument("--wikidata-hashmap",
                        help="Local hashmap to get labels efficiently",
                        dest="wikidata_hashmap",
                        required=True)
    parser.add_argument("--stats",
                        help="stats file to get insights from the dataset",
                        required=True)
    parser.add_argument("--prop-types",
                        dest="prop_types",
                        help="To record the type of each property seen during the retrieving",
                        required=True)
    args = parser.parse_args()

    # Open a Wikidata hashmap to retrieve entity labels more quickly than with the API
    dump = lmdb.open(args.wikidata_hashmap, readonly=True, readahead=False)
    txn = dump.begin()

    # Interest QIDs (Wikipedia category) are read as a JSON file
    with open(args.entities, "r", encoding="utf-8") as f:
        qids = json.load(f)
    list_entities = qids["wikidata"]

    # Dictionary to save all changes (recovered revisions)
    all_modifications = {}

    # for statistical purpose
    total_revisions = 0

    # Unit labels (tons, centimeters, etc.) are stored in a
    # dictionary to avoid multiple requests for the same data
    units = {}
    # The same applies to property labels
    property_labels = {}

    # Store revisions in a folder with the same name as the JSON file
    directory = args.entities.replace(".json", "")
    dir_path = os.path.join(os.getcwd(), directory)
    os.makedirs(dir_path, exist_ok=True)

    # Statistics on the number of items to be labeled
    distinct_values = {}
    global_stats = {
        "modifications": [0],
        "additions": [0],
        "creations": [0]
    }

    # Statistics on the number of distinct contributors (humans and bots)
    distinct_users = set()

    # To retrieve revisions of Wikidata entities in parallel
    def retrieve_all_revisions_parallel(entity_ids: List[str],
                                        units: Dict,
                                        property_labels,
                                        global_stats,
                                        all_modifications,
                                        distinct_values,
                                        distinct_users,
                                        txn,
                                        dir_path,
                                        prop_types,
                                        max_workers=5):
        """
        Retrieves all revisions for a list of entities in parallel.

        This function uses a ThreadPoolExecutor to retrieve revisions for each QID in parallel. 
        The results are collected and returned as a dictionary.

        Args:
            entity_ids (list): A list of QIDs for which revisions are to be retrieved.
            units (str): The units parameter to be passed to the get_entity_revisions function.
            property_labels (dict): The property labels to be used in the revisions.
            global_stats (dict): A dictionary to collect global statistics.
            all_modifications (list): A list to store all modifications.
            distinct_values (set): A set to collect distinct values foir stats.
            distinct_users (set): A set to collect distinct users for stats.
            txn (object): The transaction object to be used for database operations and to get labels.
            dir_path (str): The directory path where any necessary files will be stored.
            max_workers (int, optional): The maximum number of threads to use for parallel execution. Defaults to 5.

        Returns:
            dict: A dictionary where keys are entity IDs and values are the retrieved revisions.

        Raises:
            Exception: If an error occurs during the retrieving of revisions for an entity, 
                    it will be printed but not re-raised.
        """
        results = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Parallelization on the list of entities and call to the revision retrieval function
            future_to_entity = {executor.submit(get_entity_revisions,
                                                entity_id,
                                                units,
                                                property_labels,
                                                global_stats,
                                                all_modifications,
                                                distinct_values,
                                                distinct_users,
                                                txn,
                                                dir_path,
                                                prop_types): entity_id for entity_id in entity_ids}
            for future in tqdm.tqdm(concurrent.futures.as_completed(future_to_entity), total=len(entity_ids)):
                entity_id = future_to_entity[future]
                try:
                    data = future.result()
                    results.update(data)
                except Exception as exc:
                    print(f'Entity {entity_id} generated an exception: {exc}')
        return results

    # Dictionary that will contains the types of the properties
    prop_types = {}

    # Recovering revisions
    retrieve_all_revisions_parallel(list_entities,
                                    units,
                                    property_labels,
                                    global_stats,
                                    all_modifications,
                                    distinct_values,
                                    distinct_users,
                                    txn,
                                    dir_path,
                                    prop_types,
                                    32)

    # Statistics
    nb_of_values_to_check = 0

    # Statistics
    global_stats_dataset = pandas.DataFrame(global_stats)
    global_stats_dataset.to_csv(os.path.join(dir_path, "global_stats.csv"))

    print(f"Total number of revisions for the Wikipedia category: {total_revisions}")

    for qid, revisions in all_modifications.items():
        for _, contributions in revisions.items():
            for prop, prop_content in contributions.items():
                if prop in distinct_values[qid]:
                    distinct_values[qid][prop] |= prop_content

    for qid, values in distinct_values.items():
        for _, prop_content in values.items():
            nb_of_values_to_check += len(prop_content)

    print(f"Number of values to label for Wikipedia category: {nb_of_values_to_check}")

    print(f"# creations = {global_stats['creations'][0]}")
    print(f"# additions = {global_stats['additions'][0]}")
    print(f"# modifications = {global_stats['modifications'][0]}")

    print(f"# distinct users = {len(distinct_users)}")

    # pickle.dump(data, open(args.stats, "wb"))
    pickle.dump(property_labels, open('property_labels.pkl', 'wb'))
    pickle.dump(prop_types, open(args.prop_types, 'wb'))


if __name__ == "__main__":
    main()
