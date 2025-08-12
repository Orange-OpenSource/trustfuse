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
import concurrent.futures
import random

import requests
import tqdm
import time


def get_values(entity_id, prop, objects):
    """Extract triples from Wikidata page content

    Args:
        entity_id (str): Wikidata QID
        prop (str): Wikidata PID
        objects (_type_): triples present in the Wikidata page content

    Returns:
        (set, set): set of subgraph triples and a set of entities (QIDs) as objects of the triples
    """
    values = set()
    entities = set()
    for value in objects:
        if "datavalue" in value["mainsnak"]:
            try:
                # quantity
                if value["mainsnak"]["datavalue"]["type"] == "quantity":
                    unit = value["mainsnak"]["datavalue"]["value"]["unit"]
                    values.add((entity_id,
                                prop,
                                value["mainsnak"]["datavalue"]["value"]["amount"] + unit))

                # string
                if value["mainsnak"]["datavalue"]["type"] == "string":
                    values.add((entity_id,
                                prop,
                                value["mainsnak"]["datavalue"]["value"]))

                # time
                if value["mainsnak"]["datavalue"]["type"] == "time":
                    time_value = value["mainsnak"]["datavalue"]["value"]["time"].replace("+", "")
                    time_value = time_value.replace("T00:00:00Z", "")
                    values.add((entity_id,
                                prop,
                                time_value))

                # globecoordinate
                if value["mainsnak"]["datavalue"]["type"] == "globecoordinate":
                    values.add((entity_id,
                                prop + 'lat',
                                value["mainsnak"]["datavalue"]["value"]["latitude"]))
                    values.add((entity_id,
                                prop + 'lon',
                                value["mainsnak"]["datavalue"]["value"]["longitude"]))

                # monolingualtext
                if value["mainsnak"]["datavalue"]["type"] == "monolingualtext":
                    values.add((entity_id,
                                prop,
                                value["mainsnak"]["datavalue"]["value"]["text"]))

                # wikibase-entityid
                if value["mainsnak"]["datavalue"]["type"] == "wikibase-entityid":
                    values.add((entity_id,
                                prop,
                                value["mainsnak"]["datavalue"]["value"]["id"]))
                    entities.add(value["mainsnak"]["datavalue"]["value"]["id"])
            except:
                continue

    return values, entities


def get_wikidata_page_content(entity_id, depth):
    """Retrieves the content of a Wikidata page.

    Args:
        entity_id (str): Wikidata QID
        depth (int): subgraph depth

    Returns:
        Set: set of triples that constitute the subgraph
    """
    url = 'https://www.wikidata.org/w/api.php'

    entities_to_retrieve = {entity_id}
    triples = set()

    d = 0
    while d < depth:
        entities = set()
        for ent in entities_to_retrieve:
            params = {
                'action': 'wbgetentities',
                'format': 'json',
                'ids': ent,
                'props': 'labels|descriptions|claims|sitelinks'
            }
            response = None
            # Loop to ensure API response
            while response is None:
                # Query the Wikidata API with the above query parameters
                try:
                    response = requests.get(url, params=params)
                    data = response.json()
                    # print(data)
                    for prop, values in data['entities'][ent]['claims'].items():
                        new_triples, new_entities = get_values(ent, prop, values)
                        entities |= new_entities
                        triples |= new_triples

                except Exception as err:
                    time.sleep(30)
                    response = requests.get(url, params=params)
                    print(response)
                    print(f'Entity {entity_id} generated an exception: {err}')
        entities_to_retrieve = entities
        d += 1

    return triples, entities_to_retrieve


def main():

    parser = argparse.ArgumentParser(
        prog="Generate Wikidata subgraph from a list of entities")
    parser.add_argument("--entities",
                        help="list of entities of interest",
                        required=True)
    parser.add_argument("--depth",
                        help="Depth of the subgraph",
                        required=False,
                        type=int,
                        default=2)
    parser.add_argument("--deletion-ratio",
                        dest="deletion_ratio",
                        help="Deletion ratio of triples in the subgraph",
                        required=False,
                        type=int,
                        default=50)
    parser.add_argument("--output",
                        help="Resulting subgraph",
                        required=True)
    args = parser.parse_args()

    # Interest QIDs (Wikipedia category) are read as a JSON file
    with open(args.entities, encoding="utf-8") as f:
        qids = json.load(f)
    list_entities = qids["wikidata"]

    # To retrieve revisions of Wikidata entities in parallel
    def retrieve_subgraphs_parallel(entity_ids, depth, max_workers=20):
        triples = set()
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_entity = {executor.submit(get_wikidata_page_content,
                                                entity_id,
                                                depth): entity_id for entity_id in entity_ids}
            for future in tqdm.tqdm(concurrent.futures.as_completed(future_to_entity),
                                    total=len(entity_ids)):
                entity_id = future_to_entity[future]
                try:
                    data = future.result()
                    triples |= data[0]
                except Exception as e:
                    print(f'Entity {entity_id} generated an exception: {e}')
        return triples, None

    triples, _ = retrieve_subgraphs_parallel(list_entities,
                                              args.depth,
                                              max_workers=8)

    nb_to_remove = round(len(triples) * args.deletion_ratio / 100)
    to_remove = random.sample(list(triples), nb_to_remove)
    triples -= set(to_remove)

    with open(args.output, "wb") as f:
        pickle.dump(triples, f)


if __name__ == "__main__":
    main()
