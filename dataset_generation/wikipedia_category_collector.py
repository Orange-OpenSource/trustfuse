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
import csv
import re
import unicodedata

import requests
from tqdm import tqdm

NAMESPACE_ID = 14


base_url = "https://fr.wikipedia.org/w/api.php"
folder = "./resources/wikimedia/raw"


def get_wikidata_qid(wikipedia_urls):
    """Retrieves Wikidata QIDs for a list of Wikipedia URLs.

    This function takes a list of Wikipedia URLs, extracts the titles from these URLs, and queries the MediaWiki API
    to get the corresponding Wikidata QIDs. It returns a dictionary where the keys are the original Wikipedia URLs
    and the values are the Wikidata QIDs.

    Args:
        wikipedia_urls (list of str): A list of Wikipedia URLs.

    Returns:
        dict: A dictionary mapping each Wikipedia URL to its corresponding Wikidata QID. If a QID cannot be found,
        the value for that URL will be `None`.
    """
    qid_map = {}
    for url in tqdm(wikipedia_urls, desc="Converting URLs to QIDs"):
        title = url.split("/wiki/")[-1]
        params = {
            "action": "query",
            "format": "json",
            "titles": title,
            "prop": "pageprops",
        }
        response = requests.get(base_url, params=params)
        data = response.json()
        pages = data.get("query", {}).get("pages", {})
        for _, page_data in pages.items():
            if "pageprops" in page_data and "wikibase_item" in page_data["pageprops"]:
                qid = page_data["pageprops"]["wikibase_item"]
                qid_map[qid] = url
    return qid_map


def get_category_members(name):
    """Retrieves category members for a given Wikipedia category.

    This function queries the MediaWiki API to obtain list of pages that link to specified Wikipedia category.
    The function fetches members of categories and returns a list of URLs.

    Args:
        name (str): The title of the category for which to retrieve members.
        It should be in the format used in Wikipedia URLs.

    Returns:
        list of str: A list of URLs of pages that link to the specified Wikipedia category.
        The URLs are formatted as: "https://fr.wikipedia.org/wiki/Title".
    """
    params = {
        "action": "query",
        "format": "json",
        "list": "categorymembers",
        "cmtitle": name,
        "blnamespace": "0|14",  # Only include main articles and categories (see https://en.wikipedia.org/wiki/Wikipedia:Namespace)
    }
    members = []
    continue_token = True
    while continue_token:
        response = requests.get(base_url, params=params)
        data = response.json()
        if "query" in data:
            # Update the progress bar for each page processed
            for member in data["query"]["categorymembers"]:
                page_title = member["title"]
                page_url = (
                    f"https://fr.wikipedia.org/wiki/{page_title.replace(' ', '_')}"
                )
                namespace = member.get("ns", 0)
                members.append((page_url, namespace))
        if "continue" in data:
            params.update(data["continue"])
        else:
            continue_token = False
    return members


def process_members(name):
    """Processes members for a given Wikipedia category handling categories recursively.

    This function retrieves members for a specified Wikipedia category, processes each to determine if
    it is a category, and if so, retrieves and processes its members recursively. It updates a progress bar to
    reflect the processing status and returns a list of URLs that are not categories.

    Args:
        name (str): The title of the Wikipedia categoryfor which to retrieve and process members.
        It should be in the format used in Wikipedia URLs.

    Returns:
        list of str: A list of URLs of Wikipedia pages that link to the specified category.
        The URLs are formatted as: "https://fr.wikipedia.org/wiki/Title".
    """
    # Get the members
    members = get_category_members(name)
    final_urls = []
    # List of URLS already processed
    processed_urls = set()
    for _ in range(4):
        new_members = []
        with tqdm(total=len(members), desc=str("Processing members of " + name)) as pbar:
            while members:
                url, namespace = members.pop(0)
                # Skip processing if the URL has already been processed
                if url in processed_urls:
                    continue
                processed_urls.add(url)
                # If the URL is a category (namespace=14), get its members and add them to the list
                if namespace == 14:
                    category_title = url.split("/wiki/")[-1]
                    category_members = get_category_members(category_title)
                    for cat_url, cat_namespace in category_members:
                        # Add only pages to the members list and not categories
                        if cat_namespace != NAMESPACE_ID and cat_url not in processed_urls:
                            members.append((cat_url, cat_namespace))
                        elif cat_namespace == NAMESPACE_ID and cat_url not in processed_urls:
                            new_members.extend(get_category_members(category_title))
                else:
                    final_urls.append(url)
                # Update the progress bar after each URL is processed
                pbar.update(1)
        members = new_members
    return final_urls


def convert_qid_to_wikidata(json_data, target_qid):
    """Converts a single QID from a custom knowledge graph to its corresponding Wikidata QID.

    Using the properties P7 (equivalent class) and P8 (same as).

    Args:
        json_data (str): Path to the JSON file containing entities from the knowledge graph.
        target_qid (str): The QID from the knowledge graph that you want to convert to a Wikidata QID.

    Returns:
        dict: A dictionary where keys are the original QIDs from the knowledge graph
              and values are the corresponding Wikidata QIDs.
    """
    for item in json_data:
        original_qid = item.get("id")
        if original_qid == target_qid:
            claims = item.get("claims", {})
            # Check for P7 (equivalent class) property
            if "P7" in claims:
                for statement in claims["P7"]:
                    wikidata_url = statement["mainsnak"]["datavalue"]["value"]
                    # Extract QID from the URL
                    wikidata_qid = wikidata_url.split("/")[-1]
                    return wikidata_qid
            # Check for P8 (same as) property
            elif "P8" in claims:
                for statement in claims["P8"]:
                    wikidata_url = statement["mainsnak"]["datavalue"]["value"]
                    # Extract QID from the URL
                    wikidata_qid = wikidata_url.split("/")[-1]
                    return wikidata_qid
    return None


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--categories-list", 
                        dest='categories_list',
                        help="File containing one Wikipedia category per line",
                        required=True)
    parser.add_argument("--language",
                        help="Language of the Wikipedia category",
                        choices=["fr", "en", "es", "ru"],
                        required=False,
                        default="en",
                        type=str)
    args = parser.parse_args()

    category_names = []
    with open(args.categories_list, 'r', encoding='utf-8') as f:
        categories = f.readlines()
        category_names = [category for category in categories]
        
    for category_name in tqdm(category_names, desc="Processing categories"):
        # Get all Wikipedia pages pointing to the category
        members = process_members(category_name)
        # Filter out URLs that are categories
        members = [url for url in members if "/wiki/Catégorie:" not in url]
        # Remove duplicates
        members = list(set(members))
        # Convert Wikipedia page URL to Wikidata QID
        qid_map = get_wikidata_qid(members)
        # Export results
        filename = category_name.replace(" ", "_").lower()
        filename = unicodedata.normalize("NFKD", filename)
        filename = filename.encode("ASCII", "ignore").decode("ASCII")
        filename = re.sub(r'[:/\\?%*"<>|]', "-", filename)
        filename = filename + ".csv"
        with open(filename, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(qid_map)


if __name__ == "__main__":
    main()
