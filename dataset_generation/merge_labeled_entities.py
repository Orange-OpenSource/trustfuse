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
import pickle
import re
from datetime import datetime

import pandas


DATA_FOLDER_PATH = "data"


def date2seconds(date):
    """Transforms a date into seconds

    Args:
        date (string): human readable date

    Returns:
        int: the date converted in seconds
    """
    datetime_obj = datetime.strptime(date, "%Y-%m-%dT%H:%M:%SZ")
    seconds = int(datetime_obj.timestamp())
    return seconds

def main():

    with open(os.path.join("files", "buckets_by_qid_paris.pkl"), "rb") as f:
        buckets_by_qid = pickle.load(f)

    data_folder = [
        os.path.join(DATA_FOLDER_PATH, entity_file)
        for entity_file in os.listdir(DATA_FOLDER_PATH)
        ]

    end_times = []
    end_buckets = {}

    for p in data_folder:
        match = re.search(r'Q\d+', p)
        if match:
            qid = match.group()
            with open(p, "rb") as f:
                qid_buckets = pickle.load(f)
            buckets = buckets_by_qid[qid]['buckets']
            for i, bucket in enumerate(buckets):
                end_time = date2seconds(bucket["end_time"])
                end_buckets[end_time] = qid_buckets[i]
                end_times.append(end_time)

    end_times.sort()
    sorted_buckets = {}

    for i, end_time in enumerate(end_times):
        sorted_buckets[i] = end_buckets[end_time]

    with open("merged_labeled_entities.pkl", "wb") as f:
        pickle.dump(sorted_buckets, f)

if __name__ == "__main__":
    main()
