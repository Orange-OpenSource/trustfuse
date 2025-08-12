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
import copy
from datetime import datetime
import ipaddress
import pickle
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tqdm


def is_ip_or_mac(s):
    """Detects is a source s is a IP or MAC address

    Args:
        s (str): the name of the source

    Returns:
        bool: True if the source is a IP or MAC address, False otherwise
    """
    mac_regex = re.compile(
        r"^([0-9A-Fa-f]{2}[:-]){5}([0-9A-Fa-f]{2})$|"
        r"^[0-9A-Fa-f]{4}\.[0-9A-Fa-f]{4}\.[0-9A-Fa-f]{4}$"
    )

    try:
        # Check if it is a valid IP (IPv4 or IPv6)
        ipaddress.ip_address(s)
        return True
    except ValueError:
        # Check if it is a MAC address
        return bool(mac_regex.match(s))


def merge_intervals(intervals, gamma):
    """Merges intervals if the gap between them is smaller than T.

    Args:
        intervals (list of list of int): A sorted list of intervals [start, end].
        T (int): The maximum allowed gap for merging intervals.

    Returns:
        list of list of int: The merged intervals.
    """
    if not intervals:
        return []

    merged = [intervals[0]]

    for start, end, color in intervals[1:]:
        last_start, last_end, last_color = merged[-1]

        if start - last_end < gamma:
            merged[-1] = [
                last_start,
                end,
                last_color + color + [[last_end, start, "orange"]],
            ]
        else:
            merged.append([start, end, color])

    return merged


def compute_presence_intervals(
    data, qid, prop, gamma, beta, buckets_gt, plot=False, stop=1_723_240_800
):
    """
    Computes the time intervals during which each value is present
    in the dataset.

    A value's presence interval remains open if it disappears but
    reappears within `gamma` seconds.
    If an interval's duration is less than `beta`, it is ignored.

    Args:
        data (dict): A dictionary where keys are timestamps (seconds)
        and values are lists of present values.
        gamma (int): The maximum allowed gap (in seconds) for a value
        to reappear without closing its interval.
        beta (int): The minimum duration (in seconds) for an interval
        to be considered valid.

    Returns:
        dict: A dictionary with two keys:
            - "valid": A dictionary where keys are values and values are lists
            of (start, end) tuples representing presence intervals.
            - "invalid": A dictionary where keys are values and values are lists
            of (start, end) tuples that did not meet the validity criteria.
    """
    # *** The construction of intervals are made in 3 steps
    # 1st step: Simply construct the initial intervals
    # 2nd step: Check if the continuity of the values
    # and if an interruption is below a gamma threshold
    # 3rd step: Then, check if ecah interval is relevant through a beta threshold
    intervals_first_step = {}
    intervals_second_step = {}
    valid_intervals = {}
    invalid_intervals = {}
    active_values = {}
    # Sort values by timestamps (in seconds)
    time_list = sorted(data[qid][prop].keys())

    # *** 1st step: Intervals construction
    for _, t in enumerate(time_list):
        # Set of values for given T,
        # e.g.: {('French general (1867–1946)', 'French general (1867–1946)')}
        current_values = set(data[qid][prop][t])

        # Creation of the intervals
        # Add new values
        removed_values = set(active_values.keys()) - current_values
        for removed_val in removed_values:
            if removed_val not in intervals_first_step:
                intervals_first_step[removed_val] = []
            active_values[removed_val][1] = t
            active_values[removed_val][2][0][1] = t
            intervals_first_step[removed_val].append(active_values[removed_val])
            del active_values[removed_val]

        for val in current_values:
            if val not in active_values:
                active_values[val] = [
                    t,
                    float("inf"),
                    [[t, float("inf"), "black"]]
                    ]
    for remaining_val, _ in active_values.items():
        if remaining_val not in intervals_first_step:
            intervals_first_step[remaining_val] = []
        interval = active_values[remaining_val]
        interval[1] = stop
        interval[2][0][1] = stop
        intervals_first_step[remaining_val].append(interval)

    # *** Second step: merge intervals if the values are below a gamma threshold
    for val, intervals in intervals_first_step.items():
        intervals_second_step[val] = merge_intervals(intervals, gamma)

    # *** 3rd step: check the minimum of period for validity
    for val, intervals in intervals_second_step.items():
        for start_time, end_time, colors in intervals:
            if end_time - start_time < beta:
                if val not in invalid_intervals:
                    invalid_intervals[val] = []
                invalid_intervals[val].append([start_time, end_time, colors])
            else:
                if val not in valid_intervals:
                    valid_intervals[val] = []
                valid_intervals[val].append([start_time, end_time, colors])

    # *** For display purpose
    if plot:
        time_list = [
            seconds2date(t)
            for t in set(time_list) | buckets_gt[qid]["times"][prop] | {stop}
        ]
        times = sorted(
            time_list, key=lambda x: datetime.strptime(x, "%Y-%m-%dT%H:%M:%SZ")
        )

        time_values = [datetime.strptime(t, "%Y-%m-%dT%H:%M:%SZ") for t in times]
        unique_labels = sorted(set(valid_intervals.keys()) | set(invalid_intervals))
        label_to_index = {label: i for i, label in enumerate(unique_labels)}
        time_positions = np.linspace(0, len(time_values) - 1, len(time_values))
        time_mapping = dict(zip(time_values, time_positions))
        first_occurrence = {}

        _, ax = plt.subplots(figsize=(12, 6))

        for time in time_values:
            # Plot a vertical line for each unique value on X-axis
            ax.axvline(
                time_mapping[time],
                color="gray",
                linestyle="dotted",
                linewidth=0.5,
                zorder=0,
            )

        for val, intervals in valid_intervals.items():
            for _, _, colors in intervals:
                for start_time, end_time, color in colors:
                    ax.scatter(
                        time_mapping[seconds2time(start_time)],
                        label_to_index[val],
                        color="black",
                        zorder=2,
                    )
                    ax.scatter(
                        time_mapping[seconds2time(end_time)],
                        label_to_index[val],
                        color="black",
                        zorder=2,
                    )
                    ax.plot(
                        [
                            time_mapping[seconds2time(start_time)],
                            time_mapping[seconds2time(end_time)],
                        ],
                        [label_to_index[val], label_to_index[val]],
                        color=color,
                        linestyle="-",
                        linewidth=2,
                        zorder=1,
                    )

                    if val not in first_occurrence:
                        first_occurrence[val] = time_mapping[seconds2time(start_time)]
            if val in buckets_gt[qid][prop]:
                ax.plot(
                    [
                        time_mapping[
                            seconds2time(date2seconds(buckets_gt[qid][prop][val][0][0]))
                        ],
                        time_mapping[seconds2time(stop)],
                    ],
                    [label_to_index[val] - 0.2, label_to_index[val] - 0.2],
                    color="blue",
                    linestyle="-",
                    linewidth=2,
                    zorder=1,
                )
                ax.scatter(
                    time_mapping[
                        seconds2time(date2seconds(buckets_gt[qid][prop][val][0][0]))
                    ],
                    label_to_index[val] - 0.2,
                    color="blue",
                    zorder=2,
                    marker="|",
                )
                ax.scatter(
                    time_mapping[seconds2time(stop)],
                    label_to_index[val] - 0.2,
                    color="blue",
                    zorder=2,
                    marker="|",
                )

        for val, intervals in invalid_intervals.items():
            for start_time, end_time, colors in intervals:
                ax.scatter(
                    time_mapping[seconds2time(start_time)],
                    label_to_index[val],
                    color="black",
                    zorder=2,
                )
                ax.scatter(
                    time_mapping[seconds2time(end_time)],
                    label_to_index[val],
                    color="black",
                    zorder=2,
                )
                ax.plot(
                    [
                        time_mapping[seconds2time(start_time)],
                        time_mapping[seconds2time(end_time)],
                    ],
                    [label_to_index[val], label_to_index[val]],
                    color="red",
                    linestyle="-",
                    linewidth=2,
                    zorder=1,
                )
            if val not in first_occurrence:
                first_occurrence[val] = time_mapping[seconds2time(start_time)]

        # Display value labels slightly above each value point
        for label, time in first_occurrence.items():
            ax.text(
                time,
                label_to_index[label] + 0.025,
                label,
                fontsize=10,
                ha="right",
                va="bottom",
                color="black",
            )

        ax.set_yticks([])
        plt.xticks(
            time_positions,
            [t.strftime("%Y-%m-%d %H:%M") for t in time_values],
            rotation=40,
            fontsize=8,
        )
        plt.xlabel("Time")
        plt.grid(True, which="both", linestyle="--", linewidth=0.5)
        plt.show()

    return valid_intervals, invalid_intervals


def seconds2date(seconds):
    """Transforms seconds into a human readable date

    Args:
        seconds (int): date in seconds

    Returns:
        string: human readable date format
    """
    datetime_obj = datetime.fromtimestamp(seconds)
    date = datetime_obj.strftime("%Y-%m-%dT%H:%M:%SZ")
    return date


def seconds2time(seconds):
    return datetime.strptime(seconds2date(seconds), "%Y-%m-%dT%H:%M:%SZ")


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

    parser = argparse.ArgumentParser()
    parser.add_argument("--modifications", required=True)
    parser.add_argument("--buckets", required=True)
    parser.add_argument("--gt", required=True)
    args = parser.parse_args()

    # All buckets for all QIDs
    with open(args.buckets, "rb") as f:
        buckets_by_qid = pickle.load(f)

    static_props_per_bucket = {}
    seen_values = {}
    for qid in buckets_by_qid:
        seen_values[qid] = {i: {} for i in range(len(buckets_by_qid[qid]["buckets"]))}
        static_props_per_bucket[qid] = {}
        for i, bucket in enumerate(buckets_by_qid[qid]["buckets"]):
            static_props_per_bucket[qid][i] = set()
            if i - 1 in seen_values[qid]:
                seen_values[qid][i].update(copy.deepcopy(seen_values[qid][i - 1]))
            for contrib in bucket["only-bucket"]:
                static_props_per_bucket[qid][i].add(contrib[1])
                prop = contrib[1]
                value = contrib[2]
                value_id = contrib[3]
                if prop not in seen_values[qid][i]:
                    seen_values[qid][i][prop] = set()
                seen_values[qid][i][prop].add((value, value_id))

    # Buckets manually labeled
    with open(args.gt, "rb") as f:
        gt = pickle.load(f)

    with open(args.modifications, "rb") as f:
        modifications_by_timestamp_qid = pickle.load(f)

    qids_of_interest = list(buckets_by_qid.keys())

    buckets_gt = {}

    correct_values = {}

    labeled_qids = set()

    for bid in gt:
        qid = gt[bid]["GT"]["value"]["Entity"].iloc[0]
        labeled_qids.add(qid)
        if qid not in correct_values:
            correct_values[qid] = {}

        for prop, prop_values in gt[bid]["GT"]["qid"].items():
            if prop != "Entity":
                if prop not in correct_values[qid]:
                    correct_values[qid][prop] = set()
                for val in prop_values.iloc[0]:
                    correct_values[qid][prop].add(val)

    sources = {}

    # Add the bucket start and end time to the labeled data
    for qid in qids_of_interest:
        if qid not in sources:
            sources[qid] = {}
        if qid not in buckets_gt:
            # "times" key will be used to generate value intervals
            buckets_gt[qid] = {"times": {}}
        # bucket_by_qid = output of generate_bucket.py
        for bucket in buckets_by_qid[qid]["buckets"]:
            start_time = bucket["start_time"]
            end_time = bucket["end_time"]
            for claim in bucket["only-bucket"]:
                prop = claim[1]
                val = claim[2]
                val_qid = claim[3]
                if prop not in sources[qid]:
                    sources[qid][prop] = {}
                if (val, val_qid) not in sources[qid][prop]:
                    sources[qid][prop][(val, val_qid)] = []
                sources[qid][prop][(val, val_qid)].append(claim[0])
                val_time = claim[-1]
                # correct values = data manually labeled
                if qid in correct_values and prop in correct_values[qid]:
                    if prop not in buckets_gt[qid]:
                        # To store each point in time for the property
                        buckets_gt[qid]["times"][prop] = set()
                        # To store the time intervals
                        buckets_gt[qid][prop] = {}
                    # Add the start/end time only for correct values
                    # to compute automatic labeling metrics
                    if val_qid in correct_values[qid][prop]:
                        if (val, val_qid) not in buckets_gt[qid][prop]:
                            buckets_gt[qid][prop][(val, val_qid)] = [
                                [val_time, end_time]
                            ]
                            buckets_gt[qid]["times"][prop].add(date2seconds(val_time))
                            buckets_gt[qid]["times"][prop].add(date2seconds(end_time))
                else:
                    buckets_gt[qid][prop] = {}
            for prop in buckets_gt[qid]:
                if prop != "times":
                    for val in buckets_gt[qid][prop]:
                        if date2seconds(
                            buckets_gt[qid][prop][val][0][0]
                        ) < date2seconds(start_time):
                            buckets_gt[qid][prop][val].append([start_time, end_time])
                            buckets_gt[qid]["times"][prop].add(date2seconds(val_time))
                            buckets_gt[qid]["times"][prop].add(date2seconds(end_time))

    props_information = {}

    # To compute metrics
    tp, fp, tn, fn = (0.01, 0.01, 0.01, 0.01)

    # Label each (object, property) pair
    for qid, properties in tqdm.tqdm(
        buckets_gt.items(), desc="Estimating valid values"
    ):
        # Initialize
        props_information[qid] = {}
        for prop, _ in properties.items():
            if prop != "times":
                # Initialize
                props_information[qid][prop] = []
                rates = {}
                valid_intervals, _ = compute_presence_intervals(
                    modifications_by_timestamp_qid,
                    qid,
                    prop,
                    2_592_000,
                    2_592_000,
                    buckets_gt,
                    plot=False,
                )

                for val, val_intervals in valid_intervals.items():
                    rates[val] = 0
                    for interval in val_intervals:
                        rates[val] += interval[1] - interval[0]
                all_rates = [rates[val] for val in rates]
                # rates_sum = sum(all_rates)
                rate_max = 0
                if len(all_rates) > 0:
                    rate_max = max(all_rates)
                for val, rate in rates.items():
                    if rate_max > 0:
                        # r1 = rate / rates_sum
                        r2 = rate / rate_max
                        counter_ip = 0
                        ip_majority = False
                        for source in sources[qid][prop][val]:
                            if is_ip_or_mac(source):
                                counter_ip += 1
                        if counter_ip > (len(sources[qid][prop][val]) / 2):
                            ip_majority = True
                        if r2 > 0.75 and not ip_majority:
                            if len(buckets_gt[qid][prop]) > 0:
                                if val in buckets_gt[qid][prop]:
                                    tp += 1
                                else:
                                    fp += 1
                            props_information[qid][prop].append(val)

                        else:
                            if len(buckets_gt[qid][prop]) > 0:
                                if val in buckets_gt[qid][prop]:
                                    fn += 1
                                else:
                                    tn += 1
    # Compute metrics
    p, r, f1, acc = (0, 0, 0, 0)
    if tp + fp > 0:
        p = tp / (tp + fp)
    if tp + fn > 0:
        r = tp / (tp + fn)
    if tp + fp + tn + fn > 0:
        acc = (tp + tn) / (tp + fp + tn + fn)
    if r + p > 0:
        f1 = 2 * p * r / (p + r)
    p = tp / (tp + fp)

    print(f"Precision = {p}")
    print(f"Recall = {r}")
    print(f"F1-score = {f1}")
    print(f"Accuracy = {acc}")

    end_times = []
    end_buckets = {}

    # Ground Truth format to save automatic labeling
    for qid in tqdm.tqdm(qids_of_interest, desc="Prepare the dataset"):
        buckets_df = {}
        bucket_qids = {}
        for i, bucket in enumerate(buckets_by_qid[qid]["buckets"]):
            buckets_df[i] = {}
            bucket_claims = {}
            bucket_qids = {}
            for prop in static_props_per_bucket[qid][i]:
                gt_values = []
                gt_qids = []
                labeled_values = props_information[qid][prop]
                if labeled_values is not None:
                    gt_values = [
                        [
                            val[0]
                            for val in labeled_values
                            if val in seen_values[qid][i][prop]
                        ]
                    ]
                    gt_qids = [
                        [
                            val[1]
                            for val in labeled_values
                            if val in seen_values[qid][i][prop]
                        ]
                    ]
                    if len(gt_values) == 0:
                        gt_values = [[None]]
                        gt_qids = [[None]]
                else:
                    gt_values = [[None]]
                    gt_qids = [[None]]
                bucket_claims[prop] = gt_values
                bucket_qids[prop] = gt_qids
            bucket_claims["Entity"] = [qid]
            bucket_qids["Entity"] = [qid]

            bucket_claims_df = pd.DataFrame(bucket_claims)
            bucket_qids_df = pd.DataFrame(bucket_qids)
            buckets_df[i]["GT"] = {}
            buckets_df[i]["GT"]["value"] = bucket_claims_df
            buckets_df[i]["GT"]["qid"] = bucket_qids_df

            buckets_df[i]["GT"]["value_order"] = {}
            buckets_df[i]["GT"]["qid_order"] = {}

            index = i
            bucket_mapping = buckets_by_qid[qid]["buckets"][index]["only-bucket"]
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
            bucket_claims_value["Source"] = []
            bucket_claims_qid["Source"] = []
            for source, claims in claims_by_source.items():
                max_attribute_values = max(len(v) for v in claims.values())
                bucket_claims_value["Source"].extend([source] * max_attribute_values)
                bucket_claims_qid["Source"].extend([source] * max_attribute_values)
                for prop in distinct_properties:
                    if prop in claims:
                        for val in claims[prop]:
                            bucket_claims_value[prop].append(val[0])
                            bucket_claims_qid[prop].append(val[1])
                        # Complete with None to create the Dataframe correctly
                        bucket_claims_value[prop].extend(
                            [
                                None
                                for _ in range(max_attribute_values - len(claims[prop]))
                            ]
                        )
                        bucket_claims_qid[prop].extend(
                            [
                                None
                                for _ in range(max_attribute_values - len(claims[prop]))
                            ]
                        )
                    else:
                        # We fill the prop with None if the source has provided no value for it
                        bucket_claims_value[prop].extend(
                            [None for _ in range(max_attribute_values)]
                        )
                        bucket_claims_qid[prop].extend(
                            [None for _ in range(max_attribute_values)]
                        )

            bucket_claims_value["Entity"] = [qid for _ in bucket_claims_value["Source"]]
            bucket_claims_qid["Entity"] = [qid for _ in bucket_claims_value["Source"]]
            buckets_df[i]["data"] = {}
            buckets_df[i]["data"]["value"] = pd.DataFrame(bucket_claims_value)
            buckets_df[i]["data"]["qid"] = pd.DataFrame(bucket_claims_qid)
            end_time = date2seconds(bucket["end_time"])
            end_buckets[end_time] = buckets_df[i]
            end_times.append(end_time)

    end_times.sort()
    sorted_buckets = {}

    for i, end_time in enumerate(end_times):
        sorted_buckets[i] = end_buckets[end_time]

    with open("auto_gt_buckets.pkl", "wb") as f:
        pickle.dump(sorted_buckets, f)


if __name__ == "__main__":
    main()
