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

import math
import pickle
import numpy


def get_hashmap_content(key, hashmap):
    obj = hashmap.get(key.encode("ascii"))
    content = None
    if obj:
        content = pickle.loads(obj)
    return content


def euclidean_distance(v1, v2):
    """ Compute the Euclidean distance """
    return numpy.sqrt(sum(pow(v1 - v2, 2)))


def jaro_distance(s1, s2):
	# If the s are equal
    if s1 == s2:
        return 1.0
	# Length of two s
    len1 = len(s1)
    len2 = len(s2)
	# Maximum distance upto which matching is allowed
    max_dist = math.floor(max(len1, len2) / 2) - 1
	# Count of matches
    match = 0
	# Hash for matches
    hash_s1 = [0] * len(s1)
    hash_s2 = [0] * len(s2)
    for i in range(len1):
		# Check if there is any matches
        for j in range(max(0, i - max_dist), min(len2, i + max_dist + 1)):
			# If there is a match
            if (s1[i] == s2[j] and hash_s2[j] == 0):
                hash_s1[i] = 1
                hash_s2[j] = 1
                match += 1
                break
    if match == 0:
        return 0.0
    t = 0
    point = 0
    for i in range(len1):
        if hash_s1[i]:
			# Find the next matched character in second
            while hash_s2[point] == 0:
                point += 1
            if s1[i] != s2[point]:
                t += 1
            point += 1
    t = t // 2
    return (match / len1 + match / len2 + (match - t) / match) / 3.0
