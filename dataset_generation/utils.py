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

from datetime import datetime


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


def convert_seconds_to_readable_time(seconds):
    """Counts the number of days, hours, minutes, and seconds

    Args:
        seconds (int): the amount of seconds

    Returns:
        string: amount of days, hours, minutes and remaining seconds in seconds
    """
    days = seconds // 86400
    hours = (seconds % 86400) // 3600
    minutes = (seconds % 3600) // 60
    remaining_seconds = seconds % 60
    
    readable_time = ''
    if days > 0:
        readable_time += f"{days} day{'s' if days > 1 else ''} "
    if hours > 0:
        readable_time += f"{hours} hour{'s' if hours > 1 else ''} "
    if minutes > 0:
        readable_time += f"{minutes} minute{'s' if minutes > 1 else ''} "
    if remaining_seconds > 0:
        readable_time += f"{remaining_seconds} second{'s' if remaining_seconds > 1 else ''}"
    
    readable_time = readable_time.strip()
    
    return readable_time
