import re
import pandas as pd
import numpy as np

from typing import List

import tqdm


def focus_on_attributes(dataset, attributes, action='repeat', progress=tqdm):
    """Apply Focus transformation

    Args:
        attributes (list): list of he attributes to focus on. Defaults to [].
        action (str, optional): Can be ["quantity", "string", "time", ...].
        Defaults to 'repeat'.

    Returns:
        Dict: list of attributes for each bucket
    """

    if len(attributes) > 0:
        if action == 'repeat':
            selected_attributes = {
                bid: attributes
                for bid in progress.tqdm(dataset.data.keys(),
                                         desc="Apply attribute selection")
                }
            dataset.set_attributes(selected_attributes)
        if action == 'custom':
            selected_attributes = {
                bid: [attr for attr in attributes
                      if attr in dataset.data[bid].columns]
                for bid in progress.tqdm(dataset.data.keys(),
                                         desc="Apply attribute selection")
                }
            dataset.set_attributes(selected_attributes)
        if action == 'type_selection':
            selected_attributes = {
                bid: [attr for attr in dataset.data[bid].columns
                      if attr in dataset.attribute_types
                      and dataset.attribute_types[attr] in attributes]
                for bid in progress.tqdm(dataset.data.keys(),
                                         desc="Apply attribute selection")
                }
            dataset.set_attributes(selected_attributes)
    else:
        selected_attributes = {
            bid: [attr for attr in dataset.data[bid].columns
                  if dataset.attribute_types[attr] == action]
            for bid in progress.tqdm(dataset.data.keys(),
                                     desc="Apply attribute selection")
            }
        dataset.set_attributes(selected_attributes)


def data_preprocessing(dataset, preprocessing_function,
                       attributes=None, action="default",
                       modify_structure=False, progress=tqdm):
    print(f"Apply {preprocessing_function.__name__}")
    if action == "default":
        for bid in progress.tqdm(dataset.data.keys(),
                                 desc=preprocessing_function.__name__):
            for attr in attributes:
                dataset.data[bid][attr] = dataset.data[bid][attr] \
                    .apply(preprocessing_function)
                if modify_structure:
                    dataset.data[bid] = dataset.data[bid].explode(attr) \
                        .reset_index(drop=True)
                dataset.gt_data[bid][attr] = dataset.gt_data[bid][attr] \
                    .apply(preprocessing_function)
    if action == "type_selection":
        for bid in dataset.data.keys():
            for attr in dataset.attributes[bid]:
                if attr in dataset.attribute_types and \
                    dataset.attribute_types[attr] in attributes:
                    dataset.data[bid][attr] = dataset.data[bid][attr] \
                        .apply(preprocessing_function)
                    if modify_structure:
                        dataset.data[bid] = dataset.data[bid].explode(attr) \
                            .reset_index(drop=True)
                    dataset.gt_data[bid][attr] = dataset.gt_data[bid][attr] \
                        .apply(preprocessing_function)


def extract_time(date):
    """Extract hours:minutes for Flights dataset

    Args:
        date (string): date as string in any format

    Returns:
        (int, int): returns extracted hours and minutes otherwise None
    """
    if not pd.isna(date):
        date = re.sub(r'\(.*?\)', '', date)
        match = re.search(r'(\d{1,2}):(\d{2})', date)

        if match:
            hour, minute = match.groups()
            return int(hour), int(minute)

    return None


def extract_number(text):
    """Extract the first number wrapped in a text

    Args:
        text (str): text containing a number

    Returns:
        float: if a number is in the text, returns the number otherwise None
    """
    text = str(text)
    match = re.search(r'[-+]?\d*\.\d+|\d+', text)
    if match:
        return float(match.group())
    return None


def get_minutes(time):
    """Count the number of minutes in time

    Args:
        time ((int, int)): time as a pair (#hours, #minutes)

    Returns:
        int: returns the number of minutes
    """
    if time is not None:
        hours, minutes = time
        return hours * 60 + minutes
    return None


def transform_date(date):
    """Transform a time into minutes

    Args:
        date (string): date to be changed

    Returns:
        int: number of minutes
    """
    date = str(date)
    return get_minutes(extract_time(date))


def extract_number_with_commas(text):
    """Extract the first number wrapped in text

    Args:
        text (str): text containing a number

    Returns:
        float: if a number is in the text, returns the number otherwise None
    """
    text = str(text)
    match = re.search(r'[-+]?\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+', text)
    if match:
        extracted_number = match.group()
        extracted_number = extracted_number.replace(',', '')
        return float(extracted_number)
    return None


def remove_space(text: str) -> str:
    try:
        new_text = text.strip()
        return new_text
    except Exception:
        return text


def scale_units(dataset,
                attributes: List[str],
                progress=tqdm,
                lower_factor=-0.8,
                upper_factor=0.8,
                max_itr=10,
                max_workers=21) -> None:
    """Scale the values of the attributes to the same units

    Args:
        df_id (str): bucket involved by the unit scaling
        attributes (list(str)): attributes to scale
        lower_factor (float, optional): lower threshold after log10. Defaults to -0.8.
        upper_factor (float, optional): upper threshold after log10. Defaults to 0.8.
        max_itr (int, optional): number of iterations to scale/update values. Defaults to 50.
    """
    for bid in progress.tqdm(dataset.data.keys(), desc="Scale units"):
        entity_names = set(dataset.data[bid][dataset.entity_col_name])
        for entity in entity_names:
            attr_in_df = set(dataset.attributes[bid]) & set(attributes)
            for attr in attr_in_df:
                # Retrieve the values of the attribute (attr) for the entity
                values = list(dataset.data[bid][dataset.data[bid][dataset.entity_col_name] == entity][attr])
                extracted_numbers = np.array([extract_number_with_commas(val) for val in values])
                # NaN values are excluded from scaling
                ind_to_scaled = (extracted_numbers != None) & (extracted_numbers != 0) #extracted_numbers != None
                numbers_to_scale = np.copy(extracted_numbers[ind_to_scaled])
                for _ in range(max_itr):
                    main_bin_center = np.median(numbers_to_scale)
                    # Compute the power factor 10 between the values and the mean value
                    try:
                        normalized_values = numbers_to_scale / main_bin_center
                        normalized_values = normalized_values.astype(float)
                        power_factors = np.log10(normalized_values)
                        values_below = power_factors < lower_factor
                        values_above = power_factors > upper_factor
                        # We adjust the values according to their power difference
                        if np.any(values_below) or np.any(values_above):
                            numbers_to_scale[values_below] *= 10
                            numbers_to_scale[values_above] *= 0.1
                        else:
                            break
                    except Exception as e:
                        print(f'An error has ocurred : {e}.')
                # We update the values set on the same scale
                extracted_numbers[ind_to_scaled] = np.copy(numbers_to_scale)
                dataset.data[bid].loc[dataset.data[bid][dataset.entity_col_name] == entity, attr] = extracted_numbers

                # Apply the same transformation to the GT data
                dataset.gt_data[bid][attr] = dataset.gt_data[bid][attr].apply(extract_number_with_commas)


def extract_authors(authors):
    """Try to extract each author in the data"""
    if pd.isna(authors):
        return None
    if not isinstance(authors, str):
        return None
    if "; " in authors:
        names = authors.split("; ")
        return [name.lower().strip() for name in names]
    if authors.count(",") > 1 or \
        (authors.count(",") == 1 and len(authors.split()) > 3):
        names = authors.split(", ")
        formatted_names = []
        for name in names:
            parts = name.split()
            if len(parts) > 1:
                last_name = parts[-1]
                first_names = ' '.join(parts[:-1])
                formatted_names.append(
                    f"{last_name.lower()}, {first_names.lower()}"
                    )
            else:
                formatted_names.append(name.lower())
        return formatted_names
    return [authors.lower().strip()]


def split_authors(authors):
    """Identifies each author in the string"""
    return [name.strip() for name in authors.split(";") if name.strip()]


DATA_PREPROCESSING_FUNCTIONS = {
    "remove_space": remove_space,
    "extract_number": extract_number,
    "extract_number_with_commas": extract_number_with_commas,
    "extract_time": extract_time,
    "get_minutes": get_minutes,
    "transform_date": transform_date,
    "extract_authors": extract_authors,
}


METADATA_PREPROCESSING_FUNCTIONS = {
    "focus_on_attributes": focus_on_attributes,
    "scale_units": scale_units,
}
