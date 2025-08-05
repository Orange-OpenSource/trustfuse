# :hammer: Dataset Construction

## Available scripts

All scripts to reproduce our dataset WikiConflict (with statistics and the labeling method) are in this [folder](./dataset_generation/).

## A Step by Step Approach

### First step: retrieve Wikidata entities (QIDs)

#### ``wikipedia_category_collector.py``

This script collects Wikidata entities linked to a Wikipedia category. To do this, it recursively searches for pages and subpages that have an associated Wikidata entity and returns a CSV file with the QIDs linked by Wikipedia category.

```
python wikipedia_category_collector.py --categories-list CATEGORIES_LIST --language LANGUAGE
```

with
* ``CATEGORIES_LIST``: File containing one Wikipedia category per line
* ``LANGUAGE``: Language of the Wikipedia category

#### ``generate_conflicting_dataset.py``

Conflicting dataset generator : this script retrieves all the revisions of a list of entities and generates a statistics file that can be used by another script.

```
python generate_conflicting_dataset.py --entities ENTITIES --wikidata-hashmap WIKIDATA_HASHMAP --stats STATS --prop-types PROP_TYPES
```

with
* ``ENTITIES``: JSON file containing the list of entities of interest
* ``WIKIDATA_HASHMAP``: Local hashmap to get labels efficiently (optional)
* ``STATS``: stats file to get insights from the dataset
* ``PROP_TYPES``: To record the type of each property seen during the retrieving

#### ``generate_buckets.py``

Generate buckets after the Wikidata revision history collection step.

```
python generate_buckets.py --dataset DATASET --fusion-data FUSION_DATA --revisions-folder REVISIONS_FOLDER --property-labels PROPERTY_LABELS --constrained-properties CONSTRAINED_PROPERTIES --media-properties MEDIA_PROPERTIES
```

with
* ``DATASET``: Dataset containing conflicting data
* ``FUSION_DATA``: Folder path that will contain data for fusion models
* ``REVISIONS_FOLDER``: Folder path where are stored the revisions collected with ``generate_conflicting_data.py``
* ``PROPERTY_LABELS``: Pickle file containing property PID/label mapping
* ``CONSTRAINED_PROPERTIES``: Folder that contains three CSV files for three diffrent constraints in Wikidata and its associated properties
* ``MEDIA_PROPERTIES``: CSV file containing media properties such as (MP4, PNG, and others)


#### ``label_dataset.py``

This script allows you to quickly label the dataset by entering the correct values and partial orders via a console.

```
python label_dataset.py --buckets BUCKETS --latest-values LATEST_VALUES --fusion-data FUSION_DATA --constrained-properties CONSTRAINED_PROPERTIES --attribute-types ATTRIBUTE_TYPES --filters FILTERS
```

with
* ``BUCKETS``: Pickle File that contains the constructed buckets at the previous step (output of ``generate_buckets.py``)
* ``LATEST_VALUES``: Pickle file containing the latest values of each properties and entities involved in the dataset
* ``FUSION_DATA``: Folder path that will contain data for fusion models
* ``CONSTRAINED_PROPERTIES``: Pickle file containing the constrained properties of Wikidata to support the labeling
* ``ATTRIBUTE_TYPES``: Pickle file containing the mapping property/type
* ``FILTERS``: Pickle file containing the properties to filter out of the labeling or future properties to filter


#### ``retrieve_latest_values.py``

This script retrieves the latest values from Wikidata for each of the entities/properties present in the dataset to help with labeling.

```
python retrieve_latest_values.py --dataset DATASET --fusion-data FUSION_DATA --source-names SOURCE_NAMES --source-correlations SOURCE_CORRELATIONS --revisions-folder 
REVISIONS_FOLDER --property-labels PROPERTY_LABELS --constrained-properties CONSTRAINED_PROPERTIES
```

with
* ``DATASET``: Dataset containing conflicting data
* ``FUSION_DATA``: Folder path that will contain data for fusion models
* ``REVISIONS_FOLDER``: Folder path where are stored the revisions collected with ``generate_conflicting_data.py``
* ``PROPERTY_LABELS``: Pickle file containing property PID/label mapping
* ``CONSTRAINED_PROPERTIES``: Folder that contains three CSV files for three diffrent constraints in Wikidata and its associated properties