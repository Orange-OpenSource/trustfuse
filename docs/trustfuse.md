# Using TrustFuse

## Run a fusion model

#### ``fusion_pipeline.py``

Runs a basic data fusion pipeline.

```
python fusion_pipeline.py --dataset-path DATASET_PATH --attr-types ATTR_TYPES --model CRH --dataset-name DATASET_NAME --preprocess-config .\data\configurations\crh\book\preprocess_configuration.json --dynamic
```

with
* ``DATASET_PATH``: (e.g., data/input_trustfuse/&lt;dataset_name&gt;)
* ``ATTR_TYPES``: file that maps attributes to a data type (e.g., data/configurations/&lt;model_name&gt;/&lt;dataset_name&gt;/types.json)
* ``MODEL``: name of the fusion model
* ``DATASET_NAME``: name of the dataset
* ``PREPROCESS_CONFIG``: preprocessing file that contains functions to be applied on the dataset before data fusion (e.g., data/configurations/&lt;model_name&gt;/&lt;dataset_name&gt;/preprocess_configuration.json)
* ``DYNAMIC (optional)``: add ``--dynamic`` to the command only if you use WikiConflict dataset

## Models 

### Models implemented in TrustFuse (more to come)

* [CRH](https://dl.acm.org/doi/pdf/10.1145/2588555.2610509)
* [ACCU](https://dl.acm.org/doi/pdf/10.14778/1687627.1687690)
* [CATD](https://dl.acm.org/doi/pdf/10.14778/2735496.2735505)
* [GTM](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=2ffe1157df90ce94cb91f28074b43b58135cedac)
* [KDEm](https://dl.acm.org/doi/pdf/10.1145/2939672.2939837)
* [LTM](https://dl.acm.org/doi/pdf/10.14778/2168651.2168656)
* [SLiMFast](https://dl.acm.org/doi/pdf/10.1145/3035918.3035951)
* [TruthFinder](https://dl.acm.org/doi/pdf/10.1145/1281192.1281309)

### Template to implement your own fusion model
Each of the above models inherits from a ``Model`` class in ``trustfuse/models/model.py`` which prepares the input data of the fusion model in the following format: 

```python
[   # for (entity_1, attribute_1) pair
    [[index_source_1, value],
    [index_source_2, value],
    ...,
    [index_source_N, value]],

    # for (entity_1, attribute_2) pair
    [[index_source_1, value],
    [index_source_3, value]],
    ...
    # for (entity_1, attribute_N) pair
    [[...], ...],
    ...,
    # for (entity_N, attribute_N) pair
    [[...], ...]
]
```

The dataset is transformed into a matrix to simplify the operations of some fusion models, and each data source and (entity, attribute) pair is assigned to an index that will allow the fusion output to be reformatted to the input format.

Then, each model must implement a ``_fuse`` method that performs the fusion and updates a ``model_output`` attribute of the ``Model`` class and contains the most reliable facts and confidence scores in the data sources for each bucket (BID). If the fusion model does not respect the output format expected by the ``get_results`` method of the ``Model`` class, you need to override your model's ``get_results`` method and transform your model's output into that format:


```python
unified_result = {
    BID: {
        "truth": {
            "entity_name": {
                "attribute_name": ["val_1", ..., "val_N"],
                ...,
                "attribute_name": ["val_1", ..., "val_N"]
            },
            ...,
            "entity_name": {
                "attribute_name": ["val_1", ..., "val_N"],
                ...,
                "attribute_name": ["val_1", ..., "val_N"]
            }
        },
        "weights": {
            "source_name": 0.9,
            ...,
            "source_name": 0.8
        }
    },
    ...
}
```

```python
def fuse(self, dataset, bid, inputs, progress=tqdm):
    """Perform the fusion

    Args:
        dataset (Dataset): Dataset instance

    Returns:
        Dict: Results with confidence scores
    """
    self._fuse(dataset, bid, inputs, progress)
    return self.get_results(dataset)
```

These are the only requirements for implementing your own fusion model and using the other features of TrustFuse with your own model.

### Data

- [Existing fusion datasets also integrated into my repository](http://lunadong.com/fusionDataSets.htm)


#### <i class="fab fa-github"></i> Links to GitHub repositories that inspired us and from which some models have been reused or modified:

- [TDH](https://github.com/woohwanjung/truthdiscovery)

- [LTM](https://github.com/yishangru/TruthDiscovery/tree/master)

- [Multiple models](https://github.com/MengtingWan/KDEm)

- [SLiMFast, ACCU](https://github.com/HoloClean/RecordFusion/)