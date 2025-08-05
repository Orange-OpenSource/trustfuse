# [TrustFuse :volcano: & WikiConflict ] Experimenting with Knowledge Fusion for Knowledge Graphs: A Versatile Playground

#### :busts_in_silhouette: Lucas Jarnac, Yoan Chabot, and Miguel Couceiro

## :clipboard: Description

This repository proposes a playground to experiment with knowledge fusion models in a knowledge graph (KG) construction context. We also propose a dataset built from the full revision history of [Wikidata](https://www.wikidata.org/wiki/Wikidata:Main_Page) since its creation in 2012 for a Wikipedia category, which we have called [WikiConflict](https://example.com/).
This dataset aims to include different scenarios that correspond to real-life scenarios and challenges faced by KG construction:
* **long-tail phenomenon**: the dataset contains limited data for a single object, which complicates the fusion task
* **knowledge granularity**: has different levels of specificity (hierarchical or in the information)
* **heterogeneous data types**: has different data types (numeric, string, coordinate, entity, etc.)

## Documentation

- [Installation](docs/installation.md)
- [Dataset construction](docs/dataset_construction.md)
- [TrustFuse](docs/trustfuse.md)
- [TrustFuse UI](docs/trustfuse_ui.md)
