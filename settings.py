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

from trustfuse.models import CRH
from trustfuse.models import KDEm
from trustfuse.models import CATD
from trustfuse.models import GTM
from trustfuse.models import TruthFinder
from trustfuse.models import LTM
from trustfuse.models import SLiMFast
from trustfuse.models import Accu


# The gold standard contains departure/arrival information on
# 100 randomly selected flights provided by corresponding airline websites
# https://lunadong.com/fusionDataSets.htm
STOCK_HEADER = ['Source',
                'Symbol',
                'Change %',
                'Last trading price',
                'Open price',
                'Change $',
                'Volume',
                "Today's high",
                "Today's low",
                'Previous close',
                '52wk High',
                '52wk Low',
                'Shares Outstanding',
                'P/E',
                'Market cap',
                'Yield',
                'Dividend',
                'EPS']

STOCK_HEADER_GT = STOCK_HEADER.copy()
STOCK_HEADER_GT.remove('Source')

FLIGHT_HEADER = ['Source',
                 'Flights#',
                 'Scheduled departure',
                 'Actual departure',
                 'Departure gate',
                 'Scheduled arrival',
                 'Actual arrival',
                 'Arrival gate']

FLIGHT_HEADER_GT = FLIGHT_HEADER.copy()
FLIGHT_HEADER_GT.remove('Source')

BOOK_HEADER = ["Source",
               "ISBN",
               "Title",
               "Author list"]
BOOK_HEADER_GT = ["ISBN",
                  "Author list"]


# *** Fusion Model Mapping ***
MODEL_MAP = {
    'CRH': CRH,
    'KDEm': KDEm,
    'CATD': CATD,
    'GTM': GTM,
    'TruthFinder': TruthFinder,
    'SLIMFAST': SLiMFast,
    'LTM': LTM,
    'ACCU': Accu,
}

# *** Fusion Model Parameters ***
MODEL_PARAMETERS = {
    'CRH': {
        "max_itr": 10,
    },
    'CATD': {
        "numeric": True,
    },
    'KDEm': {
        "numeric": True,
    },
    'GTM': {
        "numeric": True,
    },
    'TruthFinder': {
        "max_itr": 10,
    },
    'LTM': {
        "max_itr": 10,
    },
    'KBT': {  
    },
    'SLIMFAST': {
        "max_itr": 10,
    },
    'ACCU': {
    },
}

DATASET_PARAMETERS = {
    'WikiConflict': {
        'entity_col_name': 'Entity'
    },
    'Flight': {
        'entity_col_name': 'Flights#',
        'headers': (FLIGHT_HEADER, FLIGHT_HEADER_GT),
    },
    'Stock': {
        'entity_col_name': 'Symbol',
        'headers': (STOCK_HEADER, STOCK_HEADER_GT),
    },
    "Book": {
        "entity_col_name": "ISBN",
        "headers": (BOOK_HEADER, BOOK_HEADER_GT),
    }
}
