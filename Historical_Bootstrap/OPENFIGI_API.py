#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 09:41:06 2026

@author: october
"""
from openfigi import mapping

# Your list of Bloomberg IDs (BBGID)
bloomberg_ids = [
    "BBG000BP52R2", "BBG000BC1L02", "BBG000B9ZXB4", # ... include all 50 IDs
]

# Create a request for each ID
requests = [{'idType': 'ID_BB_GLOBAL', 'idValue': id} for id in bloomberg_ids]

# Send the batch request
responses = mapping(requests)

# Process the results
for response in responses:
    if response.data:
        # The first result usually contains the ISIN
        isin = response.data[0].get('isin')
        print(f"BBGID {response.job.idValue} maps to ISIN: {isin}")
    else:
        print(f"BBGID {response.job.idValue} not found: {response.error}")