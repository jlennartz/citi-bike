#!/usr/bin/env bash

conda create -n citibike \
    pandas \
    psycopg2-binary \
    python-dotenv \
    tqdm ipykernel \
    matplotlib \
    geopandas \
    geodatasets \
    contextily