#!/usr/bin/env bash

conda create -n citibike \
    numpy==2.2.6 \
    pandas \
    psycopg2-binary \
    python-dotenv \
    tqdm ipykernel \
    matplotlib \
    geopandas \
    geodatasets \
    contextily \
    skimage \
    sqlalchemy