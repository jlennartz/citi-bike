#!/usr/bin/env bash

DATA_DIR="/home/lennartz/data/citibike"
URL="https://s3.amazonaws.com/tripdata/2023-citibike-tripdata.zip"

# download
mkdir -p "$DATA_DIR"
wget -q --show-progress -O "$DATA_DIR/2023-citibike-tripdata.zip" "$URL"
# unzip nested zips
unzip -o "$DATA_DIR/2023-citibike-tripdata.zip" -d "$DATA_DIR"
cd "$DATA_DIR/2023-citibike-tripdata"
for zipfile in ./*.zip; do
  unzip -o "$zipfile" -d .
done
# clean up
rm -f ./*.zip
rm -f "$DATA_DIR/2023-citibike-tripdata.zip"
# show file structure
cd "$DATA_DIR"
tree