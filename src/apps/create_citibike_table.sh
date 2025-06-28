#!/usr/bin/env bash
set -euo pipefail


DB_NAME="citibike2023"
DB_USER="lennartz"
PSQL="psql"
DATA_DIR="/home/lennartz/data/citibike/2023-citibike-tripdata"


if ! sudo -u postgres $PSQL -tAc \
     "SELECT 1 FROM pg_database WHERE datname = '$DB_NAME'" \
     | grep -q 1
then
  echo "Creating database '$DB_NAME'…"
  sudo -u postgres $PSQL -c "CREATE DATABASE $DB_NAME OWNER $DB_USER;"
else
  echo "Database '$DB_NAME' already exists."
fi

$PSQL -U "$DB_USER" -d "$DB_NAME" -c "DROP TABLE IF EXISTS citibike_trips;"

echo "Ensuring table 'citibike_trips' exists in '$DB_NAME'…"
$PSQL -U "$DB_USER" -d "$DB_NAME" <<'EOSQL'
CREATE TABLE IF NOT EXISTS citibike_trips (
  ride_id            TEXT PRIMARY KEY,
  rideable_type      TEXT,
  started_at         TIMESTAMP NOT NULL,
  ended_at           TIMESTAMP NOT NULL,
  start_station_name TEXT,
  start_station_id   TEXT,
  end_station_name   TEXT,
  end_station_id     TEXT,
  start_latitude     DOUBLE PRECISION,
  start_longitude    DOUBLE PRECISION,
  end_latitude       DOUBLE PRECISION,
  end_longitude      DOUBLE PRECISION,
  member_casual      TEXT
);
EOSQL

echo "Table 'citibike_trips' ensured in database '$DB_NAME'."

for csv in "$DATA_DIR"/*.csv; do
  echo "  Importing $(basename "$csv")..."
  $PSQL -U "$DB_USER" -d "$DB_NAME" \
    -c "\copy citibike_trips FROM '$csv' WITH (FORMAT csv, HEADER true);"
done