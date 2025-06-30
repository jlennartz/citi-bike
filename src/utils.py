import numpy as np
import pandas as pd
import geopandas as gpd
from pyproj import Transformer
from shapely.geometry import box
from skimage.draw import line_nd
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import psycopg2
import math
from sqlalchemy import create_engine


def setup_grid_parameters(lon_min, lat_min, lon_max, lat_max, cell_x, cell_y, cell_t, time_max):
    """
    Setup grid parameters and calculate grid dimensions.
    """
    # Create and reproject bounding box
    bbox = gpd.GeoSeries(
        [box(lon_min, lat_min, lon_max, lat_max)],
        crs="EPSG:4326"
    ).to_crs(epsg=3857)

    minx, miny, maxx, maxy = bbox.total_bounds 

    Nx = math.ceil((maxx - minx) / cell_x)  # number of cells in x-direction
    Ny = math.ceil((maxy - miny) / cell_y)  # number of cells in y-direction
    Nt = math.ceil(time_max / cell_t)       # number of time steps

    transformer = Transformer.from_crs(
        "EPSG:4326", "EPSG:3857", always_xy=True
    )

    return minx, miny, maxx, maxy, Nx, Ny, Nt, transformer


def to_grid_indices(
    lon, lat, t,
    minx, miny,
    cell_x, cell_y, cell_t,
    transformer    
):
    """
    Map (lon, lat, t) to integer grid indices (ix, iy, it).
    Can be negative or exceed (Nx-1, Ny-1, Nt-1).
    """
    # 1. Project lon/lat → Web Mercator (metres)
    x, y = transformer.transform(lon, lat)              

    # 2. Offset and floor-divide
    ix = math.floor((x - minx) / cell_x)
    iy = math.floor((y - miny) / cell_y)
    it = math.floor((t - 0.0)   / cell_t)

    return ix, iy, it


def map_to_grid_indices(args):
    """Helper for parallel grid mapping."""
    lon, lat, t, minx, miny, cell_x, cell_y, cell_t, transformer = args
    return to_grid_indices(lon, lat, t, minx, miny, cell_x, cell_y, cell_t, transformer)


def parallel_map_grid(df, cols, new_cols, minx, miny, cell_x, cell_y, cell_t, transformer, chunksize=None):
    """
    Parallel map of (lon, lat, t) -> (ix, iy, it)
    cols: list of source columns [lon, lat, t]
    new_cols: list of target columns [ix, iy, it]
    """
    n_workers = cpu_count()
    # Prepare arguments
    args = list(zip(
        df[cols[0]], df[cols[1]], df[cols[2]], 
        [minx] * len(df), [miny] * len(df),
        [cell_x] * len(df), [cell_y] * len(df), [cell_t] * len(df),
        [transformer] * len(df)
    ))
    if chunksize is None:
        chunksize = max(1, len(args) // (n_workers * 4))
    
    with Pool(n_workers) as pool:
        results = pool.map(map_to_grid_indices, args, chunksize=chunksize)
    
    # Unzip results and assign back to DataFrame
    ix, iy, it = zip(*results)
    df[new_cols[0]] = ix
    df[new_cols[1]] = iy
    df[new_cols[2]] = it
    return df


def trip_rasterization(
    df_chunk,
    Nt, Ny, Nx,
):
    """
    Given a DataFrame chunk, map each ray into a local 3D count array.
    """
    local = np.zeros((Nt, Ny, Nx), dtype=np.uint32)
    for _, row in df_chunk.iterrows():
        start = (int(row['startt']), int(row['starty']), int(row['startx']))
        stop = (int(row['endt']),  int(row['endy']),  int(row['endx']))
        zz, yy, xx = line_nd(start, stop)
        local[zz, yy, xx] += 1
    return local


def parallel_trip_rasterization(df, Nt, Ny, Nx):
    """
    Parallel processing of DataFrame in chunks for ray rasterization.
    """
    num_workers = cpu_count()
    chunks = np.array_split(df, num_workers)

    # Prepare argument tuples for each chunk
    args = [(chunk, Nt, Ny, Nx) for chunk in chunks]

    with Pool(num_workers) as pool:
        partial_counts = pool.starmap(trip_rasterization, args)

    # Combine partial results
    global_counts = np.sum(partial_counts, axis=0).astype(np.uint32)
    return global_counts


def filter_inside(indices, shape):
    """
    Filter a list of (ix,iy,it) to only those within
    0 ≤ ix < shape[2], 0 ≤ iy < shape[1], 0 ≤ it < shape[0].
    """
    Nt, Ny, Nx = shape
    inside = []
    for ix, iy, it in indices:
        if (0 <= ix < Nx and
            0 <= iy < Ny and
            0 <= it < Nt):
            inside.append((ix, iy, it))
    return inside


def process_citibike_data(batch_size, ref, minx, miny, cell_x, cell_y, cell_t, transformer, Nx, Ny, Nt):
    """
    Process CitiBike data in batches and return rasterized counts.
    """
    # Connect to database
    conn = psycopg2.connect(dbname="citibike2023", user="lennartz", host="localhost")

    # Get total row count
    with conn.cursor() as cnt_cur:
        cnt_cur.execute("""
          SELECT COUNT(*) FROM public.citibike_trips
          WHERE start_latitude  IS NOT NULL
            AND start_longitude IS NOT NULL
            AND end_latitude    IS NOT NULL
            AND end_longitude   IS NOT NULL;
        """)
        total_rows = cnt_cur.fetchone()[0]

    total_batches = math.ceil(total_rows / batch_size)

    cur = conn.cursor(name="batch_cursor")  # server-side cursor
    sample_sql = f"""
      SELECT start_latitude, start_longitude, started_at,
             end_latitude,   end_longitude,   ended_at
      FROM public.citibike_trips
      WHERE start_latitude  IS NOT NULL
        AND start_longitude IS NOT NULL
        AND end_latitude    IS NOT NULL
        AND end_longitude   IS NOT NULL;
    """

    counts_list = []

    # Execute and fetch in batches with progress bar
    cur.execute(sample_sql)
    with tqdm(total=total_batches, desc="Processing CitiBike batches") as pbar:
        while True:
            batch = cur.fetchmany(batch_size)
            if not batch:
                break
            df = pd.DataFrame(batch, columns=[
                "start_latitude", "start_longitude", "started_at",
                "end_latitude", "end_longitude", "ended_at"
            ])
            # Compute hours since the reference
            df["started_at_hours"] = (df["started_at"] - ref) / pd.Timedelta(hours=1)
            df["ended_at_hours"] = (df["ended_at"] - ref) / pd.Timedelta(hours=1)

            # Time grid mapping in parallel
            df = parallel_map_grid(
                df,
                cols=["start_longitude", "start_latitude", "started_at_hours"],
                new_cols=["startx", "starty", "startt"],
                minx=minx, miny=miny, cell_x=cell_x, cell_y=cell_y, cell_t=cell_t,
                transformer=transformer
            )
            df = parallel_map_grid(
                df,
                cols=["end_longitude", "end_latitude", "ended_at_hours"],
                new_cols=["endx", "endy", "endt"],
                minx=minx, miny=miny, cell_x=cell_x, cell_y=cell_y, cell_t=cell_t,
                transformer=transformer
            )

            # Filter data that's out of grid and drop unnecessary columns
            cols_to_remove = [
                "start_latitude", "start_longitude", "started_at",
                "end_latitude",   "end_longitude",   "ended_at",
                "started_at_hours", "ended_at_hours"
            ]

            df.drop(columns=cols_to_remove, inplace=True)
            
            mask_start = (
                (df["startx"]  >= 0) & (df["startx"]  < Nx) &
                (df["starty"]  >= 0) & (df["starty"]  < Ny) &
                (df["startt"]  >= 0) & (df["startt"]  < Nt)
            )
            mask_end   = (
                (df["endx"]    >= 0) & (df["endx"]    < Nx) &
                (df["endy"]    >= 0) & (df["endy"]    < Ny) &
                (df["endt"]    >= 0) & (df["endt"]    < Nt)
            )

            valid_mask = mask_start & mask_end
            df   = df[valid_mask].reset_index(drop=True)

            global_counts = parallel_trip_rasterization(
                df,
                Nt=Nt, Ny=Ny, Nx=Nx
            )
            counts_list.append(global_counts)
            pbar.update(1)

    cur.close()
    conn.close()

    return np.stack(counts_list, axis=0).sum(0)


def process_crash_data(crash_data_path, ref, minx, miny, cell_x, cell_y, cell_t, transformer, Nx, Ny, Nt):
    """
    Process crash data and return rasterized counts.
    """
    # Load the data
    df = pd.read_csv(crash_data_path, parse_dates=['CRASH DATE'])
    dt = pd.to_datetime(
        df['CRASH DATE'].astype(str) + ' ' + df['CRASH TIME'],
        format='%Y-%m-%d %H:%M'
    )
    df["stime_at_hours"] = np.round((dt - ref) / pd.Timedelta(hours=1)).astype(int)

    cols_to_drop = [
        'CRASH DATE',
        'CRASH TIME',	
        'BOROUGH',
        'ZIP CODE',
        'NUMBER OF CYCLIST INJURED',
        'NUMBER OF CYCLIST KILLED',
        'COLLISION_ID'
    ]

    df.drop(columns=cols_to_drop, inplace=True)
    df.dropna(how='any', inplace=True)

    df = parallel_map_grid(
        df,
        cols=["LONGITUDE", "LATITUDE", "stime_at_hours"],
        new_cols=["x", "y", "t"],
        minx=minx, miny=miny, cell_x=cell_x, cell_y=cell_y, cell_t=cell_t,
        transformer=transformer
    )

    mask   = (
        (df["x"]    >= 0) & (df["x"]    < Nx) &
        (df["y"]    >= 0) & (df["y"]    < Ny) &
        (df["t"]    >= 0) & (df["t"]    < Nt)
    )

    df   = df[mask].reset_index(drop=True)
    t_idx = df['t'].to_numpy().astype(int)
    y_idx = df['y'].to_numpy().astype(int)
    x_idx = df['x'].to_numpy().astype(int)

    crashes = np.zeros((Nt, Ny, Nx), dtype=np.uint32)
    np.add.at(crashes, (t_idx, y_idx, x_idx), 1)

    return crashes