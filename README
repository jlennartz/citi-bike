# CitiBike Crash Risk Analysis

A geospatial analysis tool that processes NYC CitiBike trip data and crash records to identify high-risk cycling areas. The project rasterizes movement patterns and collision data into 3D grids (spatial + temporal) to help insurers and city planners understand cycling safety patterns.

## Key Features

- Rasterizes CitiBike trips and crash data into customizable spatial/temporal grids
- Calculates crash risk normalized by cycling volume 
- Generates spatial heatmaps with NYC borough overlays
- Analyzes temporal patterns (day-of-week, time-of-day, seasonal trends)
- Produces visualizations and statistical reports

## Setup

### Prerequisites

- Python 3.8+
- PostgreSQL
- Conda (recommended)

### Environment Setup

1. Create and activate conda environment:
```bash
cd src/apps
bash cond-env.sh
conda activate citibike
```

2. Download CitiBike data:
Adapt paths and run
```bash
bash download_data.sh
```

3. Setup PostgreSQL database:
Configure PostgreSQL and run
```bash
bash create_citibike_table.sh
```

### Usage

1. **Rasterize data** (creates 3D grids of both datasets and projects back tp map):
```bash
cd src/apps
python rasterize_data.py --cell_x 800 --cell_y 800 --cell_t 6
```

2. **Generate visualizations**:
```bash
cd src/jupyter
jupyter notebook vis_results.ipynb
```

## Output

- Spatial heatmaps showing crash risk hotspots
- Temporal analysis of risk patterns
- Monthly trend analysis
- Statistical filtering for reliable estimates

Results are saved to `results/` directory with plots and processed data files.