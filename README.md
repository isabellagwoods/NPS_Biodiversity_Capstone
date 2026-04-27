# NPS Biodiversity Capstone
### Predicting Human Impact on Biodiversity in US National Parks
**Northwestern University STAT 390 — Capstone Project**

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Research Question](#research-question)
3. [Repository Structure](#repository-structure)
4. [Data Sources](#data-sources)
5. [Data Pipeline](#data-pipeline)
6. [Feature Engineering](#feature-engineering)
7. [Model](#model)
8. [AutoResearch Loop](#autoresearch-loop)
9. [Setup & Installation](#setup--installation)
10. [Running the Project](#running-the-project)
11. [Results](#results)
12. [Known Issues & Limitations](#known-issues--limitations)

---

## Project Overview

This project investigates how human activity affects biodiversity in US National Parks. Using a combination of ecological, climate, geographical, and visitation data, we build a predictive model for the **Shannon Diversity Index (SDI)** — a standard measure of species diversity — and identify which human impact variables most strongly predict biodiversity outcomes.

The project uses an **AutoResearch** framework in which an AI agent autonomously iterates through model experiments, logging results and advancing only when performance improves.

**Success Criteria:**
| Metric | Target |
|--------|--------|
| `val_rmse` | ≤ 0.10 |
| `val_r2` | ≥ 0.85 |

---

## Research Question

> *To what extent does human activity — measured through visitation, traffic, light pollution, air quality, and proximity to industry — predict species biodiversity in US National Parks, and which factors are most influential?*

---

## Repository Structure

```
CAPSTONE/
│
├── DATA/                          # Raw and processed data files
│   ├── raw/
│   │   ├── npspecies/             # NPSpecies bulk download outputs
│   │   ├── inaturalist/           # iNaturalist observation CSVs by park range
│   │   └── nasa/                  # AppEEARS satellite data downloads
│   ├── airQualityData.csv         # EPA air quality by county
│   ├── full_data_clean.csv        # Final model-ready dataset (one row per park)
│   └── nasa_task_ids.json         # AppEEARS task IDs for re-downloading
│
├── scrapingData.ipynb             # Main data collection and feature engineering notebook
│
├── train.py                       # Model file — only file edited during AutoResearch
├── prepare.py                     # Data preparation — DO NOT MODIFY
├── program.md                     # AutoResearch loop instructions
├── results.tsv                    # Experiment results log (not committed)
├── experiment_log.md              # Detailed experiment notes (not committed)
├── run.log                        # Output from most recent run (not committed)
│
└── README.md                      # This file
```

---

## Data Sources

| Dataset | Source | Description | Parks Covered |
|---------|--------|-------------|---------------|
| NPSpecies | NPS IRMA API | Species presence/absence records per park | ~170 parks |
| iNaturalist | iNaturalist API | Timestamped citizen science observations | ~170 parks, 4M+ records |
| NPS Visitation | NPS Stats | Annual recreation visitors, campers, backcountry use | All parks, 1934–2025 |
| NPS Traffic | NPS Stats | Monthly vehicle traffic counts by entrance | All parks, 2012–2020 |
| NPS Parks API | NPS API | Coordinates, activities, entrance fees, operating hours | All parks |
| NPS Parking API | NPS API | Parking lot accessibility and capacity | All parks |
| MODIS Land Cover | NASA AppEEARS (MCD12Q1.061) | Land cover type classification | All parks |
| MODIS NDVI | NASA AppEEARS (MOD13A1.061) | Vegetation greenness index | All parks |
| MODIS ET | NASA AppEEARS (MOD16A2.061) | Evapotranspiration | All parks |
| MODIS GPP | NASA AppEEARS (MOD17A2H.061) | Gross primary productivity | All parks |
| MODIS Snow | NASA AppEEARS (MOD10A1.061) | Snow cover (winter) | All parks |
| MODIS Fire | NASA AppEEARS (MCD64A1.061) | Burn date detection | All parks |
| MODIS LST | NASA AppEEARS (MOD11A1.061) | Land surface temperature | All parks |
| MODIS FPAR | NASA AppEEARS (MCD15A2H.061) | Fraction of photosynthetically active radiation | All parks |
| MODIS Tree Cover | NASA AppEEARS (MOD44B.061) | Percent tree cover | All parks |
| NASADEM Elevation | NASA AppEEARS (NASADEM_NC.001) | Digital elevation model | All parks |
| SMAP Soil Moisture | NASA AppEEARS (SPL3SMP_E.006) | Soil moisture | All parks |
| GRIDMET Water Balance | NASA AppEEARS | Actual evapotranspiration, precipitation deficit | All parks |
| NASA POWER Climate | NASA POWER API | Temperature, precipitation, humidity, PAR | All parks |
| Census Population | US Census API | County population and population density | All parks |
| EPA Air Quality | EPA AQS API + downloaded CSV | PM2.5, Ozone, SO2, CO, NO2, Pb by county | All parks |
| TRI Toxic Release | EPA Toxic Release Inventory | Industrial facility emissions within 50km of park | All parks |
| FCC County Lookup | FCC Area API | Maps park coordinates to county FIPS code | All parks |

---

## Data Pipeline

The full data collection pipeline is in `scrapingData.ipynb`. The pipeline runs in the following order:

### 1. Species Data (NPSpecies)
- Downloads species records for all parks from the NPS IRMA API
- Script: `download_npspecies.py`
- Output: `data/raw/npspecies/npspecies_all_parks.csv`

### 2. iNaturalist Observations
- Uses `id_above` sliding window pagination to bypass the 10,000 result cap
- Includes `fetch_with_retry()` with exponential backoff for 429/connection errors
- Collected in multiple runs due to rate limiting; merged from files:
  - `iNat_A-D.csv`, `iNat_D-G.csv`, `iNat_G-M.csv`, `iNat_M-P.csv`, `iNat_P-Y.csv`, `iNat_zion.csv`, `iNaturalist2.csv`
- Total: ~4 million observations across ~170 parks

### 3. NPS API Data
- Parks API: coordinates, activities, entrance fees, operating hours
- Parking API: lot accessibility, ADA spaces, capacity, parking cost

### 4. NASA Satellite Data (AppEEARS)
- Submitted as batch point-sample tasks via the AppEEARS API
- Task IDs saved to `nasa_task_ids.json` for re-downloading
- Date ranges: annual (2021) for slow-changing products, summer/winter snapshots for dynamic products

### 5. Climate Data (NASA POWER)
- Temperature, precipitation, humidity, PAR by park coordinates

### 6. Human Impact Data
- Census population density via US Census API
- EPA air quality via AQS API and downloaded county-level CSV
- TRI toxic release inventory with inverse-distance weighted pollution score

### 7. County Matching
- Park coordinates → county FIPS code via FCC Area API
- Used to join census and air quality data

---

## Feature Engineering

All features are aggregated to **one row per park**. The final dataset (`full_data_clean.csv`) contains the following feature groups:

### Biodiversity (Target)
| Column | Description |
|--------|-------------|
| `SDI` | Shannon Diversity Index calculated from iNaturalist observations: H = -Σ(p_i × log(p_i)) |
| `n_species` | Total unique species observed |
| `n_observations` | Total iNaturalist observations |

### iNaturalist Trends
| Column | Description |
|--------|-------------|
| `obs_slope` | Linear trend in annual observations over time |
| `species_slope` | Linear trend in annual unique species over time |

### Visitation (NPS Stats)
| Column | Description |
|--------|-------------|
| `recreation_visitors_<year>` | Recreation visitors in most recent year |
| `avg_recreation_visitors` | Mean annual recreation visitors across all years |
| `max_recreation_visitors` | Peak annual recreation visitors |
| `visit_slope` | Linear trend in annual visitors over time |
| `tent_campers_slope` | Linear trend in tent camping |
| `rv_campers_slope` | Linear trend in RV camping |
| `backcountry_slope` | Linear trend in backcountry use |
| `hours_per_visitor` | Average recreation hours per visitor |
| `overnight_ratio` | Ratio of overnight to total visitors |
| `nonrec_ratio` | Ratio of non-recreation to recreation visitors |
| `visit_covid_impact` | % change in visitors 2019→2020 |
| `backcountry_covid_impact` | % change in backcountry use 2019→2020 |

### Traffic (NPS Traffic Counts)
| Column | Description |
|--------|-------------|
| `traffic_slope` | Linear trend in annual vehicle traffic |
| `traffic_acceleration` | Change in traffic growth rate (last 3 years vs first 3 years) |
| `avg_annual_traffic` | Mean annual vehicle traffic |
| `max_annual_traffic` | Peak annual vehicle traffic |
| `recent_traffic` | Most recent year's traffic |
| `traffic_cv` | Coefficient of variation in traffic (year-to-year consistency) |
| `traffic_covid_impact` | % change in traffic 2019→2020 |
| `traffic_seasonality` | Peak month / min month ratio |

### Geography & Terrain
| Column | Description |
|--------|-------------|
| `latitude` | Park latitude |
| `longitude` | Park longitude |
| `elevation` | Park elevation in meters (Open Elevation API) |
| `NASADEM_HGT` | NASA DEM elevation |

### Climate (NASA POWER)
| Column | Description |
|--------|-------------|
| `avg_temp` | Mean annual temperature (°C) |
| `avg_temp_max` | Mean annual maximum temperature |
| `avg_temp_min` | Mean annual minimum temperature |
| `avg_temp_range` | Mean annual temperature range |
| `avg_precip` | Mean annual precipitation (mm/day) |
| `avg_humidity` | Mean annual relative humidity (%) |
| `avg_par` | Mean annual photosynthetically active radiation |

### Vegetation & Productivity (MODIS/NASA)
| Column | Description |
|--------|-------------|
| `avg_NDVI` / `max_NDVI` / `NDVI_range` | Vegetation greenness (16-day composite) |
| `avg_FPAR` / `max_FPAR` | Fraction of photosynthetically active radiation absorbed |
| `avg_ET` / `max_ET` / `ET_range` | Evapotranspiration (kg/m²/8day) |
| `avg_Gpp` / `max_Gpp` | Gross primary productivity (kg C/m²/8day) |
| `Percent_Tree_Cover` | % tree cover |
| `lc_type_code` | IGBP land cover type (1=forest … 17=water) |

### Snow & Fire
| Column | Description |
|--------|-------------|
| `avg_SNOW` / `max_SNOW` | NDSI snow cover index (winter) |
| `pct_burned` | Proportion of 2021 observations with detected fire |
| `n_burn_observations` | Number of months with detected burns |

### Soil & Water
| Column | Description |
|--------|-------------|
| `avg_soil_moisture` | Mean SMAP soil moisture |
| `soil_moisture_range` | Seasonal range of soil moisture |
| `avg_AET` | Actual evapotranspiration (GRIDMET water balance) |

### Human Impact & Pollution
| Column | Description |
|--------|-------------|
| `population` | County population (2020 Census) |
| `pop_density` | County population density (people/km²) |
| `CO_8hr` | CO air quality (8-hr average, ppm) |
| `O3_8hr` | Ozone (8-hr average, ppm) |
| `PM25_wtd` | PM2.5 weighted annual mean (µg/m³) |
| `SO2_1hr` | SO2 (1-hr average, ppb) |
| `NO2_AM` | NO2 annual mean (ppb) |
| `n_facilities` | Number of TRI industrial facilities within 50km |
| `weighted_pollution` | Inverse-distance weighted total toxic releases within 50km |
| `total_releases` | Total toxic releases within 50km (kg) |

### Park Characteristics (NPS API)
| Column | Description |
|--------|-------------|
| `n_activities` | Number of activities offered at the park |
| `n_entrances` | Number of entrance stations |
| `n_yearround_entrances` | Number of entrances open year-round |
| `avg_parking_cost` | Average parking fee |
| `total_lots` | Total number of parking lots |
| `pct_lots_accessible` | % of parking lots ADA accessible |
| `totalSpaces` | Total parking spaces |
| Entrance fee columns | One column per entrance fee type (e.g. Private Vehicle, Motorcycle) |

---

## Model

The model is defined in `train.py` and evaluated by `prepare.py`.

**Target variable:** `SDI` (Shannon Diversity Index)  
**Train/val split:** 80/20, random_state=42  
**Baseline:** LinearRegression with StandardScaler  

**Evaluation (fixed, do not modify):**
```python
val_rmse = sqrt(mean_squared_error(y_val, y_pred))
val_r2   = r2_score(y_val, y_pred)
```

**Model search order during AutoResearch:**
1. LinearRegression (baseline)
2. Ridge / Lasso / ElasticNet
3. PolynomialFeatures + Ridge
4. RandomForestRegressor
5. HistGradientBoostingRegressor
6. TransformedTargetRegressor with log transform
7. Hyperparameter tuning on best model

---

## AutoResearch Loop

The AutoResearch loop is an autonomous experiment cycle where an AI agent edits `train.py`, runs experiments, and logs results without human intervention.

**Run the loop:**
```bash
# initialize
git checkout -b autoresearch/$(date +%b%d | tr '[:upper:]' '[:lower:]')
python train.py > run.log 2>&1

# check results
grep "^val_rmse:\|^val_r2:" run.log

# log to results.tsv
echo -e "<commit>\t<val_rmse>\t<val_r2>\tkeep\t<model>\t<description>" >> results.tsv
```

See `program.md` for full loop instructions and rules.

**Hard rules:**
- Only `train.py` may be edited
- `val_rmse` calculation must never be changed
- `full_data_clean.csv` and `prepare.py` are read-only
- No new packages may be installed
- Each run must complete within 5 minutes

---

## Setup & Installation

### Requirements
```bash
pip install pandas numpy scikit-learn scipy requests thefuzz
```

### Python Environment
```bash
conda activate vscode_env
# or
/Users/isabellawoods/anaconda3/envs/vscode_env/bin/python
```

### API Keys Required
| API | Where to Get |
|-----|-------------|
| NASA Earthdata | https://urs.earthdata.nasa.gov/users/new |
| EPA AQS | https://aqs.epa.gov/data/api/signup |
| NPS API | https://www.nps.gov/subjects/developer/get-started.htm |

---

## Running the Project

### Run data collection (scraping notebook)
```bash
jupyter notebook scrapingData.ipynb
# or open in VS Code
```

### Run data preparation
```bash
python prepare.py
```

### Run baseline model
```bash
python train.py > run.log 2>&1
cat run.log
```

### Run AutoResearch loop (background, survives terminal close)
```bash
nohup python autoresearch.py > autoresearch.log 2>&1 &
echo "PID: $!"
tail -f autoresearch.log
```

---

## Results

| Commit | val_rmse | val_r2 | Status | Model | Description |
|--------|----------|--------|--------|-------|-------------|
| — | — | — | — | LinearRegression | baseline (run pending) |

*Results will be populated as AutoResearch experiments complete.*

---

## Known Issues & Limitations

- **iNaturalist rate limiting:** ~90 parks returned 0 records on the first run due to 429 errors. A retry run collected additional data but some parks may still have incomplete records.
- **Single-point satellite data:** NASA AppEEARS returns values for a single coordinate per park (the park center). Large parks like Yellowstone have significant internal variation not captured by a single point.
- **SDI sampling bias:** iNaturalist observations are biased toward accessible, popular areas within parks and toward charismatic species (birds, mammals, plants). This may underestimate true biodiversity in remote parks.
- **County-level pollution data:** EPA air quality and Census population are matched at the county level. Parks that span multiple counties or very large counties may have imprecise matches.
- **Small dataset:** ~170 parks is a small sample for ML. Cross-validation is used throughout to reduce overfitting risk.
- **Temporal mismatch:** Some features (NASA satellite data, iNaturalist SDI) are from 2021, while visitation/traffic data spans multiple decades. Slope features are used to partially address this.
