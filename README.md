<img width="1425" height="379" alt="image" src="https://github.com/user-attachments/assets/8a22d1d7-06a0-4172-8662-e18f6b383f8d" />


# QuantizedGeo-UAV

Near-Optimal Vector Quantization for Real-Time Remote Sensing-Driven Multi-UAV Coordination in Smart Cities with Bandwidth-Constrained Communications.

## Setup

```bash
pip install -r requirements.txt
```

## Run Commands

**Online mode** (fetches real building data from OpenStreetMap Overpass API):

```bash
python quantizedgeo_uav.py
```

```bash
python quantizedgeo_uav.py --offline
```

## Outputs

All outputs are saved to `./quantizedgeo_outputs/`:

## Data Sources

Online mode queries OpenStreetMap for building footprints, heights, road widths, helipads, parks, hospitals, schools, and restricted areas across five city centers (Tokyo, Shenzhen, Seoul, Singapore, Mumbai).

Offline fallback data is derived from the Skyscraper Center database, PNAS 2022 satellite mapping study, MDPI Buildings 2018, and Shenzhen urbanization research (ScienceDirect 2023).
