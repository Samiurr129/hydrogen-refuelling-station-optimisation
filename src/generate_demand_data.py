"""Generate a synthetic hydrogen demand dataset for UK freight.

The points are simulated, not measured. They cluster around real UK freight hubs
(ports and distribution centres), with daily demand drawn from a log-normal distribution.

Run from the repository root:
    python src/generate_demand_data.py
Output: data/hydrogen_demand.csv  (columns: site_id, latitude, longitude, demand_kg_per_day, hub)
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SEED = 42

# hub name, latitude, longitude, relative size, spread (km)
HUBS = [
    ("Felixstowe port", 51.96, 1.33, 1.6, 15),
    ("Southampton port", 50.90, -1.40, 1.2, 15),
    ("Immingham port", 53.62, -0.19, 1.0, 15),
    ("Midlands golden triangle", 52.45, -1.45, 2.5, 35),
    ("Greater London", 51.50, -0.12, 2.0, 30),
    ("Manchester and Trafford", 53.47, -2.30, 1.5, 25),
    ("Leeds and Bradford", 53.78, -1.65, 1.2, 20),
    ("Bristol and Avonmouth", 51.50, -2.68, 0.9, 20),
    ("Glasgow", 55.86, -4.25, 0.9, 20),
    ("Teesside", 54.57, -1.23, 0.8, 15),
]
N_SITES = 600


def main() -> None:
    rng = np.random.default_rng(SEED)
    weights = np.array([h[3] for h in HUBS])
    hub_idx = rng.choice(len(HUBS), size=N_SITES, p=weights / weights.sum())
    rows = []
    for i, h in enumerate(hub_idx):
        name, lat, lon, _, spread_km = HUBS[h]
        dlat = rng.normal(0, spread_km) / 111.0
        dlon = rng.normal(0, spread_km) / (111.0 * np.cos(np.radians(lat)))
        demand = rng.lognormal(mean=np.log(180), sigma=0.6)   # kg H2 per day per site
        rows.append((i + 1, round(lat + dlat, 5), round(lon + dlon, 5), round(demand, 1), name))
    df = pd.DataFrame(rows, columns=["site_id", "latitude", "longitude", "demand_kg_per_day", "hub"])
    out = ROOT / "data" / "hydrogen_demand.csv"
    out.parent.mkdir(exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Wrote {len(df)} synthetic demand sites to {out.relative_to(ROOT)}; "
          f"total demand {df['demand_kg_per_day'].sum():,.0f} kg/day")


if __name__ == "__main__":
    main()
