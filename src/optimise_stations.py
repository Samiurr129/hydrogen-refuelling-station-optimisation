"""Propose hydrogen refuelling station locations with demand-weighted K-Means.

Run from the repository root (after src/generate_demand_data.py):
    python src/optimise_stations.py --stations 10 --radius-km 25

Outputs in results/:
    proposed_station_locations.csv   station coordinates and the demand each serves
    clustered_hydrogen_demand.csv    each demand site with its station and distance
    coverage_by_station_count.csv    demand coverage for 3 to 20 stations
    station_map.png, coverage_curve.png
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
EARTH_R_KM = 6371.0
SEED = 42


def to_km(lat, lon, lat0):
    """Equirectangular projection to km, so K-Means distances are real distances."""
    x = np.radians(lon) * EARTH_R_KM * np.cos(np.radians(lat0))
    y = np.radians(lat) * EARTH_R_KM
    return np.column_stack([x, y])


def from_km(xy, lat0):
    lat = np.degrees(xy[:, 1] / EARTH_R_KM)
    lon = np.degrees(xy[:, 0] / (EARTH_R_KM * np.cos(np.radians(lat0))))
    return lat, lon


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 2 * EARTH_R_KM * np.arcsin(np.sqrt(a))


def place_stations(data, k, lat0):
    xy = to_km(data["latitude"].values, data["longitude"].values, lat0)
    km = KMeans(n_clusters=k, n_init=10, random_state=SEED)
    labels = km.fit_predict(xy, sample_weight=data["demand_kg_per_day"].values)
    lat, lon = from_km(km.cluster_centers_, lat0)
    dist = haversine_km(data["latitude"].values, data["longitude"].values, lat[labels], lon[labels])
    return labels, lat, lon, dist


def coverage(data, dist, radius_km):
    w = data["demand_kg_per_day"].values
    return w[dist <= radius_km].sum() / w.sum()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "hydrogen_demand.csv")
    ap.add_argument("--stations", type=int, default=10)
    ap.add_argument("--radius-km", type=float, default=25.0,
                    help="a site counts as covered if its station is within this distance")
    args = ap.parse_args()

    data = pd.read_csv(args.data)
    required = {"latitude", "longitude", "demand_kg_per_day"}
    missing = required - set(data.columns)
    if missing:
        raise SystemExit(f"{args.data} is missing columns: {sorted(missing)}")
    lat0 = data["latitude"].mean()
    out = ROOT / "results"
    out.mkdir(exist_ok=True)

    # How coverage grows with the number of stations.
    curve = []
    for k in range(3, 21):
        _, _, _, d = place_stations(data, k, lat0)
        curve.append({"stations": k, "demand_coverage_pct": round(100 * coverage(data, d, args.radius_km), 1),
                      "demand_weighted_mean_km": round(np.average(d, weights=data["demand_kg_per_day"]), 1)})
    curve = pd.DataFrame(curve)
    curve.to_csv(out / "coverage_by_station_count.csv", index=False)

    labels, lat, lon, dist = place_stations(data, args.stations, lat0)
    data = data.assign(station=labels + 1, distance_to_station_km=dist.round(2))
    stations = (data.groupby("station")
                    .agg(sites=("latitude", "size"), demand_kg_per_day=("demand_kg_per_day", "sum"),
                         max_distance_km=("distance_to_station_km", "max"))
                    .reset_index())
    stations.insert(1, "latitude", lat.round(5))
    stations.insert(2, "longitude", lon.round(5))
    stations = stations.sort_values("demand_kg_per_day", ascending=False)
    stations.to_csv(out / "proposed_station_locations.csv", index=False)
    data.to_csv(out / "clustered_hydrogen_demand.csv", index=False)

    cov = coverage(data, dist, args.radius_km)
    print(f"{args.stations} stations: {cov:.1%} of demand within {args.radius_km:g} km; "
          f"demand-weighted mean distance {np.average(dist, weights=data['demand_kg_per_day']):.1f} km")
    print(stations.to_string(index=False))

    # Map.
    fig, ax = plt.subplots(figsize=(7, 9))
    sns.scatterplot(data=data, x="longitude", y="latitude", hue="station", palette="tab20",
                    size="demand_kg_per_day", sizes=(8, 80), alpha=0.6, legend=False, ax=ax)
    ax.scatter(lon, lat, c="black", marker="X", s=180, label="Proposed station")
    ax.set_aspect(1 / np.cos(np.radians(lat0)))
    ax.set(xlabel="Longitude", ylabel="Latitude",
           title=f"{args.stations} demand-weighted station sites (synthetic UK freight demand)")
    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(out / "station_map.png", dpi=150)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(curve["stations"], curve["demand_coverage_pct"], marker="o")
    ax.axvline(args.stations, ls="--", c="grey", lw=1)
    ax.set(xlabel="Number of stations", ylabel=f"% of demand within {args.radius_km:g} km",
           title="Demand coverage by number of stations")
    plt.tight_layout()
    plt.savefig(out / "coverage_curve.png", dpi=150)
    print(f"Saved results to {out.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
