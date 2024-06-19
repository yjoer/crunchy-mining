# %%
import gc

import geopandas as gpd
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from itables import show
from sklearn.cluster import DBSCAN
from tslearn.clustering import TimeSeriesKMeans

# %matplotlib inline
# %config InlineBackend.figure_formats = ['retina']

pd.set_option("mode.copy_on_write", True)

# %% [markdown]
# ## Sample

# %%
df = pd.read_parquet("data/nyc/yellow_tripdata_2024-03.parquet")

# %%
gdf = gpd.read_file("zip://data/nyc/taxi_zones.zip")

# %%
show(gdf, scrollX=True)

# %% [markdown]
# ## Explore

# %%
gdf.borough.unique()


# %%
def plot_borough(borough: str, figsize: tuple):
    fig = plt.figure(figsize=figsize)
    ax = fig.gca()

    if borough == "all":
        gdf_subset = gdf[gdf["borough"] != "Manhattan"]
    else:
        gdf_subset = gdf[gdf["borough"] == borough]

    gdf_subset.plot(ax=ax)

    for t in gdf_subset.itertuples():
        centroid = t.geometry.centroid
        ax.text(
            centroid.x,
            centroid.y,
            t.OBJECTID,
            fontsize=8,
            color="white",
            ha="center",
            va="center",
        )

    plt.show()


# %%
plot_borough("all", (12, 12))
plot_borough("Manhattan", (12, 12))

# %% [markdown]
# ## Modify

# %% [markdown]
# ### Rename

# %%
df_renamed = df.rename(
    {
        "VendorID": "vendor_id",
        "RatecodeID": "rate_code_id",
        "PULocationID": "pick_up_location_id",
        "DOLocationID": "drop_off_location_id",
        "Airport_fee": "airport_fee",
    },
    axis=1,
)

# %% [markdown]
# ### Remove Duplicates

# %%
df_deduped = df_renamed.drop_duplicates()
len(df_renamed) - len(df_deduped)

# %% [markdown]
# ### Filter Examples

# %%
start_date_mask = df_deduped["tpep_pickup_datetime"] >= "2024-03-01"
end_date_mask = df_deduped["tpep_pickup_datetime"] < "2024-04-01"

df_filtered = df_renamed[start_date_mask & end_date_mask]
len(df_deduped) - len(df_filtered)

# %% [markdown]
# ### Integrate Centroids

# %%
gdf["geometry_x"] = gdf["geometry"].apply(lambda x: x.centroid.x)
gdf["geometry_y"] = gdf["geometry"].apply(lambda x: x.centroid.y)

# %%
df_pu = (
    df_filtered.merge(
        right=gdf[["OBJECTID", "geometry_x", "geometry_y"]],
        left_on="pick_up_location_id",
        right_on="OBJECTID",
    )
    .drop("OBJECTID", axis=1)
    .rename(mapper={"geometry_x": "pick_up_x", "geometry_y": "pick_up_y"}, axis=1)
)

# %%
df_do = (
    df_pu.merge(
        right=gdf[["OBJECTID", "geometry_x", "geometry_y"]],
        left_on="drop_off_location_id",
        right_on="OBJECTID",
    )
    .drop("OBJECTID", axis=1)
    .rename(mapper={"geometry_x": "drop_off_x", "geometry_y": "drop_off_y"}, axis=1)
)

# %%
len(df_filtered) - len(df_do)

# %%
show(df_do.iloc[:300], scrollX=True)

# %%
# df_do.to_parquet("data/nyc/yellow_tripdata_2024-03.uc.parquet", compression=None)

# %%
del df
del df_renamed
del df_deduped
del df_filtered
del df_pu

gc.collect()

# %% [markdown]
# ## Model

# %% [markdown]
# ### Time-Series Clustering

# %%
df_ts = df_do.set_index("tpep_pickup_datetime")
n_passengers = df_ts["passenger_count"].resample("h").sum().to_frame()

# %%
n_passengers["date"] = n_passengers.index.strftime("%Y-%m-%d")
n_passengers["time"] = n_passengers.index.strftime("%H:%M")

n_passengers_daily = n_passengers.pivot(
    index="date",
    columns="time",
    values="passenger_count",
)

n_passengers_daily["day_name"] = n_passengers_daily.index.map(
    lambda x: pd.to_datetime(x).day_name()
)

n_passengers_daily_inp = n_passengers_daily.iloc[:, :24]

# %% [markdown]
# #### K-Means DTW

# %%
kmeans_ts = TimeSeriesKMeans(n_clusters=3, metric="dtw", random_state=12345)
n_passengers_daily["kmeans_clusters"] = kmeans_ts.fit_predict(n_passengers_daily_inp)

# %% [markdown]
# #### DBSCAN

# %%
dbscan_ts = DBSCAN(eps=3500, min_samples=5)
n_passengers_daily["dbscan_clusters"] = dbscan_ts.fit_predict(n_passengers_daily_inp)

# %% [markdown]
# #### Results

# %%
show(n_passengers_daily, scrollX=True)


# %%
def plot_clusters_ts(algorithm: str):
    col_name = f"{algorithm}_clusters"

    plt.figure(figsize=(10, 6))
    colors = plt.cm.tab10(np.arange(10))
    anomalous_idx = n_passengers_daily[col_name].max() + 1

    for i, day in enumerate(n_passengers_daily.index):
        cluster = n_passengers_daily[col_name].iloc[i]

        if cluster == -1:
            color = anomalous_idx
            label = "Noises"
        else:
            label = f"Cluster {cluster}"
            color = cluster

        plt.plot(
            n_passengers_daily.columns[:24],
            n_passengers_daily.iloc[:, :24].loc[day],
            color=colors[color],
            label=label,
        )

    handles, labels = plt.gca().get_legend_handles_labels()
    unique_labels = list(set(labels))
    unique_handles = [handles[labels.index(label)] for label in unique_labels]

    plt.ylabel("Passenger Count")
    plt.xlabel("Hours (March 2024)")
    plt.xticks(rotation=45)

    handles, labels = plt.gca().get_legend_handles_labels()
    unique_labels = sorted(list(set(labels)))
    unique_handles = [handles[labels.index(label)] for label in unique_labels]
    plt.legend(unique_handles, unique_labels)

    plt.show()


# %%
plot_clusters_ts(algorithm="kmeans")

# %%
plot_clusters_ts(algorithm="dbscan")


# %%
def tabulate_clusters_ts(algorithm: str):
    n_passengers_daily_agg = (
        n_passengers_daily.reset_index()
        .groupby(f"{algorithm}_clusters")
        .agg({"date": list, "day_name": lambda x: list(set(x))})
    )

    show(n_passengers_daily_agg, scrollX=True)


# %%
tabulate_clusters_ts(algorithm="kmeans")

# %%
tabulate_clusters_ts(algorithm="dbscan")

# %% [markdown]
# ### Anomaly Detection in Time Series

# %%
df_tp = df_do.set_index("tpep_pickup_datetime")
n_passengers_min = df_tp["passenger_count"].resample("min").sum().to_frame()
n_passengers_min["day_name"] = n_passengers_min.index.map(lambda x: x.day_name())

# %% [markdown]
# #### DBSCAN

# %%
dbscan_tp = DBSCAN(eps=1, min_samples=50)
n_passengers_min["clusters"] = dbscan_tp.fit_predict(n_passengers_min.iloc[:, [0]])

# %% [markdown]
# #### Results

# %%
n_passengers_min_noises = n_passengers_min[n_passengers_min["clusters"] == -1]
show(n_passengers_min_noises, scrollX=True)

# %%
plt.figure(figsize=(12, 8))

for cluster in n_passengers_min["clusters"].unique():
    cluster_df = n_passengers_min[n_passengers_min["clusters"] == cluster]

    if cluster == -1:
        label = "Anomalous"
    else:
        label = "Normal Points"

    plt.scatter(cluster_df.index, cluster_df["passenger_count"], label=label)

plt.ylabel("Passenger Count")
plt.xlabel("Days (March 2024)")
plt.xticks(pd.date_range("2024-03-01", "2024-04-01"), rotation=45)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
plt.legend()

plt.show()

# %%
days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
days_ord = {day: i for i, day in enumerate(days)}

n_passengers_min_noises_days = n_passengers_min_noises.groupby("day_name").size()
n_passengers_min_noises_days = n_passengers_min_noises_days.to_frame()
n_passengers_min_noises_days["order"] = n_passengers_min_noises_days.index.map(days_ord)
n_passengers_min_noises_days.sort_values(by="order", inplace=True)
show(n_passengers_min_noises_days, scrollX=True)

# %%
plt.figure(figsize=(6, 3))

plt.bar(n_passengers_min_noises_days.index, n_passengers_min_noises_days[0])
plt.ylabel("Count")
plt.xlabel("Days of the Week")
plt.show()

# %%
hours = n_passengers_min_noises.index.hour
n_passengers_min_noises_hours = n_passengers_min_noises.groupby(hours).size()

show(n_passengers_min_noises_hours, scrollX=True)

# %%
plt.figure(figsize=(6, 3))

plt.bar(n_passengers_min_noises_hours.index, n_passengers_min_noises_hours)
plt.ylabel("Count")
plt.xlabel("Hours of a Day")
plt.xticks(n_passengers_min_noises_hours.index)
plt.show()
