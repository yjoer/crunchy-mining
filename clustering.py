# %%
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from itables import show

# %matplotlib inline
# %config InlineBackend.figure_formats = ['retina']

# %%
df = pd.read_parquet("data/nyc/yellow_tripdata_2024-03.parquet")

# %%
gdf = gpd.read_file("zip://data/nyc/taxi_zones.zip")

# %%
show(gdf, scrollX=True)

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

# %%
gdf["geometry_x"] = gdf["geometry"].apply(lambda x: x.centroid.x)
gdf["geometry_y"] = gdf["geometry"].apply(lambda x: x.centroid.y)

# %%
df_pu = (
    df.merge(
        right=gdf[["OBJECTID", "geometry_x", "geometry_y"]],
        left_on="PULocationID",
        right_on="OBJECTID",
    )
    .drop("OBJECTID", axis=1)
    .rename(mapper={"geometry_x": "pick_up_x", "geometry_y": "pick_up_y"}, axis=1)
)

# %%
df_do = (
    df_pu.merge(
        right=gdf[["OBJECTID", "geometry_x", "geometry_y"]],
        left_on="DOLocationID",
        right_on="OBJECTID",
    )
    .drop("OBJECTID", axis=1)
    .rename(mapper={"geometry_x": "drop_off_x", "geometry_y": "drop_off_y"}, axis=1)
)

# %%
df_final = df_do.rename(
    {
        "VendorID": "vendor_id",
        "RatecodeID": "rate_code_id",
        "PULocationID": "pick_up_location_id",
        "DOLocationID": "drop_off_location_id",
        "Airport_fee": "airport_fee",
    },
    axis=1,
)

# %%
show(df_final.iloc[:300], scrollX=True)

# %%
# df_final.to_parquet("data/nyc/yellow_tripdata_2024-03.uc.parquet", compression=None)

# %%
