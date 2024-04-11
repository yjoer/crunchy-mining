# %%
import altair as alt
import numpy as np
import pandas as pd

# %%
df = pd.read_csv("data/SBA.csv")

# %%
df.info()

# %%
df["MIS_Status"].value_counts()

# %%
