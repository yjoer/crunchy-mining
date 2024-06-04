# %%
# %env JAVA_HOME="C:\Progra~1\Eclips~1\jdk-17.0.11.9-hotspot"
# %env PYSPARK_PYTHON="python"

# %%
import pandas as pd
import pyspark.sql.functions as F
from itables import show
from pyspark.ml.fpm import FPGrowth
from pyspark.sql import SparkSession

# %%
spark = (
    SparkSession.builder.appName("FP-Growth")
    .config("spark.driver.extraJavaOptions", "-Xss10m")
    .getOrCreate()
)

# %% [markdown]
# ## Sample

# %%
df = pd.read_excel("data/Online Retail.xlsx")

# %%
df_spark = spark.createDataFrame(df).cache()

# %%
df_spark.printSchema()

# %%
df_spark.show()

# %% [markdown]
# ## Modify

# %% [markdown]
# ### Rename

# %%
df_renamed = (
    df_spark.withColumnRenamed("InvoiceNo", "invoice_no")
    .withColumnRenamed("StockCode", "stock_code")
    .withColumnRenamed("Description", "description")
    .withColumnRenamed("Quantity", "quantity")
    .withColumnRenamed("InvoiceDate", "invoice_date")
    .withColumnRenamed("UnitPrice", "unit_price")
    .withColumnRenamed("CustomerID", "customer_id")
    .withColumnRenamed("Country", "country")
)

# %% [markdown]
# ### Remove Duplicates

# %%
df_deduped = df_renamed.dropDuplicates()

# %%
df_renamed.count() - df_deduped.count()

# %%
df_deduped_t = df_deduped.dropDuplicates(["invoice_no", "description"])

# %%
df_deduped.count() - df_deduped_t.count()

# %% [markdown]
# ### Filter Examples

# %%
df_filtered = (
    df_deduped_t.filter(F.col("description").isNotNull())
    .filter(F.col("description") != "")
    .filter(F.col("invoice_no").isNotNull())
    .filter(F.col("invoice_no") != "")
    .filter(F.col("quantity") > 0)
    .filter(F.col("unit_price") > 0)
)

# %%
df_deduped_t.count() - df_filtered.count()

# %% [markdown]
# ### Aggregate

# %%
df_agg = df_filtered.groupBy("invoice_no").agg(
    F.collect_list("description").alias("descriptions")
)

# %%
n_transactions = df_agg.count()
n_transactions

# %% [markdown]
# ## Model

# %% [markdown]
# ### FP-Growth

# %%
fp = FPGrowth(itemsCol="descriptions", minSupport=0.01, minConfidence=0.8)

# %%
fp_model = fp.fit(df_agg)

# %%
itemsets = (
    fp_model.freqItemsets.withColumn("support", F.col("freq") / n_transactions)
    .sort(F.col("support").desc())
    .toPandas()
)

# %%
show(itemsets, scrollX=True)

# %%
rules = fp_model.associationRules.toPandas()

# %%
show(rules, scrollX=True)

# %%
