# Databricks notebook source
! pip install dlt
! pip install mlflow
! pip install tensorflow=='2.11.0'
! pip install numpy=='1.21.5'
! pip install transformers=='4.29.2'


# COMMAND ----------

import mlflow
from mlflow import MlflowClient
#import dlt
import pandas as pd

model_name = 'summarizer - vineet_srivastava@dfci_harvard_edu'

client = MlflowClient()
client.search_registered_models(filter_string=f"name = '{model_name}'")

# COMMAND ----------

from datasets import load_dataset
from transformers import pipeline

# COMMAND ----------

model_version = 1
dev_model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{model_version}")
dev_model

# COMMAND ----------

@dlt.table(
comment="Transition from Staging to Production"
)
def dev_to_stage():
    client.transition_model_version_stage(model_name, model_version, "production")
    prod_data_path = f"/dbfs/mnt/dbacademy-datasets/temp/m6_prod_data"
    prod_data = spark.read.format("delta").option("header", "true").load(prod_data_path)

    prod_model_udf = mlflow.pyfunc.spark_udf(
    spark,
    model_uri=f"models:/{model_name}/Production",
    result_type="string",
    )

    batch_inference_results = prod_data.withColumn(
    "generated_summary", prod_model_udf("document")
    )
    
    return batch_inference_results


# COMMAND ----------

