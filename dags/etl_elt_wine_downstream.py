"""
## **ETL_ELT_wine_downstream**

This DAG, as downstream job of ***`ETL_ELT_wine`*** DAG, demonstrates the
behavior of the dataset feature in Airflow.

branch_task will be triggered after both postgresql_dataset and duckdb_dataset
were updated. Then branch_task will trigger extract_wine_data_postgresql task
or extract_wine_data_duckdb task, base on condition "day of the week". Next
chart_kde task would create chart with the data from branch_task triggered task.
"""

from airflow.decorators import dag, task
from airflow.models.dataset import Dataset
from pendulum import datetime, duration, today
import logging
import os
from pathlib import Path
from airflow.utils.trigger_rule import TriggerRule
from include.utils import plot_kde

logger = logging.getLogger("airflow.task")

_PROJECT_DIR = Path(__file__).resolve().parents[1]
_DUCKDB_INSTANCE_NAME = os.getenv("DUCKDB_INSTANCE_NAME", f"{_PROJECT_DIR}/data/winequality.db")
_DUCKDB_TABLE_NAME = os.getenv("DUCKDB_TABLE_NAME", "wine_data")
_DUCKDB_TABLE_URI = f"duckdb://{_DUCKDB_INSTANCE_NAME}/{_DUCKDB_TABLE_NAME}"

_POSTGRESQL_DB_NAME = os.getenv("POSTGRESQL_DB_NAME", "winequalitydb")
_POSTGRESQL_TABLE_NAME = os.getenv("POSTGRESQL_TABLE_NAME", "wine_data")
#Set postgresql uri according to real configuration.
_POSTGRESQL_URI = f"postgresql://user:password@localhost:5432/{_POSTGRESQL_DB_NAME}"
_PLOT_DIR = os.getenv("PLOT_DIR", f"{_PROJECT_DIR}/data/plot")


postgresql_dataset = Dataset(_POSTGRESQL_URI)
duckdb_dataset = Dataset(_DUCKDB_TABLE_URI)

@dag(
    dag_id="ETL_ELT_wine_downstream",
    start_date=datetime(2025, 1, 1),
    schedule=(postgresql_dataset & duckdb_dataset),
    catchup=False,
    max_consecutive_failed_dag_runs=5,
    max_active_runs=1,
    doc_md=__doc__,
    default_args={
        "owner": "wine",
        "retries": 3,
        "retry_delay": duration(seconds=30),
    },
    tags=["ETL", "ELT", "wine_quality", "dataset-scheduled"],
    concurrency=1,
    is_paused_upon_creation=False,
)
def chart_report_wine():
    @task
    def extract_wine_data_postgresql(
        postgresql_uri: str=_POSTGRESQL_URI,
        table_name: str=_POSTGRESQL_TABLE_NAME
    ):
        import pandas as pd
        from adbc_driver_postgresql import dbapi

        logger.info(f"Extract wine data from table {table_name} of PostgreSQL.")
        with dbapi.connect(postgresql_uri) as conn:
            df = pd.read_sql("SELECT quality,alcohol FROM wine_data", conn)
        logger.info("Extracted wine data from PostgreSQL.")

        return df

    @task
    def extract_wine_data_duckdb(
        duckdb_instance_name: str=_DUCKDB_INSTANCE_NAME,
        table_name: str=_DUCKDB_TABLE_NAME,
    ):
        import duckdb

        logger.info(f"Extract wine data from table {table_name} of DuckDB")
        with duckdb.connect(duckdb_instance_name) as conn:
            df = conn.sql("SELECT quality,alcohol FROM wine_data;").df()
        logger.info("Extracted wine data from DuckDB.")

        return df

    @task(task_id="chart_kde", trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS)
    def chart_report(ti):
        res = ti.xcom_pull(key="return_value", task_ids="branch_task")
        data = ti.xcom_pull(key="return_value", task_ids=res)
        plot_kde(df=data, path=_PLOT_DIR, title=f"Alcohol KDE {res.split('_')[-1]}")
        logger.info(f"Chart directory: {_PLOT_DIR}.")
        logger.info(f"Chart report with data from {res}")

        return {"row_count": len(data)}

    @task.branch(task_id="branch_task", multiple_outputs=False)
    def branching() -> str:
        if today().weekday() < 3:
            return "extract_wine_data_postgresql"
        else:
            return "extract_wine_data_duckdb"

    # Set dependencies
    branching() >> [extract_wine_data_postgresql(), extract_wine_data_duckdb()] >> chart_report()

# Instantiate the DAG
chart_report_wine()
