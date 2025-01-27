"""
## **Wine ETL and ELT**

This DAG demonstrates ETL and ELT pipelines using Airflow.

* The ETL pipeline mocks data extraction for data about wine quality using a
modularized function, filters the data based on the wine quality class/level,
and loads the filtered data into a DuckDB database. A branch task follows the
ETL pipeline. The branch task decides which downstream task will be triggered.
One possible choice is ML task group, including data preprocess task, choose
model task and model training task. The trained model could predict alcohol
index of wine according to residual sugar, pH and other chemical or physical
indices. The other choice is a report task, named 'print_loaded_wine'. It will
print out basic statistics info of alcohol of different qualities in logs.

  ***`Current ML pipeline as a potential downstream task of ETL/ELT, only
covers part of a full ML lifecycle.`***

* The ELT pipeline shares the same data source with above ETL pipeline,
loads the raw data into a PostgreSQL database, filters the data based on the
total_sulfur_dioxide index of wine. A chart task, named "chart_kde", as a
downstream task of the ELT pipeline. The chart task will calculate the
"kernel density estimate (KDE)" of alcohol of different qualities, and draw
a picture of the result with name "alcohol_kde.pdf".

  ***`Metabase`*** accesses the PostgreSQL database as data source for further
analysis or BI jobs.

Both PostgreSQL database and DuckDB database are set as dataset. Another dag
named ***`ETL_ELT_wine_downstream`*** will be triggered after these databases
were updated.
"""

from airflow.decorators import dag, task, task_group
from airflow.models.dataset import Dataset
from airflow.models.baseoperator import chain
from airflow.models.param import Param
from pendulum import datetime, duration, today
import logging
import os
from pathlib import Path

logger = logging.getLogger("airflow.task")

_HIGH_QUALITY_THRESHOLD_PARAMETER_NAME = "high_quality_threshold" #Default 7.
_ML_SAMPLE_COUNT_THRESHOLD_PARAMETER_NAME = "ml_sample_count_threshold" #Default 200.

_PROJECT_DIR = Path(__file__).resolve().parents[1]
_WINE_DATA_SOURCE = os.getenv("WINE_DATA_SOURCE", f"{_PROJECT_DIR}/data/winequality-white.csv")
_DUCKDB_INSTANCE_NAME = os.getenv("DUCKDB_INSTANCE_NAME", f"{_PROJECT_DIR}/data/winequality.db")
_DUCKDB_TABLE_NAME = os.getenv("DUCKDB_TABLE_NAME", "wine_data")
_DUCKDB_TABLE_URI = f"duckdb://{_DUCKDB_INSTANCE_NAME}/{_DUCKDB_TABLE_NAME}"

_SAMPLES_FRAC = os.getenv("SAMPLES_FRAC", 1.0) #Default 1.0, select all samples.
_POSTGRESQL_DB_NAME = os.getenv("POSTGRESQL_DB_NAME", "winequalitydb")
_POSTGRESQL_TABLE_NAME = os.getenv("POSTGRESQL_TABLE_NAME", "wine_data")
#Set postgresql uri according to real configuration.
_POSTGRESQL_URI = f"postgresql://user:password@localhost:5432/{_POSTGRESQL_DB_NAME}"
_PLOT_DIR = os.getenv("PLOT_DIR", f"{_PROJECT_DIR}/data/plot")

_MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
_ARTIFACT_LOCATION = os.getenv("ARTIFACT_LOCATION", f"{_PROJECT_DIR}/data/mlruns")
_EXPERIMENT_NAME = os.getenv("EXPERIMENT_NAME", "wine_quality")


@dag(
    dag_id="ETL_ELT_wine",
    start_date=datetime(2025, 1, 1),
    schedule=None,
    catchup=False,
    max_consecutive_failed_dag_runs=5,
    max_active_runs=1,
    doc_md=__doc__,
    default_args={
        "owner": "wine",
        "retries": 3,
        "retry_delay": duration(seconds=30),
    },
    tags=["ETL", "ELT", "ML", "XCom", "dataset", "dynamic_task", "mlflow", "wine_quality"],
    params={
        _HIGH_QUALITY_THRESHOLD_PARAMETER_NAME: Param(
            7,
            type="number",
            title="wine quality threshold",
            description="Scale from 3 to 9, as integer. 9 is the best class/level, and 3 is ther worest class/level. Current default threshold is 7.",
        ),
        _ML_SAMPLE_COUNT_THRESHOLD_PARAMETER_NAME: Param(
            200,
            type="number",
            title="ml sample count threshold",
            description="Sample count threshold of ML model training. Default value is 200.",
        ),
    },
    concurrency=1,
    is_paused_upon_creation=False,
)
def etl_elt_wine():
    """
    ETL and ELT
    ETL:
    extract_wine_data
           v
    transform_wine_data  create_wine_table_in_duckdb
           |                     |
           +---------------------+
           v
    load_wine_data_into_duckdb

    ELT:
    Requirement: An available PostgreSQL database
                 named "winequalitydb"
    extract_wine_data
           v
    load_wine_data_into_postgresql
           v
    transform_wine_data_from_postgresql
    """
    @task(retries=2)
    def create_wine_table_in_duckdb(
        duckdb_instance_name: str=_DUCKDB_INSTANCE_NAME,
        table_name: str=_DUCKDB_TABLE_NAME,
    ):
        import duckdb

        logger.info("Creating table in DuckDB.")
        with duckdb.connect(duckdb_instance_name) as con:
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS wine_data (
                    id INT PRIMARY KEY,
                    alcohol FLOAT,
                    chlorides FLOAT,
                    citric_acid FLOAT,
                    density FLOAT,
                    fixed_acidity FLOAT,
                    free_sulfur_dioxide FLOAT,
                    high_quality BOOL,
                    pH FLOAT,
                    quality INT,
                    residual_sugar FLOAT,
                    sulphates FLOAT,
                    total_sulfur_dioxide FLOAT,
                    volatile_acidity FLOAT
                )"""
            )
        logger.info(f"Table {table_name} created in DuckDB.")


    @task
    def extract_wine_data(
        samples_frac: float=_SAMPLES_FRAC,
        path: str=_WINE_DATA_SOURCE
    ):
        """
        Retrieve data about wine. This task simulates an
        extraction step in an ETL/ELT pipeline.
        Args:
            samples_frac (float): Fraction of samples to return.
                Default is 1.0, select all the 4898 smaples of
                wine.
            path (str): Source path.
        Returns:
            A pd.DataFrame about wine.
        """
        from include.utils import get_wine_data

        logger.info(f"Retrieve data from {path}")
        wine_df = get_wine_data(samples_frac=samples_frac, path=path)

        return wine_df


    @task
    def transform_wine_data(wine_df, **context):
        """
        Filter the wine data based on the quality threshold. The
        quality scale from 3 to 9 as int. 9 stands for the best class,
        and 3 stands for the worest class.
        This task simulates a transformation step in an ETL pipeline.
        Args:
            wine_df (pd.DataFrame): A DataFrame to be transformed.
        Returns:
            Transformed data.
        """
        high_quality_threshold = context["params"][_HIGH_QUALITY_THRESHOLD_PARAMETER_NAME]

        logger.info(f"Filtering high quality wine. Threshold as: {high_quality_threshold}.")
        wine_df["high_quality"] = (wine_df["quality"] >= high_quality_threshold)
        filtered_wine_df = wine_df[(wine_df["quality"] >= high_quality_threshold)]
        logger.info(f"Total {len(filtered_wine_df)} high quality samples.")

        return filtered_wine_df


    @task(outlets=[Dataset(_DUCKDB_TABLE_URI)])
    def load_wine_data_into_duckdb(
        filtered_wine_df,
        duckdb_instance_name: str=_DUCKDB_INSTANCE_NAME,
        table_name: str=_DUCKDB_TABLE_NAME,
    ) -> dict:
        """
        Load the filtered wine data into a DuckDB database. This task
        simulates a loading step in an ETL pipeline.
        Args:
            filtered_wine_df (pd.DataFrame): Transformed data to be loaded.
            duckdb_instance_name (str): The name of the DuckDB instance.
            table_name (str): The name of the table to load the data into.
        Returns:
            Count of loaded rows.
        """
        import duckdb

        logger.info("Loading wine data into DuckDB.")
        with duckdb.connect(duckdb_instance_name) as con:
            con.sql(
                "INSERT OR IGNORE INTO wine_data BY NAME SELECT * FROM filtered_wine_df;"
            )
        logger.info("wine data loaded into DuckDB.")

        return {"row_count": len(filtered_wine_df)}


    @task(outlets=[Dataset(_POSTGRESQL_URI)])
    def load_wine_data_into_postgresql(
        wine_df,
        postgresql_uri: str=_POSTGRESQL_URI,
        table_name: str=_POSTGRESQL_TABLE_NAME
    ) -> dict:
        """
        Load raw wine data into a PostgreSQL database. This task
        simulates a loading step in an ELT pipeline.
        Args:
            wine_df (pd.DataFrame): The raw wine data to be loaded.
            postgresql_uri (str): The URI of PostgreSQL database.
            table_name (str): The name of the table to load the data into.
        Returns:
            Count of loaded rows.
        """
        from adbc_driver_postgresql import dbapi

        logger.info("Loading wine data into PostgreSQL.")
        with dbapi.connect(postgresql_uri) as con:
            wine_df.to_sql(name=table_name, con=con, if_exists="replace", index=False)
        logger.info("wine data loaded into PostgreSQL.")

        return {"row_count": len(wine_df)}


    @task
    def transform_wine_data_from_postgresql(
        postgresql_uri: str=_POSTGRESQL_URI,
        table_name: str=_POSTGRESQL_TABLE_NAME,
    ):
        """
        Filter the wine data based on the total_sulfur_dioxide index.
        The threshold hardcoded as 125.
        This task simulates a transformation step in an ELT pipeline.
        Args:
            postgresql_uri (str): The URI of PostgreSQL database.
            table_name (str): The name of the table to retrieve data.
        Returns:
            Transformed data.
        """
        from adbc_driver_postgresql import dbapi
        import pandas as pd

        logger.info("Filter wine data from PostgreSQL.")
        with dbapi.connect(postgresql_uri) as con:
            filtered_wine_df = pd.read_sql(
                "SELECT quality,alcohol FROM wine_data WHERE total_sulfur_dioxide < 125",
                con
            )
        logger.info(f"Filtered {len(filtered_wine_df)} wine samples from PostgreSQL.")

        return filtered_wine_df


    @task(task_id="chart_kde")
    def chart_loaded_wine(wine_df, path=_PLOT_DIR):
        """
        Calculate the "kernel density estimate (KDE)" of alcohol of different
        qualities, and draw a picture of the result, named "alcohol_kde.pdf".
        Args:
            wine_df (pd.DataFrame): Input data with "alocohl" and "quality" columns.
            path (str): directory of output file.
        Output:
            A chart report named "alcohol_kde.pdf".
        """
        from include.utils import plot_kde

        logger.info("Chart report start.")
        plot_kde(df=wine_df, path=path)
        logger.info(f"Chart directory: {path}.")
        logger.info("Chart report finished.")


    @task.branch(task_id="branch_task")
    def branching(**context):
        """
        Choose tasks based on day of the week and high quality sample count.
        """
        row_count = context["ti"].xcom_pull(key="row_count", task_ids="load_wine_data_into_duckdb")
        ml_sample_count_threshold = context["params"][_ML_SAMPLE_COUNT_THRESHOLD_PARAMETER_NAME]

        if today().weekday() <= 5 and row_count >= ml_sample_count_threshold:
            logger.info(f"High quality samples: {row_count}")
            return ["ml_task_group.data_preprocess_task",
                    "ml_task_group.choose_model_task",
                    "print_loaded_wine"]
        elif today().weekday() <= 5 and row_count < ml_sample_count_threshold:
            logger.info(f"Only {row_count} high quality samples, less than ml sample count threshold {ml_sample_count_threshold}.")
            return "print_loaded_wine"
        else:
            logger.info(f"High quality samples: {row_count}")
            return "print_loaded_wine"


    @task_group
    def ml_task_group():
        @task
        def data_preprocess_task(
            duckdb_instance_name: str=_DUCKDB_INSTANCE_NAME,
            table_name: str=_DUCKDB_TABLE_NAME,
        ) -> tuple:
            """
            Extract data from DuckDB.
            Args:
                duckdb_instance_name (str): The duckdb database instance.
                table_name (str): The name of the table to retrieve data.
            Returns:
                Train and test datasets.
            """
            import duckdb
            from include.utils import extract_split_data

            logger.info(f"Extracting data from {duckdb_instance_name}.")
            data = extract_split_data(duckdb_instance_name, table_name)

            return data


        @task
        def choose_model_task() -> list[int]:
            """
            Randomly choose model index from predefined model list.
            Returns:
                A list of int. Each element stands for a selected
                model's index from model list.
            """
            from include.utils import build_model, choose_model

            model_list = build_model()
            ret = choose_model(model_list)
            logger.info(f"Select model(s): {ret}.")

            return ret


        @task
        def train_model_task(data: tuple, model_index: int) -> dict:
            """
            Train ML model.
            Args:
                data (tuple): X_train, X_test, y_train, y_test.
                model_index (int): Index of selected model from
                    predefined model list.
            Returns:
                A dict including model info. For example, a
                LinearRegression pipeline's result looks like
                    {"model_name": "lr",
                     "model_pipe": str(lr_pipe),
                     "model_params": lr_param_grid,
                     "r2_train": r2_train,
                     "r2_test": r2_test
                    }
            """
            from include.utils import train_model

            logger.info("Model training process start.")
            result = train_model(data=data, model_index=model_index)
            logger.info("Model training process finished.")

            return result


        task1 = data_preprocess_task()
        task2 = choose_model_task()
        task3 = train_model_task.partial(data=task1).expand(model_index=task2)


    @task
    def print_loaded_wine(
        duckdb_instance_name: str=_DUCKDB_INSTANCE_NAME,
        table_name: str=_DUCKDB_TABLE_NAME,
    ):
        """
        Get the wine stored in the DuckDB database then calculate the
        alcohol info, grouped by wine quality and print result to the logs.
        Args:
            duckdb_instance_name (str): The duckdb database where the data
                is stored.
            table_name (str): The name of the table to retrieve data.
        """
        import duckdb
        from tabulate import tabulate

        with duckdb.connect(duckdb_instance_name) as con:
            wine_df = con.sql("SELECT alcohol,quality FROM wine_data;").df()

        wine_df = wine_df.groupby("quality", as_index=False)[["alcohol"]]\
        .agg(["mean","median","std","min","max","count"])\
        .sort_values(by="quality",ascending=False)

        logger.info(f"Alcohol info from {len(wine_df)} different qualities.")
        logger.info(tabulate(wine_df, headers="keys", tablefmt="pretty"))


    #Set dependencies
    #ETL and downstream tasks.
    create_wine_table_in_duckdb_obj = create_wine_table_in_duckdb()
    extract_wine_data_obj = extract_wine_data()
    transform_wine_data_duckdb_obj = transform_wine_data(extract_wine_data_obj)
    load_wine_data_duckdb_obj = load_wine_data_into_duckdb(transform_wine_data_duckdb_obj)

    chain(
        create_wine_table_in_duckdb_obj,
        load_wine_data_duckdb_obj,
        branching(),
        [ml_task_group(), print_loaded_wine()]
    )

    #ELT and following chart task.
    load_wine_data_postgresql_obj = load_wine_data_into_postgresql(extract_wine_data_obj)
    transform_wine_data_from_postgresql_obj = transform_wine_data_from_postgresql()

    chain(
        load_wine_data_postgresql_obj,
        transform_wine_data_from_postgresql_obj,
        chart_loaded_wine(transform_wine_data_from_postgresql_obj)
    )

#Instantiate the DAG
etl_elt_wine()
