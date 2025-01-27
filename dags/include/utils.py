from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import r2_score
import numpy as np
import pandas as pd
import os
import logging
import itertools
import duckdb
import matplotlib.pyplot as plt
from pathlib import Path
import mlflow


logger = logging.getLogger("airflow.task")

def get_wine_data(samples_frac: float=1.0, path: str=None) -> pd.DataFrame:
    """
    Retrieve data about wine.
    Args:
        samples_frac (float): Fraction of samples to return.
            Default is 1.0, select all the 4898 smaples of
            wine.
        path (str): Source path.
    Returns:
        A pd.DataFrame about wine.
    """
    wine_df = pd.read_csv(path, sep=";")
    wine_df.rename(lambda x: x.replace(" ", "_"), axis="columns", inplace=True)
    wine_df["id"] = wine_df.index + 1
    wine_df = wine_df.sample(frac=samples_frac, replace=False, random_state=1)

    return wine_df


def extract_split_data(
    duckdb_instance_name: str=None,
    table_name: str=None
) -> tuple:
    """
    Extract data from source, and split it into train and test datasets.
    Args:
        duckdb_instance_name (str): The name of the DuckDB instance.
        table_name (str): The name of the table to retrieve data.
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    with duckdb.connect(duckdb_instance_name) as con:
        df = con.sql(
            """
            SELECT \
                alcohol,\
                chlorides,\
                citric_acid,\
                density,\
                fixed_acidity,\
                free_sulfur_dioxide,\
                pH,\
                residual_sugar,\
                sulphates,\
                total_sulfur_dioxide,\
                volatile_acidity \
            FROM wine_data\
            """).df()

    X_data, y_data = df.drop("alcohol", axis=1), df["alcohol"]
    X_train, X_test, y_train, y_test = train_test_split(
        X_data, y_data, test_size=0.30, random_state=42
    )

    X_train = pd.DataFrame(X_train)
    X_test = pd.DataFrame(X_test)
    y_train = pd.DataFrame(y_train)
    y_test = pd.DataFrame(y_test)

    return (X_train, X_test, y_train, y_test)


def build_model() -> list[dict]:
    """
    Build a list of dict. Each dict contains model name,
    pipeline and parameters. For example, a LinearRegression
    pipeline's dict looks like
    {
    "model_name": "lr",
    "model_pipe": lr_pipe,
    "model_params": lr_params #transformed params based on lr_param_grid
    }
    """
    steps = [
        ("standard_scaler", StandardScaler()),
        ("polynomial", PolynomialFeatures(degree=2)),
        ("LinearRegression", LinearRegression())
    ]
    lr_pipe = Pipeline(steps)
    lr_param_grid = {"polynomial__degree": [2, 3]}

    steps = [
        ("standard_scaler", StandardScaler()),
        ("polynomial", PolynomialFeatures(degree=2)),
        ("HGBRegression",  HistGradientBoostingRegressor())
    ]
    HGBRegressor_pipe = Pipeline(steps)
    HGBRegressor_param_grid = {
        "HGBRegression__learning_rate": [0.1, 0.05],
        "HGBRegression__min_samples_leaf": [20, 25],
        "polynomial__degree":[2]
    }

    steps = [
        ("standard_scaler", StandardScaler()),
        ("polynomial", PolynomialFeatures(degree=2)),
        ("BayesianRidge", BayesianRidge(tol=1e-6, compute_score=True))
    ]
    BayesianRidge_pipe = Pipeline(steps)
    BayesianRidge_param_grid = {}

    model_name_list = ["lr","HGBR","Bayesian"]
    model_pipe_list = [lr_pipe, HGBRegressor_pipe, BayesianRidge_pipe]
    model_params_list = [lr_param_grid, HGBRegressor_param_grid, BayesianRidge_param_grid]
    models = [{"model_name": i, "model_pipe": j, "model_params": k} for i,j,k in zip(model_name_list, model_pipe_list, model_params_list)]

    model_list = []
    for i in models:
        model_name = str(i["model_name"])
        param_grid = list(ParameterGrid(i["model_params"]))
        model_list += [{"model_name": model_name,"model_pipe": model_pipe, "model_params": model_params} for model_pipe, model_params in itertools.zip_longest((), param_grid, fillvalue=i["model_pipe"])]

    return model_list


def choose_model(model_list: list[dict]) -> list[int]:
    """
    Randomly select index of given list without replace.
    At least select one.
    """
    ids = np.arange(0, np.size(model_list))
    size = np.random.choice(ids) + 1
    ret = np.random.choice(ids, size=size, replace=False).tolist()

    return ret


def train_model(
    data: tuple,
    model_index: int,
    mlflow_tracking_uri: str,
    artifact_location: str,
    experiment_name: str
) -> dict:
    """
    Train ML model with MLflow.
    Args:
        data (tuple): X_train, X_test, y_train, y_test.
        model_index (int): Index of selected model from
            predefined model list.
        mlflow_tracking_uri (str): URI of MLflow tracking server.
        artifact_location (str): The location to store run artifacts.
        experiment_name (str): Name of experiment.
    Returns:
        A dict including model info. For example, a
        LinearRegression pipeline's result looks like
        {
        "model_name": "lr",
        "model_pipe": str(lr_pipe),
        "model_params": lr_params,
        "r2_train": r2_train,
        "r2_test": r2_test,
        "experiment_id": experiment_id,
        "experiment_name": experiment_name,
        "artifact_location": artifact_location,
        "model_uri": model_uri,
        "experiment_url": experiment_url,
        "run_url": run_url
        }
    """
    # Set mlflow_tracking_uri
    mlflow.set_tracking_uri(uri=mlflow_tracking_uri)

    if not os.path.exists(artifact_location):
        os.mkdir(artifact_location)

    # Create and/or set experiment
    client = mlflow.MlflowClient()
    filter_string=f"name = '{experiment_name}'"
    experiments = client.search_experiments(filter_string=filter_string)
    if len(experiments) == 0:
        experiment_id = client.create_experiment(
            experiment_name,
            artifact_location=Path(artifact_location).as_uri(),
            tags={"version": "v1", "priority": "P1"},
        )
        experiment = client.get_experiment(experiment_id)
        mlflow.set_experiment(experiment.name)
    else:
        experiment = client.get_experiment_by_name(experiment_name)
        mlflow.set_experiment(experiment.name)

    # Start an MLflow run
    with mlflow.start_run() as run:
        mlflow.autolog()
        logger.info("Extract train and test data.")
        X_train, X_test, y_train, y_test = data
        y_train = y_train.to_numpy().ravel()
        y_test = y_test.to_numpy().ravel()

        logger.info("Extract model info.")
        model_list = build_model()
        model_dict = model_list[model_index]

        model_name = model_dict["model_name"]
        logger.info(f"Model name: {model_name}")
        model_pipe = model_dict["model_pipe"]
        logger.info(f"Model pipeline: {model_pipe}")
        model_params = model_dict["model_params"]
        logger.info(f"Model parameters: {model_params}")

        model_pipe.set_params(**model_params)
        model_pipe.fit(X_train, y_train)

        y_pred_train = model_pipe.predict(X_train)
        y_pred_test = model_pipe.predict(X_test)

        r2_train = r2_score(y_train, y_pred_train)
        r2_test = r2_score(y_test, y_pred_test)

        logger.info(f"r2 train: {r2_train:.6f}")
        model_dict["r2_train"] = np.round(r2_train, 6)
        logger.info(f"r2 test: {r2_test:.6f}")
        model_dict["r2_test"] = np.round(r2_test, 6)

        # Set tags
        mlflow.set_tag("Training Info", f"Basic {model_name} model for wine data.")
        client.set_tag(run.info.run_id, "wine_tag", f"Predict alcohol index with {model_name}")

        # Register model
        model_uri = f"runs:/{run.info.run_id}/model"
        mlflow.register_model(model_uri, f"wine-{model_name}")

        experiment_url = f"{mlflow_tracking_uri}/#/experiments/{run.info.experiment_id}"
        model_dict["experiment_id"] = experiment.experiment_id
        model_dict["experiment_name"] = experiment.name
        model_dict["artifact_location"] = experiment.artifact_location
        model_dict["model_uri"] = model_uri
        model_dict["experiment_url"] = experiment_url
        model_dict["run_url"] = f"{experiment_url}/runs/{run.info.run_id}"
        model_dict["model_pipe"] = str(model_dict["model_pipe"])

    return model_dict


def plot_kde(df, path, title="Alcohol KDE"):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.set_facecolor("seashell")
    fig.set_facecolor("seashell")

    # Custom colors
    colors = ["#DD3D2D","#F67E4B","#FDB366","#FEDA8B",
              "#EAECCC","#C2E4EF","#98CAE1","#6EA6CD"]
    selected_classes = df.quality.sort_values(ascending=False).unique()
    classes_size = selected_classes.shape[0]

    if classes_size >= 5:
        pass
    elif classes_size >= 4:
        colors = colors[::2]
    else:
        colors = colors[::3]

    # Draw KDE of each class.
    for i,j in enumerate(selected_classes):
        df[["quality","alcohol"]][df.quality == j]["alcohol"]\
        .plot.kde(color=colors[i % np.minimum(8, classes_size)])

    # Remove spines.
    ax.spines[["top","right"]].set_visible(False)

    # Add tick lines across the plot.
    ax.grid(True, ls="--", lw=.5, c="k", alpha=.2)

    # Add xlabel, title, legend and note.
    ax.set(xlabel="alcohol(%)")
    ax.set_title(title, weight="bold", size=15)
    ax.legend(
        selected_classes,
        title="Wine Class",
        loc="best",
        frameon=False,
        facecolor="seashell",
        title_fontproperties=dict(size="large",weight="bold")
    )
    ax.text(
        6, 0.35,
        f"Total {df.shape[0]} samples",
        color="#4C4D58",
        fontsize=10,
        style="italic",
        weight="bold",
        va="bottom",
        ha="left"
    )

    # Save figure as a pdf.
    if not os.path.exists(path):
        os.makedirs(path)

    file_name = f"{title.lower().replace(' ', '_')}.pdf"
    file_path = os.path.join(path, file_name)

    fig.savefig(
        file_path,
        bbox_inches="tight",
        dpi=300,
        pad_inches=0.1
    )
