import pandas as pd
from catboost import CatBoostClassifier

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

import optuna

from giskard import Dataset, Model, scan, testing
import pickle

from minio import Minio
from minio.error import S3Error

# import clickhouse_connect
import psycopg2

from functools import reduce

import clickhouse_connect

from dotenv import load_dotenv
import os

load_dotenv()

SPARK_COMPAT_VERSION = os.getenv('SPARK_COMPAT_VERSION')
SCALA_COMPAT_VERSION = os.getenv('SCALA_COMPAT_VERSION')
CATBOOST_SPARK_VERSION = os.getenv('CATBOOST_SPARK_VERSION')
DB_HOST = os.getenv('DB_HOST')
POSTGRESQL_PORT = os.getenv('POSTGRESQL_PORT')
CLICKHOUSE_PORT = os.getenv('CLICKHOUSE_PORT')
DB_USER = os.getenv('DB_USER')
DB_PASSWORD = os.getenv('DB_PASSWORD')
DB_NAME = os.getenv('DB_NAME')

ACCESS_KEY=os.getenv('MINIO_ACCESS_KEY')
SECRET_KEY=os.getenv('MINIO_SECRET_KEY')

COLUMN_TYPES = {
    'age': 'category',
    'sex': 'category',
    'job': 'category',
    'housing': 'category',
    'credit_amount': 'numeric',
    'duration': 'numeric'
}

TARGET_COLUMN_NAME = 'default'
FEATURE_COLUMNS = [i for i in COLUMN_TYPES.keys()]
FEATURE_TYPES = {i: COLUMN_TYPES[i] for i in COLUMN_TYPES if i != TARGET_COLUMN_NAME}

COLUMNS_TO_SCALE = [key for key in COLUMN_TYPES.keys() if COLUMN_TYPES[key] == "numeric"]
COLUMNS_TO_ENCODE = [key for key in COLUMN_TYPES.keys() if COLUMN_TYPES[key] == "category"]


# get feature columns
conn = psycopg2.connect(dbname='postgres',
                                user=DB_USER,
                                password=DB_PASSWORD,
                                host='localhost',
                                port=POSTGRESQL_PORT)
cur = conn.cursor()
cur.execute(f"SELECT {reduce(lambda a,b: a + ', ' + b, FEATURE_COLUMNS)} FROM credit;")
X = pd.DataFrame(cur.fetchall(), columns=FEATURE_COLUMNS)
conn.commit()
conn.close()

# get targer column
conn = psycopg2.connect(dbname='postgres',
                                user=DB_USER,
                                password=DB_PASSWORD,
                                host='localhost',
                                port=POSTGRESQL_PORT)
cur = conn.cursor()
cur.execute(f'SELECT cr."{TARGET_COLUMN_NAME}" FROM credit cr;')
y = [x[0] for x in cur.fetchall()]
conn.commit()
conn.close()

# Train test split ----------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size = 0.25,
    random_state=42
)

# Pipeline --------------------------------------------
numeric_transformer = Pipeline(steps = [
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
categorical_transformer = Pipeline(steps = [
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, COLUMNS_TO_SCALE),
        ("cat", categorical_transformer, COLUMNS_TO_ENCODE)
    ]
)

# OPTIMIZE HYPERPARAMETERS ----------------------------------------------------------
def objective(trial):    
    # CatBoostClassifier hyperparams
    param = {
        "objective": trial.suggest_categorical("objective", ["Logloss", "CrossEntropy"]),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.01, 0.1),
        "depth": trial.suggest_int("depth", 1, 12),
        "boosting_type": trial.suggest_categorical("boosting_type", ["Ordered", "Plain"]),
        "bootstrap_type": trial.suggest_categorical(
            "bootstrap_type", ["Bayesian", "Bernoulli", "MVS"]
        ),
        "used_ram_limit": "3gb",
    }

    # model
    estimator = CatBoostClassifier(**param, verbose=False)
    
    # pipelines
    clf_pipeline = Pipeline(steps = [
    ("preprocessor", preprocessor),
    ("classifier", estimator)
])
    # cross-validation accuracy counting
    accuracy = cross_val_score(clf_pipeline, X, y, cv=10, scoring= 'accuracy').mean()
    return accuracy

#study = optuna.create_study(direction="maximize", study_name="CBC-2023-01-14-14-30", storage='sqlite:///db/CBC-2023-01-14-14-30.db')
# study = optuna.create_study(direction="maximize", study_name=f"{datetime.now().strftime('%Y%m%d-%H%M%S')}")
study = optuna.create_study(direction="maximize", study_name="credit_classifier")

# Hyperparams searching
study.optimize(objective, n_trials=10)

# best result is
params = study.best_params

# MODEL WITH BEST PARAMS ##########################################33
catboostclassifier = Pipeline(steps = [
    ("preprocessor", preprocessor),
    ("classifier", CatBoostClassifier(**params, verbose=False))
])
catboostclassifier.fit(X_train, y_train)

# save model.pkl
with open('model.pkl', 'wb') as f:
    pickle.dump(catboostclassifier, f)

# minio save model.pkl
def s3_upload_model():
    # Create a client with the MinIO server playground, its access key
    # and secret key.
    client = Minio("localhost:9099",
        access_key=ACCESS_KEY,
        secret_key=SECRET_KEY,
        secure=False
    )

    # The file to upload, change this path if needed
    source_file = "model.pkl"

    # The destination bucket and filename on the MinIO server
    bucket_name = "credit-model"
    destination_file = "model.pkl"

    # Make the bucket if it doesn't exist.
    found = client.bucket_exists(bucket_name)
    if not found:
        client.make_bucket(bucket_name)
        print("Created bucket", bucket_name)
    else:
        print("Bucket", bucket_name, "already exists")

    # Upload the file, renaming it in the process
    client.fput_object(
        bucket_name, destination_file, source_file,
    )
    print(
        source_file, "successfully uploaded as object",
        destination_file, "to bucket", bucket_name,
    )

try:
    s3_upload_model()
except S3Error as exc:
    print("error occurred.", exc)