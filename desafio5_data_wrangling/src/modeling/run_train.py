"""Arquivo para treinamento de modelos"""

# %% carregamento das bibliotecas
import logging
import os
import sys

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PowerTransformer

# diretório raiz
sys.path.append(os.path.abspath(os.path.join("..")))
from config.logging_config import setup_logging
from features.transformers import OutlierDetector

# %% carregamento do dataset
caminho_raiz = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
caminho_dados_processados = os.path.join(
    caminho_raiz, "data", "processed", "tbl_anuncios_processed.csv"
)
data = pd.read_csv(caminho_dados_processados, index_col="Unnamed: 0", sep=",")

# %% configurações de logging
setup_logging()

# %% construção do ColumnTransformer com pipeline para sequência de transformações
num_cols = [
    "accommodates",
    "bathrooms",
    "bedrooms",
    "beds",
    "number_of_reviews",
    "review_scores_rating",
]
cat_cols = ["room_type", "Zona"]

num_pipeline = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="mean")),
        ("outlier", OutlierDetector()),
        ("power_standard", PowerTransformer(method="yeo-johnson", standardize=True)),
    ]
)


cat_pipeline = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ]
)

# %% contrução do ColumnTransformer
preprocessing = ColumnTransformer(
    [
        ("numeric_cols", num_pipeline, num_cols),
        ("categorical_cols", cat_pipeline, cat_cols),
    ],
    remainder="passthrough",
)

# %% Lógica de Execução Principal
if __name__ == "__main__":

    # instância do objeto logger
    logger = logging.getLogger(__name__)

    # %%carregamento dos dados

    data = pd.read_csv(caminho_dados_processados, index_col="Unnamed: 0", sep=",")

    logger.info(f"dados carregados com sucesso:{data.shape}")

    # %% separação dos dados em treino e teste
    X = data.drop(["latitude", "longitude", "price"], axis=1)
    y = data["price"]

    X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.2, random_state=42)

    logger.info(
        f"X_treino:{X_treino.shape}, X_teste:{X_teste.shape}, y_treino:{y_treino.shape}, y_teste:{y_teste.shape}"
    )

    # %% ajuste e transformação dos dados
    X_treino_transformed_array = preprocessing.fit_transform(X_treino)
    X_teste_transformed_array = preprocessing.transform(X_teste)

    X_train_transformed = pd.DataFrame(
        X_treino_transformed_array, columns=preprocessing.get_feature_names_out()
    )
    X_teste_transformed = pd.DataFrame(
        X_teste_transformed_array, columns=preprocessing.get_feature_names_out()
    )

    # %% visualização do dataset de treinamento processado
    print("\n--- Dados de Treino Transformados ---")
    logger.info(f"X_train_transformed:{X_train_transformed.shape}")
    print(X_train_transformed.head())

    # %% visualização do dataset de teste processado sem data leakage
    print("\n--- Dados de Teste Transformados ---")
    logger.info(f"X_teste_transformed:{X_teste_transformed.shape}")
    print(X_teste_transformed.shape)
    print(X_teste_transformed.head())
