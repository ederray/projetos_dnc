"""Arquivo de funções e classes de pré-processamento dos dados"""

import logging
from typing import Optional

import numpy as np
from pandas import DataFrame
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, PowerTransformer

# instância do objeto logger
logger = logging.getLogger(__name__)


def power_transform(
    df: DataFrame, cat_col: str = None, metodo: str = "yeo-johnson", cols: Optional[list] = None
) -> DataFrame:
    """
    Aplica PowerTransformer (Box-Cox ou Yeo-Johnson) às colunas numéricas,
    opcionalmente agrupando os dados por uma coluna categórica.

    params:
        df (pd.DataFrame): DataFrame de entrada.
        cat_col (str): Nome da coluna categórica usada para agrupar.
        metodo (str, optional): Método do PowerTransformer ('yeo-johnson' ou 'box-cox').
        cols (list, optional): Lista de colunas numéricas a transformar.
                               Se None, aplica em todas as numéricas.

    returns:
        pd.DataFrame: DataFrame com as colunas numéricas transformadas por grupo.
    """
    df = df.copy()

    # Determina as colunas numéricas se não foram passadas
    if cols is None:
        cols = df.select_dtypes(include="number").columns.tolist()

    def _transform(group: DataFrame) -> DataFrame:
        try:

            transformer = PowerTransformer(method=metodo, standardize=True)
            if cols is None:
                group = transformer.fit_transform(group)
            else:
                group[cols] = transformer.fit_transform(group[cols])
                return group

        except Exception as e:
            logger.error(f"Erro ao transformar grupo: {e}")
            return group

    try:
        if cat_col and cat_col in df.columns:
            transformed = df.groupby(cat_col, group_keys=False).apply(_transform)
            return transformed.reset_index(drop=True)
        else:
            return _transform(df)

    except Exception as e:
        logger.error(f"Erro geral no power_transform: {e}")
        return df


class OutlierDetector(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=1.5):
        self.threshold = threshold
        self.median_values = None

    def fit(self, X, y=None):
        X = X.astype(float)
        q1 = np.percentile(X, 25, axis=0)
        q3 = np.percentile(X, 75, axis=0)

        iqr = q3 - q1

        self.lower_bound = q1 - self.threshold * iqr
        self.upper_bound = q3 + self.threshold * iqr

        self.median_values = np.median(X, axis=0)

        return self

    def transform(self, X):
        X = X.astype(float)  # Garante que os dados são float
        X_outliers_removed = np.where(
            (X < self.lower_bound) | (X > self.upper_bound), self.median_values, X
        )
        return X_outliers_removed

    def get_feature_names_out(self, input_features=None):
        return input_features


def one_hot_encoding(dados_ajuste: DataFrame) -> DataFrame:
    encoder = OneHotEncoder()
    dados_encoded = encoder.fit_transform(dados_ajuste)
    return DataFrame(dados_encoded.toarray(), columns=encoder.get_feature_names_out())
