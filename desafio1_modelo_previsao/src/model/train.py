"""Funções de construção do modelo"""
from pandas import DataFrame
from sklearn.preprocessing import FunctionTransformer, PowerTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyRegressor
from statsmodels.tsa.arima.model import ARIMA
from src.data.preprocessing import media_movel, lag_data


def gerar_features(df: DataFrame) -> DataFrame:
    """Função que constroi as features temporais"""
    df = media_movel(df, coluna='Preço', p=[2, 3, 5, 7])
    df = lag_data(df, coluna='Preço', lag=[2, 3, 5, 7])

    return DataFrame


def criar_pipeline() -> Pipeline:
    """Função que cria o pipeline do modelo"""
    pipeline = Pipeline(
        [('feat_engineering', FunctionTransformer(gerar_features, validate=True)),
         ('impute', SimpleImputer(strategy='constant', fill_value=0)),
         ('power', PowerTransformer(method='yeo-johnson')),
         ('model', DummyRegressor(strategy='mean'))
         ]
    )
    return pipeline


def treinar_modelo():
    """Função para treino do modelo"""

    return


def previsao_valores():
    """Função para previsão de valores do modelo"""

    return


def metricas_validacao():
    """Função para gerar métricas de avaliação do modelo."""

    return
