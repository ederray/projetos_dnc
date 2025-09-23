"""Funções utilitárias de manipulação dos dados"""

import logging
from typing import Any

from ipywidgets import interact
import numpy as np
from pandas import DataFrame

# instância do objeto logger
logger = logging.getLogger(__name__)


def filtragem_interativa_valores_categoricos(df: DataFrame, coluna: str) -> DataFrame:
    """
    Função que aplica um filtro iterativo para selecionar os dados do dataset a partir dos valores da coluna selecionada.

    params:

        df (DataFrame): DataFrame de entrada.
        coluna (str): coluna com categorias para filtragem interativa.

    returns:

        DataFrame: DataFrame com filtragem interativa.

    """

    lista = sorted(df[coluna].unique())

    @interact(valor=lista)
    def gerar_dataframe(valor):
        filtro = df.query(f"{coluna}=='{valor}'")
        return filtro


def filtrar_dataset(df: DataFrame, query: str) -> DataFrame:
    """
    Função que aplica um filtro em uma variável categorica ou em um conjunto delas através do método df.query

    params:

        df (DataFrame): DataFrame de entrada.
        query (str): query de filtragem dos dados do dataset.

    returns:

        DataFrame: DataFrame filtrado pela condição.

    """
    output = None
    try:
        output = df.query(query)
        return output

    except Exception as e:
        logger.error(e)
        raise


def imputar_dados_room_type_entire_home_apt(df: DataFrame) -> DataFrame:
    """
    Função para transformar e tratar os valores das colunas bathrooms, bedrooms e
    beds relacionados ao filtro da coluna room_type=='Entire home/apt'.

    params:

        df (DataFrame): DataFrame de entrada.

    returns:

        DataFrame: DataFrame filtrado pela condição.

    """
    df_filtrado = filtrar_dataset(df, query="room_type=='Entire home/apt'")

    # regras de substituição de valores
    df_filtrado.loc[df_filtrado["bathrooms"] < 1, "bathrooms"] = 1
    df_filtrado.loc[(df_filtrado["bedrooms"] < 1), "bedrooms"] = 0
    df_filtrado.loc[(df_filtrado["beds"] < 1), "beds"] = 0

    # quatidade de camas definidas a partir de uma taxa de acomodações/2
    df_filtrado.loc[df_filtrado["beds"].isna(), "beds"] = np.ceil(df_filtrado["accommodates"] / 2)

    return df_filtrado


def imputar_dados_room_type_private_room(df: DataFrame) -> DataFrame:
    """
    Função para transformar e tratar os valores das colunas bathrooms, bedrooms e
    beds relacionados ao filtro da coluna room_type=='Private room'.

    params:

        df (DataFrame): DataFrame de entrada.

    returns:

        DataFrame: DataFrame filtrado pela condição.

    """

    df_filtrado = filtrar_dataset(df, query="room_type=='Private room'")

    # regras de substituição de valores
    df_filtrado.loc[(df_filtrado["bedrooms"].isna()) | (df["bedrooms"] == 0), "bedrooms"] = 1

    # quatidade de camas definidas a partir de uma taxa de acomodações/2
    df_filtrado.loc[(df_filtrado["beds"].isna()) | (df["beds"] == 0), "beds"] = np.ceil(
        df_filtrado["accommodates"] / 2
    )

    return df_filtrado


def imputar_dados_room_type_shared_room(df: DataFrame) -> DataFrame:
    """
    Função para transformar e tratar os valores das colunas bathrooms, bedrooms e
    beds relacionados ao filtro da coluna room_type=='Shared room'.

    params:

        df (DataFrame): DataFrame de entrada.

    returns:

        DataFrame: DataFrame filtrado pela condição.

    """
    df_filtrado = filtrar_dataset(df, query="room_type=='Shared room'")

    # regras de substituição de valores
    df_filtrado.loc[df_filtrado["bedrooms"].isna(), ["bedrooms"]] = 0
    df_filtrado.loc[df_filtrado["bathrooms"].isna(), ["bathrooms"]] = 0
    df_filtrado.loc[df_filtrado["beds"].isna(), ["beds"]] = 0

    return df_filtrado


def imputar_dados_room_type_hotel_room(df: DataFrame) -> DataFrame:
    """
    Função para transformar e tratar os valores das colunas bathrooms, bedrooms e
    beds relacionados ao filtro da coluna room_type=='Hotel room'.

    params:

        df (DataFrame): DataFrame de entrada.

    returns:

        DataFrame: DataFrame filtrado pela condição.

    """

    df_filtrado = filtrar_dataset(df, query="room_type=='Hotel room'")

    # regras de substituição de valores
    df_filtrado.loc[df_filtrado["bathrooms"] < 1, "bathrooms"] = 1
    df_filtrado.loc[df_filtrado["bedrooms"] < 1, "bedrooms"] = 1

    # quatidade de camas definidas a partir de uma taxa de acomodações/2
    df_filtrado.loc[df_filtrado["beds"].isna(), "beds"] = np.ceil(df_filtrado["accommodates"] / 2)

    return df_filtrado


def imputar_dados_price(df: DataFrame):
    """
    Função para transformar e tratar os valores da coluna price com a média por tipo de acomodação
    em cada bairro ou com a media do tipo de acomodação.

    params:

        df (DataFrame): DataFrame de entrada.

    returns:

        DataFrame: DataFrame filtrado pela condição.

    """

    df_copia = df.copy()

    try:
        # Imputação de valores vazios por tipo de quarto em cada bairro
        df_copia["price"] = df_copia.groupby(["room_type", "neighbourhood_cleansed"])[
            "price"
        ].transform(lambda x: x.fillna(x.mean()))

        df_copia["price"] = df_copia.groupby("room_type")["price"].transform(
            lambda x: x.fillna(x.mean())
        )

    except Exception as e:
        logger.error(e)
        raise

    return df_copia


def substituir_valores(
    df: DataFrame, filtro_linhas: list, filtro_colunas: list, valor: Any
) -> DataFrame:
    """
    Função que substitui os valores a partir dos filtros de linha ou coluna informados para o valor determinado.

    params:

        df (DataFrame): DataFrame de entrada.
        filtro_linhas (list): lista de valores de index para filtragem.
        filtro_colunas (list): lista de valores de index para filtragem.
        valor

    returns:

        DataFrame: DataFrame filtrado pela condição.

    """
    df.loc[filtro_linhas, filtro_colunas] = valor
    return df


def selecao_colunas(df: DataFrame, colunas: list) -> DataFrame:
    """
    Função que seleciona as colunas para montagem do dataset.

    params:

        df (DataFrame): DataFrame de entrada.
        colunas (list): lista de colunas para filtrar o dataset.

    returns:

        DataFrame: DataFrame filtrado com as colunas selecionadas.

    """
    return df[colunas]


def agrupar_dados(
    df: DataFrame, cols_agrup: list, cols_filter: list = None, agr=None
) -> DataFrame:
    """
    Função que agrupa as colunas e resume os valores por algum critério de agregação definido.

    params:

        df (DataFrame): DataFrame de entrada.
        cols_agrup (list): lista de colunas para agrupar o dataset.
        cols_filter (list): lista de colunas para filtrar o dataset.
        arg (list): função ou critério de agregação. Ex: np.sum(), 'sum', 'count'.

    returns:

        DataFrame: DataFrame filtrado com as colunas selecionadas.

    """
    try:
        if not cols_filter:
            logger.info(f"Agrupamento selecionado: {cols_agrup}, método: {agr}")
            df = df.groupby(by=cols_agrup).agg(agr)
        else:
            logger.info(
                f"Agrupamento selecionado: {cols_agrup}, filtragem dataset:{cols_filter}, método: {agr}"
            )
            df = df.groupby(by=cols_agrup)[cols_filter].agg(agr)

    except Exception as e:
        logger.error(e)

    return df
