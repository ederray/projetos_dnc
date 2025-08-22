"""Funções de tratamento dos dados"""
from IPython.display import display
from ipywidgets import interact, HTML, Output, Dropdown, VBox
import logging
import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame, Series
import sidetable
import missingno as msno
import pandas as pd
from sklearn.preprocessing import StandardScaler, PowerTransformer

# instância do objeto logger
logger = logging.getLogger(__name__)

def amostra_dados(df: DataFrame) -> DataFrame:
    """Função para retornar a amostragem dos dados"""
    return df.sample(3)

def contagem_valores(coluna:Series) -> None: 
    """Função que realiza a contagem de valores por coluna"""
    return coluna.value_counts()

def verificacao_nulos(df:DataFrame) -> Series:
    """Função que realiza a contagem de valores nulos por feature do dataset"""
    output = df.isna().sum()
    return output

def filtrar_linhas_valores_nulos(df:DataFrame) -> pd.DataFrame:
    """Função que aplica o filtro de valores nulos no dataframe e retorna um dataframe filtrado com a correspondência."""
    output = df[df.isna().any(axis=1)]
    logger.info(f"Contagem de linhas nulas para o dataframe:{output.shape[0]}")
    return output



def frequencia_valores_nulos(df:DataFrame) -> pd.DataFrame:
    """Função que gera uma matriz esparsa com a visualização dos valores nulos intercalado com valores preenchidos por coluna"""
    return df.stb.missing()

def verificar_linhas_duplicadas(df:pd.DataFrame)->pd.DataFrame:

    """Função que retorna um dataframe contendo as linhas duplicadas do dataset inputado."""
    output = \
    (
    df
    .groupby(df.columns.tolist(), dropna=False)
    .size()
    .to_frame('n_duplicates')
    .query('n_duplicates>1')
    .sort_values('n_duplicates', ascending=False)
    .head(5)
    )
    return output

def remover_duplicados(df: DataFrame, coluna: str) -> DataFrame:
    """Função para remoção de valores duplicados."""
    df.drop_duplicates(subset=[coluna], keep='first', inplace=True)
    return df

def filtragem_iterativa_valores_catogoricos(df: DataFrame, coluna: str) -> DataFrame:
    """Função que aplica um filtro iterativo para selecionar os dados do dataset a partir dos valores da coluna selecionada."""
    
    lista = sorted(df[coluna].unique())
    @interact(valor = lista)
    def gerar_dataframe(valor):
        filtro = df.query(f"{coluna}=='{valor}'")

        return filtro

def filtrar_feature_valor_categorico(df: DataFrame, query:str) -> DataFrame:
    """Função que aplica um filtro em uma variável categorica ou em um conjunto delas através do método df.query"""
    try:
        output = df.query(query)
    except Exception as e:
        logger.error(e)
    return output

def imputar_dados_room_type_entire_home_apt(df: DataFrame):
    """Função para transformar e tratar os valores das colunas bathrooms, bedrooms e 
    beds relacioanados ao filtro da coluna room_type=='Entire home/apt'."""

    # filtra o dataset a partir dos valores da coluna room_type == 'Entire home/apt'
    df_filtrado = filtrar_feature_valor_categorico(df, query="room_type=='Entire home/apt'")

    # 'Entire home/apt' exige a presença de 1 banheiro na residência por legislação.
    df_filtrado.loc[df_filtrado['bathrooms']<1,'bathrooms'] = 1

    # 'Entire home/apt' com bedrooms e beds menor que 1 provavelmente corresponde a um tipo de acomodação kitnet ou studio.
    df_filtrado.loc[(df_filtrado['bedrooms'] < 1) | (df_filtrado['beds'] < 1),['bedrooms','beds']] = 0

    # quantidade de banheiros e quartos vazios preenchidos com a moda de ocorrência dos valores
    df_filtrado.loc[df_filtrado['bathrooms'].isna(),'bathrooms'] = df_filtrado['bathrooms'].mode()[0]
    df_filtrado.loc[df_filtrado['bedrooms'].isna(),'bedrooms'] = df_filtrado['bedrooms'].mode()[0]

    # quatidade de camas definidas a partir de uma taxa de acomodações/2
    df_filtrado.loc[df_filtrado['beds'].isna(),'beds'] = np.ceil(df_filtrado['accommodates'] / 2)

    return df_filtrado


def imputar_dados_room_type_private_room(df: DataFrame):
    """Função para transformar e tratar os valores das colunas bathrooms, bedrooms e 
    beds relacioanados ao filtro da coluna room_type=='Private room'."""

    # filtra o dataset a partir dos valores da coluna room_type == 'Private room'
    df_filtrado = filtrar_feature_valor_categorico(df, query="room_type=='Private room'")

    # realiza o tratamento de valores a partir das regras definindas:
    # 'Private room' exige a presença de 1 quarto exclusivo
    df_filtrado.loc[df_filtrado['bedrooms'].isna(),'bedrooms'] = 1

    # quatidade de camas definidas a partir de uma taxa de acomodações/2
    df_filtrado.loc[df_filtrado['beds'].isna(),'beds'] = np.ceil(df_filtrado['accommodates'] / 2)

    # quantidade de banheiros vazios preenchidos com a moda.
    df_filtrado.loc[df_filtrado['bathrooms'].isna(),'bathrooms'] = df_filtrado['bathrooms'].mode()[0]


    return df_filtrado

def imputar_dados_room_type_shared_room(df: DataFrame):
    """Função para transformar e tratar os valores das colunas bathrooms, bedrooms e 
    beds relacioanados ao filtro da coluna room_type=='Shared room'."""

    # filtra o dataset a partir dos valores da coluna room_type == 'Shared room'
    df_filtrado = filtrar_feature_valor_categorico(df, query="room_type=='Shared room'")

    # 'Shared room' não exige a presença de 1 quarto ou banheiro exclusivos.
    df_filtrado.loc[df_filtrado['bedrooms'].isna(),['bedrooms','bathrooms','beds']] = 0

    return df_filtrado

def imputar_dados_room_type_hotel_room(df: DataFrame):
    """Função para transformar e tratar os valores das colunas bathrooms, bedrooms e 
    beds relacioanados ao filtro da coluna room_type=='Hotel room'."""

    # filtra o dataset a partir dos valores da coluna room_type == 'Hotel room'
    df_filtrado = filtrar_feature_valor_categorico(df, query="room_type=='Hotel room'")

    # quantidade de quartos preenchidos com a moda
    df_filtrado.loc[df_filtrado['bedrooms'].isna(),'bedrooms'] = df_filtrado['bedrooms'].mode()[0]

    # quantidade de banheiros vazios preenchidos com a moda.
    df_filtrado.loc[df_filtrado['bathrooms'].isna(),'bathrooms'] = df_filtrado['bathrooms'].mode()[0]

    # quantidade de banheiros menor que 1 preenchidos com valor 1, já que quarto de hotel tem banheiro.
    df_filtrado.loc[df_filtrado['bathrooms']<1,'bathrooms'] = 1

    # quatidade de camas definidas a partir de uma taxa de acomodações/2
    df_filtrado.loc[df_filtrado['beds'].isna(),'beds'] = np.ceil(df_filtrado['accommodates'] / 2)

    return df_filtrado

def imputar_dados_price(df: DataFrame):
    """Função para transformar e tratar os valores da coluna price com a média por tipo de acomodação 
    em cada bairro ou com a media do tipo de acomodação."""

    # cópia do dataset original
    df_copia = df.copy() # Criamos uma 

    try:
        # Imputação de valores vazios por tipo de quarto em cada bairro
        df_copia['price'] = df_copia.groupby(['room_type', 'neighbourhood_cleansed'])['price'].transform(
            lambda x: x.fillna(x.mean())
        )

        # Imputação de valores vazios restantes por tipo de quarto.
        df_copia['price'] = df_copia.groupby('room_type')['price'].transform(
            lambda x: x.fillna(x.mean())
        )

    except Exception as e:
        print(f"Ocorreu um erro durante a imputação de preços: {e}")
        # Retorna o DataFrame original caso ocorra um erro
        return df

    return df_copia


def selecao_colunas(df: DataFrame, colunas: list) -> DataFrame:
    """Função que seleciona as colunas para montagem do dataset"""
    return df[colunas]


def agrupar_dados(df: pd.DataFrame, cols_agrup: list, cols_filter: list=None, agr=None) -> pd.DataFrame:
    """Função que agrupa as colunas para montagem do dataset."""
    try:
        if not cols_filter:
            logger.info(f'Agrupamento selecionado: {cols_agrup}, método: {agr}')
            df = df.groupby(by=cols_agrup).agg(agr)
        else:
            logger.info(f'Agrupamento selecionado: {cols_agrup}, filtragem dataset:{cols_filter}, método: {agr}')
            df = df.groupby(by=cols_agrup)[cols_filter].agg(agr)

    except Exception as e:
        logger.error(e)

    return df


