"""Funções de tratamento dos dados"""
import logging
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame, Series
import pandas as pd
from ipywidgets import interact, HTML, Output, Dropdown, VBox
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

def filtrar_linhas_valores_nulos(df:DataFrame) -> DataFrame:
    """Função que aplica o filtro de valores nulos no dataframe e retorna um dataframe filtrado com a correspondência."""
    output = df[df.isna().any(axis=1)]
    logger.info(f"Contagem de linhas nulas para o dataframe:{output.shape[0]}")
    return output

def frequencia_valores_nulos(df:DataFrame) -> DataFrame:
    """Função que gera uma matriz esparsa com a visualização dos valores nulos intercalado com valores preenchidos por coluna"""
    return df.stb.missing()

def verificar_linhas_duplicadas(df:DataFrame)->DataFrame:
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

def filtragem_interativa_valores_categoricos(df: DataFrame, coluna: str) -> DataFrame:
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

def adicionar_geolocalizacao_por_lista_bairros(df: DataFrame,lista_bairros: list,nome_coluna_bairro_df: str) -> DataFrame:
    """
    Adiciona colunas de latitude e longitude a um DataFrame com base em uma lista de bairros.

    Args:
        df (DataFrame): O DataFrame original a ser modificado.
        lista_bairros (list): Uma lista de strings com os bairros a serem geocodificados.
        nome_coluna_bairro_df (str): O nome da coluna de bairros no DataFrame original.

    Returns:
        DataFrame: O DataFrame original com as novas colunas 'latitude' e 'longitude' adicionadas.
    """
    # construção de um df com dados de localização para simplificar a atualização do df transformado
    df_geopy = pd.DataFrame(lista_bairros, columns=['Bairro'])
    
    # colunas necessárias pela lib geopy para instanciar os valores de geolocalizão
    df_geopy['cidade_estado'] = 'Rio de Janeiro, RJ'
    df_geopy['endereco_completo'] = df_geopy['Bairro'] + ', ' + df_geopy['cidade_estado']

    # instância do objeto de geolocalização com limite de tempo
    geolocator = Nominatim(user_agent="projeto_eda_airbnb", timeout=10)
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=2)

    # aplica a função para encontrar os dados de latitude e longitude.
    df_geopy['localizacao'] = df_geopy['endereco_completo'].apply(geocode)
    df_geopy['latitude'] = df_geopy['localizacao'].apply(lambda loc: loc.latitude if loc else None)
    df_geopy['longitude'] = df_geopy['localizacao'].apply(lambda loc: loc.longitude if loc else None)
    
    # junção do dataset com as colunas necessárias para retorno da função
    df_final = pd.merge(df, df_geopy[['Bairro','latitude','longitude']], left_on=nome_coluna_bairro_df, right_on='Bairro', how='left').drop('Bairro',axis=1)
        
    return df_final

def imputar_dados_room_type_entire_home_apt(df: DataFrame):
    """Função para transformar e tratar os valores das colunas bathrooms, bedrooms e 
    beds relacionados ao filtro da coluna room_type=='Entire home/apt'."""

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

def substituir_valores(df: DataFrame, filtro_linhas:list, filtro_colunas:list, valor) -> DataFrame:
    
    df.loc[filtro_linhas, filtro_colunas] = valor
    return df

def selecao_colunas(df: DataFrame, colunas: list) -> DataFrame:
    """Função que seleciona as colunas para montagem do dataset"""
    return df[colunas]


def agrupar_dados(df: DataFrame, cols_agrup: list, cols_filter: list=None, agr=None) -> DataFrame:
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


