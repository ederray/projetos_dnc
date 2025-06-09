"""Funções de tratamento dos dados"""
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt

def amostra_dados(df:pd.DataFrame)->pd.DataFrame:
    """Função para retornar a amostragem dos dados"""
    return df.sample(3)

def remover_duplicados(df:pd.DataFrame, coluna:str)->pd.DataFrame:
    """Função para remoção de valores duplicados."""
    df.drop_duplicates(subset = [coluna], keep='first',inplace=True)
    return df

def selecao_colunas(df:pd.DataFrame, colunas:list)->pd.DataFrame:
    """Função que seleciona as colunas para montagem do dataset"""
    return df[colunas]

def agrupar_dados(df:pd.DataFrame, colunas:list, agr)->pd.DataFrame:
    """Função que agrupa as colunas para montagem do dataset."""
    df = df.groupby(by=colunas).agg(agr)
    return df

def media_movel(df:pd.DataFrame, colunas:list, p:int)->pd.DataFrame:
    """Função que gera coluna de média móvel"""   
    for coluna in colunas: 
        df[f'{coluna}_rm'] = df[colunas].rolling(p).mean()
    return df

def grafico_decomposicao_temporal(df:pd.DataFrame,target:str, n:int):
    """Função para construção de uma decomposição de série temporal."""
    decomp = seasonal_decompose(df[target], model='additive', period=n)
    decomp.plot()
    return plt.show()

def grafico_acf():


def grafico_pacf():