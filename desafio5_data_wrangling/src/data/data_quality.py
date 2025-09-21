"""Funções para inspeção e avaliação da qualidade dos dados."""
import logging
from pandas import DataFrame, Series
import sidetable as stb

# Configuração do logger para este módulo
logger = logging.getLogger(__name__)


def amostra_dados(df: DataFrame) -> DataFrame:
    """Função para retornar uma amostragem aleatória dos dados."""
    return df.sample(3)


def contagem_valores(coluna: Series) -> None:
    """Função que realiza a contagem de valores por coluna."""
    return coluna.value_counts()


def verificacao_nulos(df: DataFrame) -> Series:
    """Função que realiza a contagem de valores nulos por feature do dataset."""
    return df.isna().sum()


def filtrar_linhas_valores_nulos(df: DataFrame) -> DataFrame:
    """Função que aplica o filtro de valores nulos no dataframe e retorna um dataframe filtrado."""
    output = df[df.isna().any(axis=1)]
    logger.info(f"Contagem de linhas com valores nulos: {output.shape[0]}")
    return output


def frequencia_valores_nulos(df: DataFrame) -> DataFrame:
    """Gera uma tabela com a contagem e frequência de valores nulos por coluna."""
    return df.stb.missing()


def verificar_linhas_duplicadas(df: DataFrame) -> DataFrame:
    """Função que retorna um dataframe contendo as linhas duplicadas do dataset inputado."""
    output = (
        df.groupby(df.columns.tolist(), dropna=False)
        .size()
        .to_frame('n_duplicates')
        .query('n_duplicates>1')
        .sort_values('n_duplicates', ascending=False)
        .head(5)
    )
    return output


def remover_duplicados(df: DataFrame, coluna: str) -> DataFrame:
    """Função para remoção de valores duplicados em uma coluna."""
    df.drop_duplicates(subset=[coluna], keep='first', inplace=True)
    return df