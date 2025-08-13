"""Funções de tratamento dos dados"""
from IPython.display import display
from ipywidgets import interact, HTML, Output, Dropdown, VBox
import logging
import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame
import sidetable
import missingno as msno
import pandas as pd
from sklearn.preprocessing import StandardScaler, PowerTransformer

# instância do objeto logger
logger = logging.getLogger(__name__)

def amostra_dados(df: DataFrame) -> DataFrame:
    """Função para retornar a amostragem dos dados"""
    return df.sample(3)


def verificacao_nulos(df:DataFrame) -> pd.Series:
    """Função que realiza a contagem de valores nulos por feature do dataset"""
    output = df.isna().sum()
    return output

def filtrar_linhas_valores_nulos(df:DataFrame) -> pd.DataFrame:
    """Função que aplica o filtro de valores nulos no dataframe e retorna um dataframe filtrado com a correspondência."""
    output = df[df.isna().any(axis=1)]
    logger.info(f"Contagem de linhas nulas para o dataframe:{output.shape[0]}")
    return output

def matriz_valores_nulos(df:DataFrame)-> plt.plot:
    """Função que gera uma matriz esparsa com a visualização dos valores nulos intercalado com valores preenchidos por coluna"""
    msno.matrix(df, figsize=(10,4))
    plt.title("Matriz esparsa de valores nulos.", fontdict={'fontsize':12})
    return plt.show()

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


def boxplot_analise_descritiva_categorica(df: pd.DataFrame, distribuicao: list[float], feature: str):
    """
    Exibe um boxplot das features numéricas para cada valor selecionado de uma feature categórica.

    Args:
        df (pd.DataFrame): DataFrame com dados.
        distribuicao (list[float]): Lista de percentis para a análise (não usada diretamente aqui, mas mantida).
        feature (str): Coluna categórica a ser usada para seleção (ex: 'ticker', 'setor', etc.).
    """
    opcoes = sorted(df[feature].dropna().unique())

    @interact(coluna=opcoes)
    def plot(coluna):
        try:
            logger.info(f"Gerando boxplot para {feature} = {coluna}")

            # Filtra os dados com base na seleção
            dados_filtrados = df[df[feature] == coluna]
            dados_numericos = dados_filtrados.select_dtypes(include='number')

            if dados_numericos.empty:
                logger.warning(f"Nenhuma coluna numérica para {feature} = {coluna}")
                print("Nenhum dado numérico disponível.")
                return

            # Aplica minmax_scale
            scaler = StandardScaler()
            norm = PowerTransformer(method='yeo-johnson')
            dados_normalizados = pd.DataFrame(scaler.fit_transform(norm.fit_transform(dados_numericos)), 
                columns=dados_numericos.columns)

            # Cria boxplot com as colunas no eixo X
            plt.figure(figsize=(14, 6))
            dados_normalizados.boxplot()
            plt.xticks(rotation=60)
            plt.title(f"Análise Descritiva - {feature}: {coluna}")
            plt.ylabel("Valor Normalizado (MinMax)")
            plt.tight_layout()
            plt.show()

        except Exception as e:
            logger.error(f"Erro ao gerar gráfico: {e}")
            print("Erro:", e)


def histograma_feature_categorica(df: pd.DataFrame, feature: str):
    """
    Exibe histogramas das colunas numéricas para os registros filtrados por uma feature categórica.

    Args:
        df (pd.DataFrame): DataFrame com os dados.
        feature (str): Coluna categórica usada para filtrar os dados (ex: 'ticker').
    """
    opcoes = sorted(df[feature].dropna().unique())

    @interact(coluna=opcoes)
    def plot(coluna):
        try:
            logger.info(f"Gerando histograma para {feature} = {coluna}")

            # Filtra os dados
            dados_filtrados = df[df[feature] == coluna]
            dados_numericos = dados_filtrados.select_dtypes(include='number')

            transform_box_cox = PowerTransformer(method='yeo-johnson').fit_transform(dados_numericos)

            if dados_numericos.empty:
                logger.warning(f"Nenhuma coluna numérica para {feature} = {coluna}")
                print("Nenhum dado numérico disponível.")
                return

            # Normaliza com MinMaxScaler
            normalizado = pd.DataFrame(
                StandardScaler().fit_transform(transform_box_cox),
                columns=dados_numericos.columns
            )

            # Define layout dos subplots
            num_colunas = len(normalizado.columns)
            cols = 3  # Número de colunas de plots por linha
            rows = int(np.ceil(num_colunas / cols))

            fig, axs = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
            axs = axs.flatten()  # Para indexar linearmente

            for i, col in enumerate(normalizado.columns):
                axs[i].hist(normalizado[col], color='steelblue', alpha=0.7)
                axs[i].set_title(col)
                axs[i].set_xlabel('Valor Normalizado')
                axs[i].set_ylabel('Frequência')

            # Remove eixos não utilizados
            for j in range(i + 1, len(axs)):
                fig.delaxes(axs[j])

            fig.suptitle(f"Distribuição de Features Numéricas - {feature}: {coluna}", fontsize=16)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.show()

        except Exception as e:
            logger.error(f"Erro ao gerar gráfico: {e}")
            print("Erro:", e)