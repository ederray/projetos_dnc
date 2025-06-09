"""Função para manipulação de datasets"""
import pandas as pd
def salvar_dataset(df:pd.DataFrame, path:str):
    """Função para salvar o dataset na pasta destino."""
    return df.to_csv(path)
