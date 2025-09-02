"""Funções de tratamento dos dados"""
import logging
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from geopy.distance import great_circle
from scipy.stats import chisquare, shapiro, kstest, norm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import numpy as np
from pandas import DataFrame, Series
import pandas as pd
from ipywidgets import interact, HTML, Output, Dropdown, VBox, interactive
from IPython.display import display, HTML
from sklearn.preprocessing import PowerTransformer
from typing import Dict, Tuple, Optional

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

def verificar_linhas_duplicadas(df:DataFrame) -> DataFrame:
    """Função que retorna um dataframe contendo as linhas duplicadas do dataset inputado."""
    output = \
    (df.groupby(df.columns.tolist(), dropna=False)
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
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=0.5)

    # aplica a função para encontrar os dados de latitude e longitude.
    df_geopy['localizacao'] = df_geopy['endereco_completo'].apply(geocode)
    df_geopy['latitude'] = df_geopy['localizacao'].apply(lambda loc: loc.latitude if loc else None)
    df_geopy['longitude'] = df_geopy['localizacao'].apply(lambda loc: loc.longitude if loc else None)
    
    # junção do dataset com as colunas necessárias para retorno da função
    df_final = pd.merge(df, df_geopy[['Bairro','latitude','longitude']], left_on=nome_coluna_bairro_df, right_on='Bairro', how='left').drop('Bairro',axis=1)
        
    return df_final


def adicionar_informacoes_geograficas(
    df: DataFrame,
    nome_coluna_bairro_df: str,
    zonas_rj: Optional[Dict[str, list]] = None,
    pontos_transporte: Optional[Dict[str, Tuple[float, float]]] = None,
    pontos_turisticos: Optional[Dict[str, Tuple[float, float]]] = None,
    user_agent: str = "projeto_eda_airbnb",
) -> DataFrame:
    """
    Adiciona colunas de latitude, longitude, zona da cidade e proximidade a pontos de interesse.

    Args:
        df (DataFrame): O DataFrame original a ser modificado.
        nome_coluna_bairro_df (str): O nome da coluna de bairros no DataFrame original.
        zonas_rj (dict, opcional): Mapeamento de zonas para bairros.
        pontos_transporte (dict, opcional): Nome -> coordenadas (lat, lon) de pontos de transporte.
        pontos_turisticos (dict, opcional): Nome -> coordenadas (lat, lon) de pontos turísticos.
        user_agent (str, opcional): User agent para a API do Nominatim.

    Returns:
        DataFrame: O DataFrame original com as novas colunas adicionadas.
    """

    # =========================
    # Valores padrão
    # =========================
    if zonas_rj is None:
        zonas_rj = {
            "Zona Sul": ["Botafogo", "Catete", "Copacabana", "Cosme Velho", "Flamengo", "Gávea", "Humaitá",
                         "Ipanema", "Jardim Botânico", "Lagoa", "Laranjeiras", "Leblon", "Leme", "Rocinha",
                         "São Conrado", "Urca", "Vidigal"],
            "Zona Norte": ["Tijuca", "Vila Isabel", "Méier", "Madureira"],  # simplificado
            "Zona Oeste": ["Bangu", "Barra da Tijuca", "Campo Grande", "Recreio dos Bandeirantes"],
            "Centro": ["Centro", "Lapa", "Glória", "Santa Teresa"],
        }

    if pontos_transporte is None:
        pontos_transporte = {
            "Aeroporto_Santos_Dumont": (-22.9103, -43.1633),
            "Aeroporto_Galeão": (-22.8130, -43.2471),
            "Estacao_Central_do_Brasil": (-22.9031, -43.1901),
            "Estação_Cinelândia":(-22.9103,-43.1762),
            "Estação_Jardim_Oceânico": (-23.0069, -43.3039),
            "Estação_Botafogo": (-22.9515, -43.1812),
            "Estação_Maracanã": (-22.9126, -43.2280),
            "Estação_Pavuna": (-22.8152, -43.3644)
        }

    if pontos_turisticos is None:
        pontos_turisticos = {
            'Pão_de_Açúcar': (-22.9519, -43.1593),
            'Cristo_Redentor': (-22.9519, -43.2104),
            'Praia_de_Copacabana': (-22.9712, -43.1852),
            'Maracanã': (-22.9121, -43.2302),
            'Jardim_Botânico': (-22.9691, -43.2259),
            'Sapucaí_(Sambódromo)': (-22.9070, -43.1947),
            'Estádio_Nilton_Santos': (-22.8931, -43.2905),   
        }

    # =========================
    # Funções auxiliares
    # =========================
    def obter_zona(bairro: str) -> str:
        for zona, bairros_lista in zonas_rj.items():
            if bairro in bairros_lista:
                return zona
        return "Não mapeada"

    def calcular_distancia(coord1: Tuple[float, float], coord2: Tuple[float, float]) -> Optional[float]:
        if None in coord1 or None in coord2:
            return None
        return great_circle(coord1, coord2).km

    # =========================
    # Geocodificação
    # =========================
    bairros_unicos = df[nome_coluna_bairro_df].dropna().unique()
    enderecos = [f"{bairro}, Rio de Janeiro, RJ" for bairro in bairros_unicos]

    geolocator = Nominatim(user_agent=user_agent, timeout=10)
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=0.5)

    mapa_coordenadas = {}
    for bairro, endereco in zip(bairros_unicos, enderecos):
        try:
            localizacao = geocode(endereco)
            if localizacao:
                mapa_coordenadas[bairro] = (localizacao.latitude, localizacao.longitude)
            else:
                mapa_coordenadas[bairro] = (None, None)
        except Exception:
            mapa_coordenadas[bairro] = (None, None)

    # =========================
    # Adicionar colunas
    # =========================
    df = df.copy()
    df["Zona"] = df[nome_coluna_bairro_df].map(obter_zona)
    df["latitude"] = df[nome_coluna_bairro_df].map(lambda b: mapa_coordenadas.get(b, (None, None))[0])
    df["longitude"] = df[nome_coluna_bairro_df].map(lambda b: mapa_coordenadas.get(b, (None, None))[1])

    coords_df = list(zip(df["latitude"], df["longitude"]))

    # Distâncias para pontos de transporte
    for nome, coords in pontos_transporte.items():
        df[f"Dist_{nome}_km"] = [
            calcular_distancia(coord, coords) for coord in coords_df
        ]

    # Distâncias para pontos turísticos
    for nome, coords in pontos_turisticos.items():
        df[f"Dist_{nome}_km"] = [
            calcular_distancia(coord, coords) for coord in coords_df
        ]

    return df

def criar_flags_proximidade(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cria colunas de flag binárias (0 ou 1) para indicar proximidade a pontos de interesse,
    com base em regras pré-definidas e com uma nomenclatura de coluna personalizada.
    
    Args:
        df (pd.DataFrame): O DataFrame original com as colunas de distância em km.

    Returns:
        pd.DataFrame: O DataFrame com as novas colunas de flag adicionadas.
    """
    df_com_flags = df.copy()

    # Dicionário com as regras de proximidade fornecidas
    limites_proximidade = {
        'Aeroporto_Santos_Dumont': 10,
        'Aeroporto_Galeão': 15,
        'Pão de Açúcar': 5,
        'Cristo Redentor': 5,
        'Praia de Copacabana': 5,
        'Maracanã': 10,
        'Jardim Botânico': 5,
        'Sapucaí (Sambódromo)': 10,
        'Estádio Nilton Santos': 10,
        'Estacao_Central_do_Brasil': 10,
        'Estação_Cinelândia':5,
        'Estação_Jardim_Oceânico': 5,
        'Estação_Botafogo': 5,
        'Estação_Maracanã': 5,
        'Estação_Pavuna': 5
    }
    
    for ponto, limite in limites_proximidade.items():
        # Trata o nome do ponto para remover caracteres especiais, caso existam
        ponto_formatado = ponto.replace('Ã£', 'ã').replace('Ã§', 'ç').replace('Ã³', 'ó')
        
        # Constrói o nome da coluna de distância original
        coluna_dist = f"Dist_{ponto_formatado.replace(' ', '_')}_km"
        
        # Constrói o novo nome da coluna de flag, seguindo o padrão que você pediu
        nome_local = ponto_formatado.replace(' ', '_')
        coluna_flag = f"Fl_lteq_{limite}_km_{nome_local}"
        
        if coluna_dist in df_com_flags.columns:
            df_com_flags[coluna_flag] = df_com_flags[coluna_dist].apply(
                lambda x: 1 if pd.notna(x) and x <= limite else 0
            )
            print(f"Coluna '{coluna_flag}' criada com sucesso.")
        else:
            print(f"Aviso: Coluna de distância '{coluna_dist}' não encontrada. A flag não será criada.")
            
    return df_com_flags

def imputar_dados_room_type_entire_home_apt(df: DataFrame):
    """Função para transformar e tratar os valores das colunas bathrooms, bedrooms e 
    beds relacionados ao filtro da coluna room_type=='Entire home/apt'."""

    # filtra o dataset a partir dos valores da coluna room_type == 'Entire home/apt'
    df_filtrado = filtrar_feature_valor_categorico(df, query="room_type=='Entire home/apt'")

    # 'Entire home/apt' exige a presença de 1 banheiro na residência por legislação.
    df_filtrado.loc[df_filtrado['bathrooms']<1,'bathrooms'] = 1

    # 'Entire home/apt' com bedrooms e beds menor que 1 provavelmente corresponde a um tipo de acomodação kitnet ou studio.
    df_filtrado.loc[(df_filtrado['bedrooms'] < 1), 'bedrooms'] = 0
    df_filtrado.loc[(df_filtrado['beds'] < 1), 'beds'] = 0

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
    df_filtrado.loc[(df_filtrado['bedrooms'].isna()) | (df['bedrooms'] == 0),'bedrooms'] = 1

    # quatidade de camas definidas a partir de uma taxa de acomodações/2
    df_filtrado.loc[(df_filtrado['beds'].isna()) | (df['beds'] == 0),'beds'] = np.ceil(df_filtrado['accommodates'] / 2)

    return df_filtrado

def imputar_dados_room_type_shared_room(df: DataFrame):
    """Função para transformar e tratar os valores das colunas bathrooms, bedrooms e 
    beds relacioanados ao filtro da coluna room_type=='Shared room'."""

    # filtra o dataset a partir dos valores da coluna room_type == 'Shared room'
    df_filtrado = filtrar_feature_valor_categorico(df, query="room_type=='Shared room'")

    # 'Shared room' não exige a presença de 1 quarto ou banheiro exclusivos.
    df_filtrado.loc[df_filtrado['bedrooms'].isna(),['bedrooms']] = 0
    df_filtrado.loc[df_filtrado['bathrooms'].isna(),['bathrooms']] = 0
    df_filtrado.loc[df_filtrado['beds'].isna(),['beds']] = 0

    return df_filtrado

def imputar_dados_room_type_hotel_room(df: DataFrame):
    """Função para transformar e tratar os valores das colunas bathrooms, bedrooms e 
    beds relacioanados ao filtro da coluna room_type=='Hotel room'."""

    # filtra o dataset a partir dos valores da coluna room_type == 'Hotel room'
    df_filtrado = filtrar_feature_valor_categorico(df, query="room_type=='Hotel room'")

    # quantidade de banheiros menor que 1 preenchidos com valor 1, já que quarto de hotel tem banheiro.
    df_filtrado.loc[df_filtrado['bathrooms']<1,'bathrooms'] = 1
    df_filtrado.loc[df_filtrado['bedrooms']<1,'bedrooms'] = 1

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
    """Função que subsitui os valores a partir dos filtros de linha ou coluna informados para o valor determinado."""
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

def teste_qui_quadrado_normalidade(df:DataFrame, cat_col:str, num_cols:list, bins=10, alpha=0.05) -> DataFrame:
    "Função que gera um dataset com a avaliação do teste qui quadrado das categorias da feature indicada."
    results = []

    for category in df[cat_col].unique():
        df_category = df[df[cat_col] == category]
        
        for num_col in num_cols:
            data = df_category[num_col].dropna()

            if len(data) < bins:
                continue

            # padronizar
            zscores = (data - data.mean()) / data.std()

            # observado
            obs, bin_edges = np.histogram(zscores, bins=bins)
            
            # esperado (usando normal padrão)
            cdf_vals = norm.cdf(bin_edges)
            expected_probs = np.diff(cdf_vals)
            expected = expected_probs * len(zscores)

            # teste qui-quadrado
            chi2, p = chisquare(f_obs=obs, f_exp=expected)

            results.append({
                "Categoria": category,
                "Coluna": num_col,
                "Chi2": chi2,
                "p-value": p,
                "Normal?": "Sim" if p > alpha else "Não"
            })

    return pd.DataFrame(results)

def teste_normalidade_por_categoria_auto(df:DataFrame, cat_col:str, num_cols:list, alpha=0.05) -> DataFrame:
    """Função que aplica teste de normalidade para as categorias de uma coluna a partir da quantidade de amostra disponível e retorna um dataset com as análises."""
    results = []

    for category in df[cat_col].unique():
        subset = df[df[cat_col] == category]

        for num_col in num_cols:
            data = subset[num_col].dropna().values
            n = len(data)

            if n < 3:  # amostra muito pequena
                results.append({
                    "Categoria": category,
                    "Coluna": num_col,
                    "N": n,
                    "Teste": None,
                    "Estatística": None,
                    "p-value": None,
                    "Normal?": "Amostra insuficiente"
                })
                continue

            # escolha do teste
            if n < 500:
                test_name = "Shapiro-Wilk"
                stat, p = shapiro(data)
            else:
                test_name = "Kolmogorov-Smirnov"
                # padronizar antes de aplicar KS contra normal padrão
                zscores = (data - np.mean(data)) / np.std(data, ddof=1)
                stat, p = kstest(zscores, 'norm')

            results.append({
                "Categoria": category,
                "Coluna": num_col,
                "N": n,
                "Teste": test_name,
                "Estatística": stat,
                "p-value": p,
                "Normal?": "Sim" if p > alpha else "Não"
            })

    return pd.DataFrame(results)

def verificacao_outlier(array, extreme = False):

    "Função para verificar outliers em um array."
    q1,q3 = np.quantile(array, [0.25, 0.75])
    iqr = q3-q1

    factor = 3 if extreme else 1.5
    upper_outlier = q3+factor*iqr
    lower_outlier = q1-factor*iqr

    return (array < lower_outlier) | (array > upper_outlier)

def power_transform_coluna_categorica(df: pd.DataFrame,cat_col: str,metodo: str = 'yeo-johnson', cols: Optional[list] = None) -> pd.DataFrame:
    """
    Aplica PowerTransformer (Box-Cox ou Yeo-Johnson) às colunas numéricas,
    agrupando os dados por uma coluna categórica.

    Args:
        df (pd.DataFrame): DataFrame de entrada com colunas numéricas e categóricas.
        cat_col (str): Nome da coluna categórica usada para agrupar.
        metodo (str, optional): Método do PowerTransformer ('yeo-johnson' ou 'box-cox').
        cols (list, optional): Lista de colunas numéricas a transformar. 
                               Se None, aplica em todas as numéricas.

    Returns:
        pd.DataFrame: DataFrame com as colunas numéricas transformadas por grupo.
    """
    df = df.copy()
    
    # Seleção de colunas numéricas (caso o usuário não especifique)
    if cols is None:
        cols = df.select_dtypes(include='number').columns.tolist()

    def _transform(group: pd.DataFrame) -> DataFrame:
        transformer = PowerTransformer(method=metodo, standardize=True)
        group = group.copy()
        group[cols] = transformer.fit_transform(group[cols])
        return group

    return df.groupby(cat_col, group_keys=False).apply(_transform)


def analise_vif_interativo(df: pd.DataFrame, coluna: str):
    """Função que realiza o teste VIF para as features da tabela a partir da coluna de filtro."""

    lista = sorted(df[coluna].dropna().unique())

    @interact(valor_selecionado=lista)
    def executar_analise_vif(valor_selecionado):
        # Filtra pelo valor selecionado
        df_filtrado = df[df[coluna] == valor_selecionado].copy()

        # Seleciona apenas features numéricas
        features_num = df_filtrado.select_dtypes(include='number').columns
        df_features = df_filtrado[features_num].dropna()

        if df_features.shape[1] < 2:
            display(HTML(f"<h3>Poucas features numéricas para {coluna}: {valor_selecionado}</h3>"))
            return

        # Função para calcular o VIF
        def vif_calculator(df_to_vif):
            vif_data = pd.DataFrame()
            vif_data['Feature'] = df_to_vif.columns
            vif_data['VIF'] = [
                variance_inflation_factor(df_to_vif.values, i) 
                for i in range(df_to_vif.shape[1])
            ]
            return vif_data.sort_values(by="VIF", ascending=False)

        vif_resultado = vif_calculator(df_features)

        # Exibe o resultado
        display(HTML(f"<h3>Análise VIF para {coluna}: {valor_selecionado}</h3>"))
        display(vif_resultado)

def analise_vif_interativo_2(df: pd.DataFrame, coluna: str):
    """Função que realiza o teste VIF para as features da tabela a partir da coluna de filtro."""

    lista = sorted(df[coluna].dropna().unique())

    @interact(valor_selecionado=lista)
    def executar_analise_vif(valor_selecionado):
        # Filtra pelo valor selecionado
        df_filtrado = df[df[coluna] == valor_selecionado].copy()

        # Seleciona apenas features numéricas
        features_num = df_filtrado.select_dtypes(include='number').columns
        df_features = df_filtrado[features_num].dropna()

        # Remove colunas constantes (sem variação)
        df_features = df_features.loc[:, df_features.nunique() > 1]

        if df_features.shape[1] < 2:
            display(HTML(f"<h3>Poucas features numéricas válidas para {coluna}: {valor_selecionado}</h3>"))
            return

        # Função para calcular o VIF
        def vif_calculator(df_to_vif):
            vif_data = pd.DataFrame()
            vif_data['Feature'] = df_to_vif.columns
            vif_data['VIF'] = [
                variance_inflation_factor(df_to_vif.values, i) 
                for i in range(df_to_vif.shape[1])
            ]
            # Substitui inf por NaN e remove linhas inválidas
            vif_data = vif_data.replace([np.inf, -np.inf], np.nan).dropna()
            return vif_data.sort_values(by="VIF", ascending=False)

        vif_resultado = vif_calculator(df_features)

        # Exibe o resultado
        display(HTML(f"<h3>Análise VIF para {coluna}: {valor_selecionado}</h3>"))
        display(vif_resultado)