"""Função de construção de features com dados de geolocalização"""

import logging
import re
from typing import Dict, Optional, Tuple
import unicodedata

from geopy.distance import great_circle
from geopy.extra.rate_limiter import RateLimiter
from geopy.geocoders import Nominatim
import pandas as pd
from pandas import DataFrame

# instância do objeto logger
logger = logging.getLogger(__name__)


def adicionar_geolocalizacao_por_lista_bairros(
    df: DataFrame, lista_bairros: list, nome_coluna_bairro_df: str
) -> DataFrame:
    """
    Adiciona colunas de latitude e longitude a um DataFrame com base em uma lista de bairros.
    Uso da api geopy para pesquisa dos dados.

    params:
        df (DataFrame): O DataFrame original a ser modificado.
        lista_bairros (list): Uma lista de strings com os bairros a serem geocodificados.
        nome_coluna_bairro_df (str): O nome da coluna de bairros no DataFrame original.

    returns:
        DataFrame: O DataFrame original com as novas colunas 'localizacao', 'latitude' e 'longitude' adicionadas.
    """
    try:
        df_geopy = pd.DataFrame(lista_bairros, columns=["Bairro"])

        # colunas necessárias pela lib geopy para instanciar os valores de geolocalizão
        df_geopy["cidade_estado"] = "Rio de Janeiro, RJ"
        df_geopy["endereco_completo"] = df_geopy["Bairro"] + ", " + df_geopy["cidade_estado"]

        geolocator = Nominatim(user_agent="projeto_eda_airbnb", timeout=10)
        geocode = RateLimiter(geolocator.geocode, min_delay_seconds=0.4)

        df_geopy["localizacao"] = df_geopy["endereco_completo"].apply(geocode)
        df_geopy["latitude"] = df_geopy["localizacao"].apply(
            lambda loc: loc.latitude if loc else None
        )
        df_geopy["longitude"] = df_geopy["localizacao"].apply(
            lambda loc: loc.longitude if loc else None
        )

        df_final = pd.merge(
            df,
            df_geopy[["Bairro", "latitude", "longitude"]],
            left_on=nome_coluna_bairro_df,
            right_on="Bairro",
            how="left",
        ).drop("Bairro", axis=1)

        return df_final

    except Exception as e:
        logger.error(e)
        raise


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

    params:
        df (DataFrame): O DataFrame original a ser modificado.
        nome_coluna_bairro_df (str): O nome da coluna de bairros no DataFrame original.
        zonas_rj (dict, opcional): Mapeamento de zonas para bairros.
        pontos_transporte (dict, opcional): Nome -> coordenadas (lat, lon) de pontos de transporte.
        pontos_turisticos (dict, opcional): Nome -> coordenadas (lat, lon) de pontos turísticos.
        user_agent (str, opcional): User agent para a API do Nominatim.

    returns:
        DataFrame: O DataFrame original com as novas colunas adicionadas.
    """

    try:
        # =========================
        # Valores padrão
        # =========================
        if zonas_rj is None:
            zonas_rj = {
                "Zona Sul": [
                    "Botafogo",
                    "Catete",
                    "Copacabana",
                    "Cosme Velho",
                    "Flamengo",
                    "Gávea",
                    "Glória",
                    "Humaitá",
                    "Ipanema",
                    "Jardim Botânico",
                    "Lagoa",
                    "Laranjeiras",
                    "Leblon",
                    "Leme",
                    "Rocinha",
                    "São Conrado",
                    "Urca",
                    "Vidigal",
                ],
                "Zona Norte": [
                    "Acari",
                    "Abolição",
                    "Água Santa",
                    "Alto da Boa Vista",
                    "Anchieta",
                    "Andaraí",
                    "Bancários",
                    "Barros Filho",
                    "Bento Ribeiro",
                    "Bonsucesso",
                    "Brás de Pina",
                    "Cachambi",
                    "Cacuia",
                    "Campinho",
                    "Cascadura",
                    "Cavalcanti",
                    "Cidade Universitária",
                    "Cocotá",
                    "Coelho Neto",
                    "Colégio",
                    "Complexo do Alemão",
                    "Cordovil",
                    "Costa Barros",
                    "Del Castilho",
                    "Encantado",
                    "Engenheiro Leal",
                    "Engenho da Rainha",
                    "Engenho de Dentro",
                    "Engenho Novo",
                    "Freguesia",
                    "Freguesia (Ilha)",
                    "Galeão",
                    "Grajaú",
                    "Guadalupe",
                    "Higienópolis",
                    "Honório Gurgel",
                    "Irajá",
                    "Inhaúma",
                    "Jacaré",
                    "Jacarezinho",
                    "Jardim América",
                    "Jardim Carioca",
                    "Jardim Guanabara",
                    "Lins de Vasconcelos",
                    "Madureira",
                    "Manguinhos",
                    "Maracanã",
                    "Maré",
                    "Marechal Hermes",
                    "Maria da Graça",
                    "Méier",
                    "Moneró",
                    "Olaria",
                    "Oswaldo Cruz",
                    "Osvaldo Cruz",
                    "Parada de Lucas",
                    "Parque Anchieta",
                    "Parque Colúmbia",
                    "Pavuna",
                    "Penha",
                    "Penha Circular",
                    "Piedade",
                    "Pilares",
                    "Pitangueiras",
                    "Portuguesa",
                    "Praia da Bandeira",
                    "Praça da Bandeira",
                    "Quintino Bocaiúva",
                    "Ramos",
                    "Riachuelo",
                    "Ribeira",
                    "Ricardo de Albuquerque",
                    "Rocha",
                    "Rocha Miranda",
                    "Sampaio",
                    "São Francisco Xavier",
                    "Todos os Santos",
                    "Tauá",
                    "Tomás Coelho",
                    "Tijuca",
                    "Turiaçú",
                    "Vaz Lobo",
                    "Vicente de Carvalho",
                    "Vigário Geral",
                    "Vista Alegre",
                    "Vila da Penha",
                    "Vila Isabel",
                    "Vila Kosmos",
                    "Zumbi",
                ],
                "Zona Oeste": [
                    "Anil",
                    "Barra da Tijuca",
                    "Camorim",
                    "Campo Grande",
                    "Cidade de Deus",
                    "Curicica",
                    "Freguesia de Jacarepaguá",
                    "Freguesia (Jacarepaguá)",
                    "Gardênia Azul",
                    "Grumari",
                    "Itanhangá",
                    "Jacarepaguá",
                    "Joá",
                    "Pechincha",
                    "Praça Seca",
                    "Rio das Pedras",
                    "Recreio dos Bandeirantes",
                    "Tanque",
                    "Taquara",
                    "Vargem Grande",
                    "Vargem Pequena",
                    "Vila Valqueire",
                    "Jardim Sulacap",
                    "Bangu",
                    "Campo dos Afonsos",
                    "Deodoro",
                    "Padre Miguel",
                    "Realengo",
                    "Santíssimo",
                    "Senador Camará",
                    "Senador Vasconcelos",
                    "Sepetiba",
                    "Vila Kennedy",
                    "Vila Militar",
                    "Barra de Guaratiba",
                    "Gericinó",
                    "Guaratiba",
                    "Inhoaíba",
                    "Paciência",
                    "Pedra de Guaratiba",
                    "Santa Cruz",
                    "Cosmos",
                ],
                "Centro": [
                    "Benfica",
                    "Bento Ribeiro",
                    "Catumbi",
                    "Caju",
                    "Centro",
                    "Cidade Nova",
                    "Estácio",
                    "Gamboa",
                    "Glória",
                    "Lapa",
                    "Mangueira",
                    "Paquetá",
                    "Rio Comprido",
                    "Santa Teresa",
                    "Santo Cristo",
                    "Saúde",
                    "São Cristóvão",
                    "Vasco da Gama",
                ],
            }

        if pontos_transporte is None:
            pontos_transporte = {
                "Aeroporto_Santos_Dumont": (-22.9103, -43.1633),
                "Aeroporto_Galeão": (-22.8130, -43.2471),
                "Estacao_Central_do_Brasil": (-22.9031, -43.1901),
                "Estação_Cinelândia": (-22.9103, -43.1762),
                "Estação_Jardim_Oceânico": (-23.0069, -43.3039),
                "Estação_Botafogo": (-22.9515, -43.1812),
                "Estação_Maracanã": (-22.9126, -43.2280),
                "Estação_Pavuna": (-22.8152, -43.3644),
            }

        if pontos_turisticos is None:
            pontos_turisticos = {
                "Pão_de_Açúcar": (-22.9519, -43.1593),
                "Cristo_Redentor": (-22.9519, -43.2104),
                "Praia_de_Copacabana": (-22.9712, -43.1852),
                "Arcos_da_Lapa": (-22.9128, -43.1799),
                "Museu_do_Amanha": (-22.8940, -43.1794),
                "Maracanã": (-22.9121, -43.2302),
                "Jardim_Botânico": (-22.9691, -43.2259),
                "Sapucaí (Sambódromo)": (-22.9070, -43.1947),
                "Estádio_Nilton_Santos": (-22.8931, -43.2905),
            }

        # =========================
        # Funções auxiliares
        # =========================
        def obter_zona(bairro: str) -> str:
            for zona, bairros_lista in zonas_rj.items():
                if bairro in bairros_lista:
                    return zona
            return "Não mapeada"

        def calcular_distancia(
            coord1: Tuple[float, float], coord2: Tuple[float, float]
        ) -> Optional[float]:
            if None in coord1 or None in coord2:
                return None
            return great_circle(coord1, coord2).km

        # =========================
        # Geocodificação
        # =========================
        bairros_unicos = df[nome_coluna_bairro_df].dropna().unique()
        enderecos = [f"{bairro}, Rio de Janeiro, RJ" for bairro in bairros_unicos]

        geolocator = Nominatim(user_agent=user_agent, timeout=10)
        geocode = RateLimiter(geolocator.geocode, min_delay_seconds=0.4)

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
        df["latitude"] = df[nome_coluna_bairro_df].map(
            lambda b: mapa_coordenadas.get(b, (None, None))[0]
        )
        df["longitude"] = df[nome_coluna_bairro_df].map(
            lambda b: mapa_coordenadas.get(b, (None, None))[1]
        )

        coords_df = list(zip(df["latitude"], df["longitude"]))

        # Distâncias para pontos de transporte
        for nome, coords in pontos_transporte.items():
            df[f"Dist_{nome}_km"] = [calcular_distancia(coord, coords) for coord in coords_df]

        # Distâncias para pontos turísticos
        for nome, coords in pontos_turisticos.items():
            df[f"Dist_{nome}_km"] = [calcular_distancia(coord, coords) for coord in coords_df]

        return df

    except Exception as e:
        logger.error(e)
        raise


def criar_flags_proximidade(df: DataFrame) -> DataFrame:
    """
    Cria colunas de flag binárias (0 ou 1) para indicar proximidade a pontos de interesse,
    comparando nomes de colunas de forma mais tolerante (sem acentos, case-insensitive).

    params:
        df (DataFrame): O DataFrame original com as colunas de distância em km.

    return:
        DataFrame: DataFrame com as novas colunas de flag adicionadas.
    """
    try:

        def normalizar_texto(texto: str) -> str:
            # remove acentos, converte para ascii, substitui não alfanumérico por _
            texto = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode("utf-8")
            texto = re.sub(r"[^0-9a-zA-Z_]+", "_", texto)
            return texto.lower().strip("_")

        df_com_flags = df.copy()

        # Dicionário com as regras de proximidade fornecidas
        limites_proximidade = {
            "Aeroporto_Santos_Dumont": 10,
            "Aeroporto_Galeão": 15,
            "Pão de Açúcar": 5,
            "Cristo Redentor": 5,
            "Praia de Copacabana": 5,
            "Maracanã": 5,
            "Arcos da Lapa": 5,
            "Museu do Amanhã": 5,
            "Jardim Botânico": 5,
            "Sapucaí (Sambódromo)": 5,
            "Estádio Nilton Santos": 10,
            "Estacao Central do Brasil": 10,
            "Estação Cinelândia": 5,
            "Estação Jardim Oceânico": 5,
            "Estação Botafogo": 5,
            "Estação Maracanã": 5,
            "Estação Pavuna": 5,
        }

        # normaliza colunas existentes do df uma vez
        colunas_normalizadas = {normalizar_texto(c): c for c in df_com_flags.columns}

        for ponto, limite in limites_proximidade.items():
            nome_normalizado = normalizar_texto(f"dist_{ponto}_km")

            if nome_normalizado in colunas_normalizadas:
                col_df = colunas_normalizadas[nome_normalizado]

                nome_local_flag = normalizar_texto(ponto)
                coluna_flag = f"fl_lteq_{limite}_km_{nome_local_flag}"

                df_com_flags[coluna_flag] = df_com_flags[col_df].apply(
                    lambda x: 1 if pd.notna(x) and x <= limite else 0
                )
                print(f"Coluna '{coluna_flag}' criada com sucesso a partir de '{col_df}'.")
            else:
                print(
                    f"Aviso: Coluna de distância para '{ponto}' não encontrada. A flag não será criada."
                )

        return df_com_flags

    except Exception as e:
        logger.error(e)
        raise
