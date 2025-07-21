"""
Módulo para gerenciar o download e carregamento de dados do Kaggle
"""
import os
import streamlit as st
import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi
import zipfile


def download_dataset():
    """
    Baixa o dataset do Kaggle se ainda não existir localmente.
    
    Configura as credenciais da API do Kaggle usando st.secrets e
    baixa o dataset 'bobbyscience/league-of-legends-diamond-ranked-games-10-min'.
    """
    # Nome do arquivo esperado
    filename = 'high_diamond_ranked_10min.csv'
    
    # Verifica se o arquivo já existe
    if os.path.exists(filename):
        print(f"Dataset '{filename}' já existe. Pulando download.")
        return
    
    print("Dataset não encontrado. Iniciando download do Kaggle...")
    
    try:
        # Configura as variáveis de ambiente com as credenciais do Streamlit
        os.environ['KAGGLE_USERNAME'] = st.secrets['kaggle']['username']
        os.environ['KAGGLE_KEY'] = st.secrets['kaggle']['key']
        
        # Instancia e autentica a API do Kaggle
        api = KaggleApi()
        api.authenticate()
        
        # Baixa o dataset
        dataset_name = 'bobbyscience/league-of-legends-diamond-ranked-games-10-min'
        api.dataset_download_files(dataset_name, path='.', unzip=True)
        
        print(f"Download concluído! Dataset '{filename}' baixado com sucesso.")
        
    except Exception as e:
        print(f"Erro ao baixar o dataset: {str(e)}")
        raise


@st.cache_data
def load_data():
    """
    Carrega o dataset em um DataFrame do Pandas.
    
    Esta função é decorada com @st.cache_data para otimizar a performance,
    evitando recarregar os dados a cada interação.
    
    Returns:
        pd.DataFrame: DataFrame contendo os dados do jogo
    """
    # Garante que o dataset foi baixado
    download_dataset()
    
    # Carrega o CSV em um DataFrame
    df = pd.read_csv('high_diamond_ranked_10min.csv')
    
    return df