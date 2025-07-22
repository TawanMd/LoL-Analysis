"""
Módulo para gerenciar o download e carregamento de dados do Kaggle
"""
import os
import streamlit as st
import pandas as pd
import json
from kaggle.api.kaggle_api_extended import KaggleApi
import zipfile
from utils import get_logger

# Initialize logger
logger = get_logger(__name__)

def setup_kaggle_credentials() -> None:
    """
    Configura as credenciais do Kaggle a partir dos segredos do Streamlit.
    
    Cria o arquivo ~/.kaggle/kaggle.json com as credenciais e define as permissões.
    """
    # Verifica se os segredos necessários estão disponíveis
    if "KAGGLE_USERNAME" not in st.secrets or "KAGGLE_KEY" not in st.secrets:
        st.error("Credenciais do Kaggle não encontradas nos segredos. Por favor, adicione KAGGLE_USERNAME e KAGGLE_KEY.")
        st.stop()
    
    # Define o caminho do diretório .kaggle na home do usuário
    kaggle_dir = os.path.join(os.path.expanduser('~'), '.kaggle')
    os.makedirs(kaggle_dir, exist_ok=True)
    
    # Cria o dicionário com as credenciais
    kaggle_credentials = {
        "username": st.secrets["KAGGLE_USERNAME"],
        "key": st.secrets["KAGGLE_KEY"]
    }
    
    # Escreve as credenciais no arquivo kaggle.json
    kaggle_json_path = os.path.join(kaggle_dir, 'kaggle.json')
    with open(kaggle_json_path, 'w') as f:
        json.dump(kaggle_credentials, f)
    
    # Altera as permissões do arquivo para 600 (apenas o usuário pode ler e escrever)
    os.chmod(kaggle_json_path, 0o600)

def download_dataset() -> None:
    """
    Baixa o dataset do Kaggle se ainda não existir localmente.
    
    Configura as credenciais da API do Kaggle criando o arquivo kaggle.json
    a partir dos segredos do Streamlit e baixa o dataset.
    """
    # Nome do arquivo esperado
    filename = 'high_diamond_ranked_10min.csv'
    
    # Verifica se o arquivo já existe
    if os.path.exists(filename):
        logger.info(f"Dataset '{filename}' já existe. Pulando download.")
        return
    
    logger.info("Dataset não encontrado. Iniciando download do Kaggle...")
    
    try:
        # Configura as credenciais do Kaggle
        setup_kaggle_credentials()
        
        # Instancia e autentica a API do Kaggle
        api = KaggleApi()
        api.authenticate()
        
        # Baixa o dataset
        dataset_name = 'bobbyscience/league-of-legends-diamond-ranked-games-10-min'
        api.dataset_download_files(dataset_name, path='.', unzip=True)
        
        logger.info(f"Download concluído! Dataset '{filename}' baixado com sucesso.")
        
    except Exception as e:
        logger.error(f"Erro ao baixar o dataset: {str(e)}")
        raise

@st.cache_data
def load_data() -> pd.DataFrame:
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