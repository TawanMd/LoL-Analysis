"""
Script de exemplo para configurar e testar o projeto LoL Analysis
"""
import os
import json
from utils import get_logger

# Initialize logger
logger = get_logger(__name__)


def create_example_secrets() -> None:
    """Cria um exemplo de arquivo secrets.toml"""
    secrets_content = """# Exemplo de configuração - SUBSTITUA COM SUAS CREDENCIAIS REAIS!
[kaggle]
username = "seu_username_aqui"
key = "sua_chave_api_aqui"
"""
    
    os.makedirs('.streamlit', exist_ok=True)
    
    secrets_path = '.streamlit/secrets.toml'
    if not os.path.exists(secrets_path):
        with open(secrets_path, 'w') as f:
            f.write(secrets_content)
        logger.info(f"✅ Arquivo de exemplo criado: {secrets_path}")
        logger.warning("IMPORTANTE: Edite este arquivo com suas credenciais reais do Kaggle!")
    else:
        logger.info(f"ℹ️  Arquivo {secrets_path} já existe.")


def check_dependencies() -> bool:
    """Verifica se as dependências estão instaladas"""
    logger.info("📦 Verificando dependências...")
    
    required_packages = [
        'streamlit',
        'pandas',
        'sklearn',
        'joblib',
        'seaborn',
        'matplotlib',
        'kaggle'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            logger.info(f"✅ {package} instalado")
        except ImportError:
            logger.error(f"❌ {package} não encontrado")
            missing_packages.append(package)
    
    if missing_packages:
        logger.warning(f"Pacotes faltando: {', '.join(missing_packages)}")
        logger.info("Execute: pip install -r requirements.txt")
        return False
    
    return True


def check_kaggle_credentials() -> bool:
    """Verifica se as credenciais do Kaggle estão configuradas"""
    logger.info("🔑 Verificando credenciais do Kaggle...")
    
    secrets_path = '.streamlit/secrets.toml'
    if not os.path.exists(secrets_path):
        logger.error("❌ Arquivo de secrets não encontrado")
        return False
    
    with open(secrets_path, 'r') as f:
        content = f.read()
        
    if 'seu_username_aqui' in content or 'sua_chave_api_aqui' in content:
        logger.warning("Credenciais do Kaggle ainda não foram configuradas!")
        logger.info("   Edite o arquivo .streamlit/secrets.toml com suas credenciais reais")
        return False
    
    logger.info("✅ Arquivo de credenciais encontrado")
    return True


def check_model() -> bool:
    """Verifica se o modelo treinado existe"""
    logger.info("🤖 Verificando modelo treinado...")
    
    if os.path.exists('lol_win_predictor.joblib'):
        logger.info("✅ Modelo encontrado: lol_win_predictor.joblib")
        return True
    else:
        logger.error("❌ Modelo não encontrado")
        logger.info("   Execute: python train_model.py")
        return False


def main() -> None:
    """Função principal"""
    logger.info("=== 🎮 LoL Analysis - Verificação de Setup ===")
    
    # Cria arquivo de exemplo se necessário
    create_example_secrets()
    
    # Verifica dependências
    deps_ok = check_dependencies()
    
    # Verifica credenciais
    creds_ok = check_kaggle_credentials()
    
    # Verifica modelo
    model_ok = check_model()
    
    logger.info("\n" + "="*50)
    logger.info("📊 RESUMO DO STATUS:")
    logger.info("="*50)
    
    all_ok = deps_ok and creds_ok and model_ok
    
    if all_ok:
        logger.info("\n✅ Tudo pronto! Você pode executar:")
        logger.info("   streamlit run app.py")
    else:
        logger.warning("\n⚠️  Algumas configurações precisam ser feitas:")
        
        if not deps_ok:
            logger.info("\n1. Instale as dependências:")
            logger.info("   pip install -r requirements.txt")
        
        if not creds_ok:
            logger.info("\n2. Configure suas credenciais do Kaggle:")
            logger.info("   - Obtenha suas credenciais em: https://www.kaggle.com/account")
            logger.info("   - Edite o arquivo .streamlit/secrets.toml")
        
        if not model_ok:
            logger.info("\n3. Treine o modelo (após configurar as credenciais):")
            logger.info("   python train_model.py")
    
    logger.info("\n📚 Para mais informações, consulte o README.md")


if __name__ == "__main__":
    main()