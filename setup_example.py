"""
Script de exemplo para configurar e testar o projeto LoL Analysis
"""
import os
import json


def create_example_secrets():
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
        print(f"✅ Arquivo de exemplo criado: {secrets_path}")
        print("⚠️  IMPORTANTE: Edite este arquivo com suas credenciais reais do Kaggle!")
    else:
        print(f"ℹ️  Arquivo {secrets_path} já existe.")


def check_dependencies():
    """Verifica se as dependências estão instaladas"""
    print("\n📦 Verificando dependências...")
    
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
            print(f"✅ {package} instalado")
        except ImportError:
            print(f"❌ {package} não encontrado")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  Pacotes faltando: {', '.join(missing_packages)}")
        print("Execute: pip install -r requirements.txt")
        return False
    
    return True


def check_kaggle_credentials():
    """Verifica se as credenciais do Kaggle estão configuradas"""
    print("\n🔑 Verificando credenciais do Kaggle...")
    
    secrets_path = '.streamlit/secrets.toml'
    if not os.path.exists(secrets_path):
        print("❌ Arquivo de secrets não encontrado")
        return False
    
    with open(secrets_path, 'r') as f:
        content = f.read()
        
    if 'seu_username_aqui' in content or 'sua_chave_api_aqui' in content:
        print("⚠️  Credenciais do Kaggle ainda não foram configuradas!")
        print("   Edite o arquivo .streamlit/secrets.toml com suas credenciais reais")
        return False
    
    print("✅ Arquivo de credenciais encontrado")
    return True


def check_model():
    """Verifica se o modelo treinado existe"""
    print("\n🤖 Verificando modelo treinado...")
    
    if os.path.exists('lol_win_predictor.joblib'):
        print("✅ Modelo encontrado: lol_win_predictor.joblib")
        return True
    else:
        print("❌ Modelo não encontrado")
        print("   Execute: python train_model.py")
        return False


def main():
    """Função principal"""
    print("=== 🎮 LoL Analysis - Verificação de Setup ===\n")
    
    # Cria arquivo de exemplo se necessário
    create_example_secrets()
    
    # Verifica dependências
    deps_ok = check_dependencies()
    
    # Verifica credenciais
    creds_ok = check_kaggle_credentials()
    
    # Verifica modelo
    model_ok = check_model()
    
    print("\n" + "="*50)
    print("📊 RESUMO DO STATUS:")
    print("="*50)
    
    all_ok = deps_ok and creds_ok and model_ok
    
    if all_ok:
        print("\n✅ Tudo pronto! Você pode executar:")
        print("   streamlit run app.py")
    else:
        print("\n⚠️  Algumas configurações precisam ser feitas:")
        
        if not deps_ok:
            print("\n1. Instale as dependências:")
            print("   pip install -r requirements.txt")
        
        if not creds_ok:
            print("\n2. Configure suas credenciais do Kaggle:")
            print("   - Obtenha suas credenciais em: https://www.kaggle.com/account")
            print("   - Edite o arquivo .streamlit/secrets.toml")
        
        if not model_ok:
            print("\n3. Treine o modelo (após configurar as credenciais):")
            print("   python train_model.py")
    
    print("\n📚 Para mais informações, consulte o README.md")


if __name__ == "__main__":
    main()