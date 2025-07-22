"""
Script para treinar o modelo de predição de vitória do League of Legends
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from data_manager import load_data
from typing import List, Tuple, Dict, Any
from utils import get_logger

# Initialize logger
logger = get_logger(__name__)


def prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    Prepara as features para o modelo baseado nos dados disponíveis.
    
    Args:
        df: DataFrame com os dados do jogo
        
    Returns:
        X: Features para treino
        y: Target (blueWins)
        feature_names: Lista com nomes das features
    """
    # Features relacionadas ao time azul que são boas preditoras aos 10 minutos
    feature_columns = [
        'blueGoldDiff',           # Diferença de ouro
        'blueExperienceDiff',     # Diferença de experiência
        'blueKills',              # Abates do time azul
        'blueDeaths',             # Mortes do time azul
        'blueAssists',            # Assistências do time azul
        'blueDragons',            # Dragões do time azul
        'blueHeralds',            # Arautos do time azul
        'blueTowersDestroyed',    # Torres destruídas pelo time azul
        'blueFirstBlood',         # First blood do time azul
        'blueCSPerMin',           # CS por minuto do time azul
        'blueGoldPerMin',         # Ouro por minuto do time azul
    ]
    
    # Verifica quais features estão disponíveis no dataset
    available_features = [col for col in feature_columns if col in df.columns]
    
    logger.info(f"Features disponíveis: {available_features}")
    
    # Se não tivermos todas as features esperadas, vamos usar as básicas
    if len(available_features) < 5:
        # Features mínimas que sabemos que existem
        available_features = [
            'blueGoldDiff',
            'blueExperienceDiff',
            'blueKills',
            'blueDragons',
            'blueFirstBlood'
        ]
    
    X = df[available_features].copy()
    y = df['blueWins'].copy()
    
    return X, y, available_features


def train_model(X: pd.DataFrame, y: pd.Series, feature_names: List[str]) -> Tuple[RandomForestClassifier, StandardScaler, Dict[str, Any]]:
    """
    Treina um modelo Random Forest para prever vitórias.
    
    Args:
        X: Features
        y: Target
        feature_names: Nomes das features
        
    Returns:
        model: Modelo treinado
        scaler: Scaler usado para normalização
        metrics: Métricas de avaliação
    """
    # Divide os dados em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Normaliza as features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define o grid de parâmetros para o GridSearchCV
    param_grid = {
        'n_estimators': [100, 150],
        'max_depth': [10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }

    # Cria o modelo
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    
    # Instancia o GridSearchCV
    logger.info("Iniciando a busca de hiperparâmetros com GridSearchCV...")
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2, scoring='accuracy')
    
    # Treina o modelo com GridSearchCV
    grid_search.fit(X_train_scaled, y_train)
    
    # Pega o melhor modelo
    model = grid_search.best_estimator_
    logger.info(f"Melhores parâmetros encontrados: {grid_search.best_params_}")
    
    # Faz predições
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)
    
    # Calcula métricas
    accuracy = accuracy_score(y_test, y_pred)
    
    # Cross-validation com o melhor modelo
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
    
    logger.info(f"Acurácia no conjunto de teste com o melhor modelo: {accuracy:.4f}")
    logger.info(f"Acurácia média (cross-validation) com o melhor modelo: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Relatório de classificação
    logger.info("Relatório de Classificação:\n" + classification_report(y_test, y_pred, target_names=['Derrota', 'Vitória']))
    
    # Importância das features
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    logger.info("Importância das Features:\n" + feature_importance.to_string())
    
    # Plota a importância das features
    plt.figure(figsize=(10, 6))
    sns.barplot(data=feature_importance, x='importance', y='feature')
    plt.title('Importância das Features no Modelo')
    plt.xlabel('Importância')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()
    
    # Matriz de confusão
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Matriz de Confusão')
    plt.ylabel('Valor Real')
    plt.xlabel('Valor Predito')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    metrics = {
        'accuracy': accuracy,
        'cv_scores': cv_scores,
        'feature_importance': feature_importance,
        'confusion_matrix': cm,
        'best_params': grid_search.best_params_
    }
    
    return model, scaler, metrics


def save_model_components(model: RandomForestClassifier, scaler: StandardScaler, feature_names: List[str], filename: str = 'lol_win_predictor.joblib') -> None:
    """
    Salva o modelo, scaler e nomes das features em um único arquivo.
    """
    # Cria um dicionário com todos os componentes
    model_data = {
        'model': model,
        'scaler': scaler,
        'feature_names': feature_names,
        'version': '1.0'
    }
    
    # Salva tudo em um único arquivo
    joblib.dump(model_data, filename)
    logger.info(f"Modelo salvo como '{filename}'")


def main() -> None:
    """Função principal para treinar e salvar o modelo"""
    
    logger.info("=== Treinamento do Modelo de Predição LoL ===")
    
    # Carrega os dados
    logger.info("Carregando dados...")
    df = load_data()
    logger.info(f"Total de partidas carregadas: {len(df):,}")
    
    # Prepara as features
    X, y, feature_names = prepare_features(df)
    logger.info(f"Shape dos dados: X={X.shape}, y={y.shape}")
    logger.info(f"Distribuição do target: {y.value_counts().to_dict()}")
    
    # Treina o modelo
    model, scaler, metrics = train_model(X, y, feature_names)
    
    # Salva o modelo e componentes
    logger.info("Salvando o modelo...")
    save_model_components(model, scaler, feature_names)
    
    # Salva informações sobre o modelo
    model_info = {
        'features': feature_names,
        'accuracy': metrics['accuracy'],
        'cv_scores_mean': metrics['cv_scores'].mean(),
        'cv_scores_std': metrics['cv_scores'].std(),
        'best_params': metrics['best_params']
    }
    
    with open('model_info.txt', 'w') as f:
        f.write("=== Informações do Modelo ===\n\n")
        f.write(f"Melhores Parâmetros: {model_info['best_params']}\n\n")
        f.write(f"Features utilizadas: {', '.join(feature_names)}\n")
        f.write(f"Acurácia no teste: {model_info['accuracy']:.4f}\n")
        f.write(f"Acurácia CV: {model_info['cv_scores_mean']:.4f} (+/- {model_info['cv_scores_std'] * 2:.4f})\n")
        f.write("\nImportância das Features:\n")
        f.write(metrics['feature_importance'].to_string())
    
    logger.info("✅ Treinamento concluído com sucesso!")
    logger.info("📊 Gráficos salvos: feature_importance.png, confusion_matrix.png")
    logger.info("📄 Informações do modelo salvas em: model_info.txt")
    
    # Teste rápido do modelo
    logger.info("=== Teste Rápido do Modelo ===")
    test_data = pd.DataFrame({
        'blueGoldDiff': [1000],
        'blueExperienceDiff': [500],
        'blueKills': [5],
        'blueDragons': [1],
        'blueFirstBlood': [1]
    })
    
    # Adiciona features faltantes se necessário
    for feature in feature_names:
        if feature not in test_data.columns:
            test_data[feature] = 0
    
    # Reordena as colunas para corresponder ao treino
    test_data = test_data[feature_names]
    
    # Faz a predição usando o scaler e o modelo
    test_data_scaled = scaler.transform(test_data)
    prob = model.predict_proba(test_data_scaled)[0]
    
    logger.info("Exemplo de predição:")
    logger.info(f"Entrada: {test_data.iloc[0].to_dict()}")
    logger.info(f"Probabilidade de vitória do time azul: {prob[1]:.2%}")


if __name__ == "__main__":
    main()