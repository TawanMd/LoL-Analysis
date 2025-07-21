"""
Script para treinar o modelo de predição de vitória do League of Legends
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from data_manager import load_data


def prepare_features(df):
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
    
    print(f"Features disponíveis: {available_features}")
    
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


def train_model(X, y, feature_names):
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
    
    # Cria e treina o modelo
    print("Treinando o modelo Random Forest...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Faz predições
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)
    
    # Calcula métricas
    accuracy = accuracy_score(y_test, y_pred)
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
    
    print(f"\nAcurácia no conjunto de teste: {accuracy:.4f}")
    print(f"Acurácia média (cross-validation): {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Relatório de classificação
    print("\nRelatório de Classificação:")
    print(classification_report(y_test, y_pred, target_names=['Derrota', 'Vitória']))
    
    # Importância das features
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nImportância das Features:")
    print(feature_importance)
    
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
        'confusion_matrix': cm
    }
    
    return model, scaler, metrics


def save_model_components(model, scaler, feature_names, filename='lol_win_predictor.joblib'):
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
    print(f"Modelo salvo como '{filename}'")


def main():
    """Função principal para treinar e salvar o modelo"""
    
    print("=== Treinamento do Modelo de Predição LoL ===\n")
    
    # Carrega os dados
    print("Carregando dados...")
    df = load_data()
    print(f"Total de partidas carregadas: {len(df):,}")
    
    # Prepara as features
    X, y, feature_names = prepare_features(df)
    print(f"\nShape dos dados: X={X.shape}, y={y.shape}")
    print(f"Distribuição do target: {y.value_counts().to_dict()}")
    
    # Treina o modelo
    model, scaler, metrics = train_model(X, y, feature_names)
    
    # Salva o modelo e componentes
    print("\nSalvando o modelo...")
    save_model_components(model, scaler, feature_names)
    
    # Salva informações sobre o modelo
    model_info = {
        'features': feature_names,
        'accuracy': metrics['accuracy'],
        'cv_scores_mean': metrics['cv_scores'].mean(),
        'cv_scores_std': metrics['cv_scores'].std()
    }
    
    with open('model_info.txt', 'w') as f:
        f.write("=== Informações do Modelo ===\n\n")
        f.write(f"Features utilizadas: {', '.join(feature_names)}\n")
        f.write(f"Acurácia no teste: {model_info['accuracy']:.4f}\n")
        f.write(f"Acurácia CV: {model_info['cv_scores_mean']:.4f} (+/- {model_info['cv_scores_std'] * 2:.4f})\n")
        f.write("\nImportância das Features:\n")
        f.write(metrics['feature_importance'].to_string())
    
    print("\n✅ Treinamento concluído com sucesso!")
    print("📊 Gráficos salvos: feature_importance.png, confusion_matrix.png")
    print("📄 Informações do modelo salvas em: model_info.txt")
    
    # Teste rápido do modelo
    print("\n=== Teste Rápido do Modelo ===")
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
    
    print(f"\nExemplo de predição:")
    print(f"Entrada: {test_data.iloc[0].to_dict()}")
    print(f"Probabilidade de vitória do time azul: {prob[1]:.2%}")


if __name__ == "__main__":
    main()