# League of Legends Analysis Dashboard 🎮

Um dashboard interativo para análise de partidas de League of Legends usando Streamlit e Machine Learning.

## 📋 Funcionalidades

- **Análise Exploratória**: Visualizações interativas dos dados de partidas Diamond
- **Simulador de Vitória**: Prevê a probabilidade de vitória baseado em estatísticas aos 10 minutos de jogo

## 🚀 Configuração Rápida

Execute o script de verificação para checar se tudo está configurado:

```bash
python setup_example.py
```

### 1. Instalar Dependências

```bash
pip install -r requirements.txt
```

### 2. Configurar Credenciais do Kaggle

1. Obtenha suas credenciais da API do Kaggle:
   - Acesse [https://www.kaggle.com/account](https://www.kaggle.com/account)
   - Role até "API" e clique em "Create New API Token"
   - Um arquivo `kaggle.json` será baixado

2. Configure o arquivo de secrets do Streamlit:
   - Edite o arquivo `.streamlit/secrets.toml`
   - Adicione suas credenciais do Kaggle

### 3. Treinar o Modelo

Execute o script de treinamento para criar o modelo:

```bash
python train_model.py
```

Isso irá:
- Baixar o dataset do Kaggle (se necessário)
- Treinar um modelo Random Forest
- Salvar o modelo como `lol_win_predictor.joblib`
- Gerar gráficos de análise do modelo

### 4. Executar a Aplicação

```bash
streamlit run app.py
```

## 📊 Dataset

O dashboard utiliza o dataset [League of Legends Diamond Ranked Games (10 min)](https://www.kaggle.com/datasets/bobbyscience/league-of-legends-diamond-ranked-games-10-min) do Kaggle.

## 🔧 Estrutura do Projeto

```
LoL-Analysis/
├── .streamlit/
│   └── secrets.toml      # Credenciais da API (não commitado)
├── app.py                # Aplicação principal Streamlit
├── data_manager.py       # Gerenciamento de dados e API Kaggle
├── train_model.py        # Script para treinar o modelo ML
├── setup_example.py      # Script de verificação e setup
├── requirements.txt      # Dependências do projeto
├── README.md            # Este arquivo
└── lol_win_predictor.joblib  # Modelo ML (gerado pelo train_model.py)
```

## ⚠️ Notas Importantes

- O arquivo `secrets.toml` contém credenciais sensíveis e não deve ser commitado no git
- O modelo `lol_win_predictor.joblib` deve ser compatível com as features usadas no simulador
- O dataset será baixado automaticamente na primeira execução

## 🤝 Contribuições

Sinta-se à vontade para abrir issues ou pull requests para melhorias!