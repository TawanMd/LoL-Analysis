"""
Aplicação Streamlit para análise de dados de League of Legends
"""
import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
from data_manager import load_data
from typing import Dict, Any

# Configuração da página
st.set_page_config(
    page_title="LoL Analysis Dashboard",
    page_icon="🎮",
    layout="wide"
)

@st.cache_resource
def load_model() -> Dict[str, Any]:
    """
    Carrega o modelo pré-treinado usando cache para otimizar performance.
    
    Returns:
        dict: Dicionário contendo modelo, scaler e feature_names
    """
    model_data = joblib.load('lol_win_predictor.joblib')
    return model_data


def main() -> None:
    """Função principal da aplicação"""
    
    # Título principal
    st.title("🎮 League of Legends - Análise de Partidas Diamond")
    st.markdown("---")
    
    # Carrega os dados
    try:
        df = load_data()
        st.success(f"✅ Dados carregados com sucesso! Total de partidas: {len(df):,}")
    except Exception as e:
        st.error(f"❌ Erro ao carregar os dados: {str(e)}")
        st.stop()
    
    # Carrega o modelo
    try:
        model_data = load_model()
        model = model_data['model']
        scaler = model_data['scaler']
        feature_names = model_data['feature_names']
    except Exception as e:
        st.warning(f"⚠️ Modelo não encontrado: {str(e)}")
        model_data = None
        model = None
    
    # Cria as abas
    tab1, tab2 = st.tabs(["📊 Análise Exploratória", "🎯 Simulador de Vitória"])
    
    # Aba 1: Análise Exploratória
    with tab1:
        st.header("Análise Exploratória dos Dados")
        st.markdown("""
        Explore as métricas chave que influenciam a vitória no League of Legends.
        Analise como a diferença de ouro e o first blood impactam a probabilidade de vitória do Time Azul.
        """)
        
        # Layout de três colunas
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Probabilidade de Vitória por Diferença de Ouro")

            # Define bins for gold difference
            bins = [-float('inf'), -2000, 0, 2000, float('inf')]
            labels = ['< -2000', '-2000 a 0', '0 a 2000', '> 2000']
            df['gold_diff_bin'] = pd.cut(df['blueGoldDiff'], bins=bins, labels=labels, right=False)

            # Calculate win rates for each bin
            gold_diff_stats = df.groupby(df['gold_diff_bin'].astype(str))['blueWins'].agg(['count', lambda x: (x == 1).sum()])
            gold_diff_stats.columns = ['total_games', 'wins']
            gold_diff_stats['win_rate'] = gold_diff_stats['wins'] / gold_diff_stats['total_games']

            # Create the barplot of win probability with Plotly
            fig1 = px.bar(
                gold_diff_stats,
                x=gold_diff_stats.index,
                y='win_rate',
                title='Probabilidade de Vitória com Base na Diferença de Ouro',
                labels={'win_rate': 'Probabilidade de Vitória', 'x': 'Diferença de Ouro (Azul - Vermelho) aos 10 min'},
                text_auto=True,
                color_discrete_sequence=px.colors.sequential.Viridis
            )
            fig1.update_layout(yaxis_tickformat='.0%')
            st.plotly_chart(fig1, use_container_width=True)
            
            # Insights
            st.info("💡 **Insight**: Times com vantagem de ouro aos 10 minutos têm maior probabilidade de vitória.")
        
        with col2:
            st.subheader("Probabilidade de Vitória com First Blood")

            # Calcula a taxa de vitória com base no First Blood
            first_blood_stats = df.groupby('blueFirstBlood')['blueWins'].agg(['count', lambda x: (x == 1).sum()])
            first_blood_stats.columns = ['total_games', 'wins']
            first_blood_stats['win_rate'] = first_blood_stats['wins'] / first_blood_stats['total_games']

            # Create the barplot of win probability with Plotly
            fig2 = px.bar(
                first_blood_stats,
                x=first_blood_stats.index,
                y='win_rate',
                title='Probabilidade de Vitória com Base no First Blood',
                labels={'win_rate': 'Probabilidade de Vitória', 'blueFirstBlood': 'Time Azul Conseguiu First Blood'},
                color_discrete_sequence=px.colors.sequential.Viridis
            )
            fig2.update_xaxes(ticktext=['Não', 'Sim'], tickvals=[0, 1])
            fig2.update_layout(yaxis_tickformat='.0%')
            st.plotly_chart(fig2, use_container_width=True)
            
            # Insights
            st.info("💡 **Insight**: Conseguir o First Blood aumenta as chances de vitória, mas não é determinante.")
        
        with col3:
            st.subheader("Importância das Features para a Vitória")
            
            # Calcula a importância das features
            feature_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)

            # Cria o gráfico de barras de importância das features com Plotly
            fig3 = px.bar(
                feature_importance,
                x='importance',
                y='feature',
                orientation='h',
                title='Importância de Cada Métrica para a Predição',
                labels={'importance': 'Importância', 'feature': 'Métrica'},
                color_discrete_sequence=px.colors.sequential.Viridis
            )
            fig3.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig3, use_container_width=True)
            
            st.info("💡 **Insight**: A diferença de ouro e experiência são os fatores mais decisivos para a vitória.")
    
    # Aba 2: Simulador de Vitória
    with tab2:
        st.header("🎯 Simulador de Probabilidade de Vitória")
        
        if model is None:
            st.error("❌ Modelo não carregado. Por favor, certifique-se de que o arquivo 'lol_win_predictor.joblib' está presente.")
            st.stop()
        
        st.markdown("Configure os parâmetros da partida aos 10 minutos para prever a probabilidade de vitória do Time Azul:")
        
        # Layout de duas colunas para os inputs
        col1, col2 = st.columns(2)
        
        with col1:
            # Slider para diferença de ouro
            gold_diff = st.slider(
                'Diferença de Ouro (Azul - Vermelho)',
                min_value=-10000,
                max_value=10000,
                value=0,
                step=100,
                help="Valores positivos indicam vantagem do time azul"
            )
            
            # Slider para diferença de experiência
            exp_diff = st.slider(
                'Diferença de Experiência (Azul - Vermelho)',
                min_value=-10000,
                max_value=10000,
                value=0,
                step=100,
                help="Valores positivos indicam vantagem do time azul"
            )
        
        with col2:
            # Selectbox para first blood
            first_blood = st.selectbox(
                'Quem conseguiu o First Blood?',
                options=['Time Vermelho', 'Time Azul'],
                index=0
            )
            first_blood_value = 1 if first_blood == 'Time Azul' else 0
            
            # Number input para abates
            blue_kills = st.number_input(
                'Abates do Time Azul',
                min_value=0,
                max_value=50,
                value=5,
                step=1
            )
            
            # Radio para dragões
            blue_dragons = st.radio(
                'Dragões do Time Azul',
                options=[0, 1, 2],
                horizontal=True
            )
        
        # Botão para calcular
        if st.button('🔮 Calcular Probabilidade', type='primary'):
            # Cria DataFrame com os inputs do usuário
            user_data = pd.DataFrame({
                'blueGoldDiff': [gold_diff],
                'blueExperienceDiff': [exp_diff],
                'blueFirstBlood': [first_blood_value],
                'blueKills': [blue_kills],
                'blueDragons': [blue_dragons]
            })
            
            # Adiciona features faltantes com valor 0
            for feature in feature_names:
                if feature not in user_data.columns:
                    user_data[feature] = 0
            
            # Reordena as colunas para corresponder ao modelo
            user_data = user_data[feature_names]
            
            st.info(f"📋 Features utilizadas pelo modelo: {', '.join(feature_names)}")
            
            try:
                # Aplica o scaler e faz a predição
                user_data_scaled = scaler.transform(user_data)
                probability = model.predict_proba(user_data_scaled)[0]
                win_prob = probability[1]  # Probabilidade da classe 1 (vitória)
                
                # Exibe os resultados
                st.markdown("---")
                st.subheader("📊 Resultado da Predição")
                
                # Métricas
                col1, col2, col3 = st.columns([1, 2, 1])
                
                with col2:
                    # Métrica principal
                    st.metric(
                        label="Probabilidade de Vitória do Time Azul",
                        value=f"{win_prob:.1%}",
                        delta=f"{win_prob - 0.5:.1%} vs 50%"
                    )
                    
                    # Barra de progresso
                    st.progress(win_prob)
                    
                    # Interpretação
                    if win_prob > 0.7:
                        st.success("✅ **Alta probabilidade de vitória!** O Time Azul está em grande vantagem.")
                    elif win_prob > 0.5:
                        st.info("📈 **Vantagem para o Time Azul.** A partida está favorável.")
                    elif win_prob > 0.3:
                        st.warning("⚠️ **Desvantagem para o Time Azul.** A situação está difícil.")
                    else:
                        st.error("❌ **Baixa probabilidade de vitória!** O Time Azul está em grande desvantagem.")
                
            except Exception as e:
                st.error(f"❌ Erro ao fazer a predição: {str(e)}")
                st.info("💡 Certifique-se de que o modelo foi treinado com as features corretas.")


if __name__ == "__main__":
    main()