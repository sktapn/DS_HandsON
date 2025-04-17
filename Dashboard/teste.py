import streamlit as st
import pandas as pd
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import plotly.express as px
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier

# Configuração da Página e CSS Customizado
st.set_page_config(
    page_title="Análise de Desnutrição Infantil no Brasil",
    page_icon="👶",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* Variáveis de cores - Tema Pastel Elegante */
    :root {
        --primary: #3b82f6; /* Azul pastel */
        --primary-light: #dbeafe; /* Azul claro para fundos */
        --secondary: #10B981; /* Verde pastel */
        --secondary-light: #D1FAE5; /* Verde claro para destaques */
        --purple: #8B5CF6; /* Roxo pastel */
        --purple-light: #EDE9FE; /* Roxo claro para fundos */
        --background: linear-gradient(135deg, #F9FAFB 0%, #EFF6FF 100%); /* Gradiente de fundo */
        --card: #FFFFFF; /* Fundo dos cards */
        --border: #E5E7EB; /* Borda suave */
        --text-dark: #1F2937; /* Texto principal */
        --text-medium: #4B5563; /* Texto secundário */
    }

    /* Estilo geral da página */
    .main {
        background: var(--background);
        padding: 2rem;
    }

    /* Cards de métricas */
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        transition: transform 0.2s, box-shadow 0.2s;
        border: 1px solid var(--border);
    }
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.08);
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: bold;
        color: var(--primary);
    }

    /* Cabeçalhos */
    .main-header {
        font-size: 2.2rem;
        color: var(--primary);
        font-weight: 700;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid var(--primary-light);
    }
    .sub-header {
        font-size: 1.5rem;
        color: var(--primary);
        margin: 2rem 0 1rem;
        font-weight: 600;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        padding-bottom: 0.3rem;
        border-bottom: 2px solid var(--primary-light);
    }

    /* Tabelas */
    .dataframe {
        width: 100%;
        border-collapse: collapse;
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.05);
    }
    .dataframe th {
        background: var(--primary-light);
        color: var(--primary);
        padding: 0.75rem 1rem;
        text-align: left;
        font-weight: 600;
    }
    .dataframe td {
        padding: 0.75rem 1rem;
        border-top: 1px solid var(--border);
    }
    .dataframe tr:nth-child(even) {
        background-color: #F9FAFB;
    }

    /* Cards de insights */
    .insight-card {
        background: linear-gradient(135deg, var(--primary-light), white);
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        border-left: 4px solid var(--primary);
    }
    .insight-card.green {
        background: linear-gradient(135deg, var(--secondary-light), white);
        border-left: 4px solid var(--secondary);
    }
    .insight-card.purple {
        background: linear-gradient(135deg, var(--purple-light), white);
        border-left: 4px solid var(--purple);
    }
    .insight-title {
        font-size: 1.2rem;
        font-weight: 600;
        color: var(--text-dark);
        margin-bottom: 0.5rem;
    }

    /* Botão Fazer Análise */
    .stButton>button {
        background-color: var(--primary);
        color: white;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: background-color 0.2s;
    }
    .stButton>button:hover {
        background-color: #2563eb;
    }
</style>
""", unsafe_allow_html=True)

# Ícones e paleta de cores para o tema
nutrition_icons = {
    "main": "👶",
    "nutrition": "🍎",
    "development": "📏",
    "infrastructure": "🏠",
    "socioeconomic": "👨‍👩‍👧‍👦",
    "warning": "⚠️",
    "success": "✅",
    "info": "ℹ️",
    "health": "❤️",
    "education": "📚",
    "regional": "🗺️",
    "prediction": "🔍"
}
nutrition_palette = ["#1E3A8A", "#10B981", "#6B7280", "#9d4edd", "#ef476f", "#059669"]

# Cabeçalho e Introdução
st.markdown("""
<div class="main-header">
    <span></span>
    <div>Análise de Desnutrição Infantil no Brasil</div>
</div>
<div class="section">
    <p style="font-size: 1rem; color: var(--text-medium);">
        Atualização: 03/04/2025 15:00 | Fonte: Secretarias Estaduais de Saúde, Brasil 2025
    </p>
    <p style="font-size: 1.1rem; line-height: 1.6; color: var(--text-medium);">
        Esta plataforma oferece uma análise detalhada da desnutrição infantil no Brasil, explorando padrões regionais, fatores socioeconômicos e condições de infraestrutura que influenciam o bem-estar e o desenvolvimento das crianças brasileiras.
    </p>
</div>
""", unsafe_allow_html=True)

# Barra Horizontal para Controles
st.markdown('<div class="control-bar">', unsafe_allow_html=True)
col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 1])

# Estado para controlar os popups e predição
if 'show_csv_popup' not in st.session_state:
    st.session_state.show_csv_popup = False
if 'show_regioes_popup' not in st.session_state:
    st.session_state.show_regioes_popup = False
if 'show_alimentos_popup' not in st.session_state:
    st.session_state.show_alimentos_popup = False
if 'show_domicilio_popup' not in st.session_state:
    st.session_state.show_domicilio_popup = False
if 'show_prediction' not in st.session_state:
    st.session_state.show_prediction = False

# Estado para armazenar os valores selecionados
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None
if 'regioes' not in st.session_state:
    st.session_state.regioes = ['Norte']
if 'acesso_alimentos' not in st.session_state:
    st.session_state.acesso_alimentos = "Todos"
if 'tipo_domicilio' not in st.session_state:
    st.session_state.tipo_domicilio = ['Casa', 'Apartamento']

# Widget "Adicionar CSV"
with col1:
    st.markdown("""
    <div class="custom-widget" onclick="document.getElementById('csv-widget').click()">
        <span class="custom-widget-icon">📂</span>
        <div>
            <div class="custom-widget-text">Adicionar CSV</div>
            <div class="custom-widget-subtext">Limite 200MB por arquivo • CSV</div>
        </div>
    </div>
    <input type="hidden" id="csv-widget" />
    """, unsafe_allow_html=True)
    if not st.session_state.show_csv_popup:
        st.session_state.show_csv_popup = True
    if st.session_state.show_csv_popup:
        with st.container():
            st.markdown('<div class="popup-container">', unsafe_allow_html=True)
            uploaded_file = st.file_uploader("", type=["csv"], label_visibility="collapsed")
            if uploaded_file:
                st.session_state.uploaded_file = uploaded_file
                st.markdown(f'<p style="color: var(--text-medium); font-size: 0.9rem;">{uploaded_file.name} ({uploaded_file.size / 1024:.1f}KB) <span style="color: var(--primary); cursor: pointer;" onclick="document.getElementById(\'csv-clear\').click()">✖</span></p>', unsafe_allow_html=True)
                if st.button("", key="csv-clear", on_click=lambda: st.session_state.update({'uploaded_file': None})):
                    pass
            if st.button("Confirmar", key="confirm_csv"):
                st.session_state.show_csv_popup = False
            st.markdown('</div>', unsafe_allow_html=True)

# Widget "Escolher Regiões"
with col2:
    st.markdown("""
    <div class="custom-widget" onclick="document.getElementById('regioes-widget').click()">
        <span class="custom-widget-icon">🗺️</span>
        <div>
            <div class="custom-widget-text">Escolher Regiões</div>
            <div class="custom-widget-subtext">Selecione as regiões para análise</div>
        </div>
    </div>
    <input type="hidden" id="regioes-widget" />
    """, unsafe_allow_html=True)
    if not st.session_state.show_regioes_popup:
        st.session_state.show_regioes_popup = True
    if st.session_state.show_regioes_popup:
        with st.container():
            st.markdown('<div class="popup-container">', unsafe_allow_html=True)
            regioes = st.multiselect(
                "",
                options=['Norte', 'Sul', 'Sudeste', 'Centro-Oeste', 'Nordeste'],
                default=st.session_state.regioes,
                label_visibility="collapsed"
            )
            if regioes:
                st.session_state.regioes = regioes
            if st.button("Confirmar", key="confirm_regioes"):
                st.session_state.show_regioes_popup = False
            st.markdown('</div>', unsafe_allow_html=True)

# Widget "Acesso a Alimentos"
with col3:
    st.markdown("""
    <div class="custom-widget" onclick="document.getElementById('alimentos-widget').click()">
        <span class="custom-widget-icon">🍎</span>
        <div>
            <div class="custom-widget-text">Acesso a Alimentos</div>
            <div class="custom-widget-subtext">Filtre por acesso a alimentos básicos</div>
        </div>
    </div>
    <input type="hidden" id="alimentos-widget" />
    """, unsafe_allow_html=True)
    if not st.session_state.show_alimentos_popup:
        st.session_state.show_alimentos_popup = True
    if st.session_state.show_alimentos_popup:
        with st.container():
            st.markdown('<div class="popup-container">', unsafe_allow_html=True)
            acesso_alimentos = st.radio(
                "",
                options=["Todos", "Sim, sempre", "Sim, quase sempre", "Sim, às vezes"],
                index=["Todos", "Sim, sempre", "Sim, quase sempre", "Sim, às vezes"].index(st.session_state.acesso_alimentos),
                horizontal=True,
                label_visibility="collapsed"
            )
            st.session_state.acesso_alimentos = acesso_alimentos
            if st.button("Confirmar", key="confirm_alimentos"):
                st.session_state.show_alimentos_popup = False
            st.markdown('</div>', unsafe_allow_html=True)

# Widget "Tipo de Domicílio"
with col4:
    st.markdown("""
    <div class="custom-widget" onclick="document.getElementById('domicilio-widget').click()">
        <span class="custom-widget-icon">🏠</span>
        <div>
            <div class="custom-widget-text">Tipo de Domicílio</div>
            <div class="custom-widget-subtext">Selecione os tipos de domicílio</div>
        </div>
    </div>
    <input type="hidden" id="domicilio-widget" />
    """, unsafe_allow_html=True)
    if not st.session_state.show_domicilio_popup:
        st.session_state.show_domicilio_popup = True
    if st.session_state.show_domicilio_popup:
        with st.container():
            st.markdown('<div class="popup-container">', unsafe_allow_html=True)
            tipo_domicilio = st.multiselect(
                "",
                options=['Casa', 'Apartamento', 'Habitação em casa de cômodos'],
                default=st.session_state.tipo_domicilio,
                label_visibility="collapsed"
            )
            if tipo_domicilio:
                st.session_state.tipo_domicilio = tipo_domicilio
            if st.button("Confirmar", key="confirm_domicilio"):
                st.session_state.show_domicilio_popup = False
            st.markdown('</div>', unsafe_allow_html=True)

# Botão "Atualizar Análise"
with col5:
    st.markdown('<div style="margin-left: auto;">', unsafe_allow_html=True)
    if st.button("Atualizar Análise", key="update_btn"):
        st.success("Análise atualizada com sucesso! ✅")
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# Função para Carregar Dados (Cache)
@st.cache_data
def load_data(file):
    if file is not None:
        df = pd.read_csv(file)
    else:
        st.markdown("""
        <div class="section">
            <p style="color: var(--text-medium);">
                Nenhum arquivo carregado. Utilizando dados simulados para demonstração.
            </p>
        </div>
        """, unsafe_allow_html=True)
        df = pd.DataFrame({
            'Região': np.random.choice(['Norte', 'Sul', 'Sudeste', 'Centro-Oeste', 'Nordeste'], 100),
            'Sexo': np.random.choice(['Masculino', 'Feminino'], 100),
            'Idade': np.random.randint(0, 5, 100),
            'Idade em Meses': np.random.randint(0, 60, 100),
            'Moradores que Alimentaram Acabamento (Sim)': np.random.choice(['Sim', 'Não'], 100),
            'Moradores que Alimentaram Acabamento (Não)': np.random.choice(['Sim', 'Não'], 100),
            'Tipo de Domicílio': np.random.choice(['Casa', 'Apartamento', 'Habitação em casa de cômodos'], 100),
            'Possui Cozinha': np.random.choice(['Sim', 'Não'], 100),
            'Ocupação': np.random.choice(['Próprio de algum morador - já pago', 'Alugado', 'Cedido de outra forma'], 100),
            'Situação do Registro': np.random.choice(['Urbano'], 100),
            'Presença de Tosse': np.random.choice(['Sim', 'Não'], 100),
            'Tipo de Respiração': np.random.choice(['Sim', 'Não'], 100),
            'Alimentos Básicos': np.random.choice(['Sim, sempre', 'Sim, quase sempre', 'Sim, às vezes'], 100),
            'Nivel Escolaridade': np.random.choice(['1°ano do ensino médio', '3°ano do ensino médio', 'Ensino superior completo'], 100),
            'Beneficios': np.random.choice(['A', 'F', ''], 100),
            'Faixa de Renda': np.random.choice(['Até R$ 1.000,00', 'De R$ 1.001,00 até R$ 2.000,00', 'De R$ 2.001,00 até R$ 3.000,00'], 100),
            'Cor Pessoa': np.random.choice(['Branca', 'Parda (mulata, cabocla, cafuza, mameluca ou mestiça)'], 100)
        })
    return df

df = load_data(st.session_state.uploaded_file)

# Aplicando filtros
df = df[df['Região'].isin(st.session_state.regioes)]
df = df[df['Tipo de Domicílio'].isin(st.session_state.tipo_domicilio)]
if st.session_state.acesso_alimentos != "Todos":
    df = df[df['Alimentos Básicos'] == st.session_state.acesso_alimentos]
df['Idade em Meses'] = df['Idade em Meses'].astype(str).str.replace(' meses', '', regex=False).str.strip()
df['Idade em Meses'] = pd.to_numeric(df['Idade em Meses'], errors='coerce')

# Pré-processamento da coluna "Idade em Meses"
df['Idade em Meses'] = df['Idade em Meses'].astype(str).str.replace(' meses', '', regex=False).str.strip()
df['Idade em Meses'] = pd.to_numeric(df['Idade em Meses'], errors='coerce')

# Geração de Scores e Indicadores
def recode_alimentos(valor):
    if isinstance(valor, str):
        if "Sim, sempre" in valor:
            return 1.0
        elif "Sim, quase sempre" in valor:
            return 0.5
    return 0.0

def recode_tosse(valor):
    if isinstance(valor, str):
        val = valor.strip().lower()
        if val == "não":
            return 1.0
        elif val == "sim":
            return 0.0
    return 0.0

def recode_cozinha(valor):
    if isinstance(valor, str):
        val = valor.strip().lower()
        if val == "sim":
            return 1.0
        elif val == "não":
            return 0.0
    return 0.0

df['alimentos_score'] = df['Alimentos Básicos'].apply(recode_alimentos)
df['tosse_score'] = df['Presença de Tosse'].apply(recode_tosse)
df['cozinha_score'] = df['Possui Cozinha'].apply(recode_cozinha)
df['indice_desenvolvimento'] = df[['alimentos_score', 'tosse_score', 'cozinha_score']].mean(axis=1)

# Criação de df_final para a Comparação por Dimensão
resultados = []
df_regiao_group = df.groupby('Região')['indice_desenvolvimento'].mean().reset_index()
df_regiao_group['dimensao'] = 'Região'
df_regiao_group.rename(columns={'Região': 'categoria', 'indice_desenvolvimento': 'indice_medio'}, inplace=True)
resultados.append(df_regiao_group)
df_escolaridade_group = df.groupby('Nivel Escolaridade')['indice_desenvolvimento'].mean().reset_index()
df_escolaridade_group['dimensao'] = 'Nível Escolaridade'
df_escolaridade_group.rename(columns={'Nivel Escolaridade': 'categoria', 'indice_desenvolvimento': 'indice_medio'}, inplace=True)
resultados.append(df_escolaridade_group)
df_renda_group = df.groupby('Faixa de Renda')['indice_desenvolvimento'].mean().reset_index()
df_renda_group['dimensao'] = 'Faixa de Renda'
df_renda_group.rename(columns={'Faixa de Renda': 'categoria', 'indice_desenvolvimento': 'indice_medio'}, inplace=True)
resultados.append(df_renda_group)
df_cor_group = df.groupby('Cor Pessoa')['indice_desenvolvimento'].mean().reset_index()
df_cor_group['dimensao'] = 'Cor Pessoa'
df_cor_group.rename(columns={'Cor Pessoa': 'categoria', 'indice_desenvolvimento': 'indice_medio'}, inplace=True)
resultados.append(df_cor_group)
df_final = pd.concat(resultados, ignore_index=True)

# Cards de Métricas no Topo
st.markdown('<div class="sub-header"><span>📊</span> Indicadores Gerais</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header-description">Resumo das principais métricas sobre desnutrição infantil no Brasil, incluindo número de crianças analisadas, índice de desenvolvimento e acesso a alimentos.</div>', unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)
total_criancas = len(df)
indice_medio = df['indice_desenvolvimento'].mean()
acesso_alimentos_sim = len(df[df['alimentos_score'] == 1.0])

with col1:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">Total de Crianças Analisadas</div>
        <div class="metric-value">{}</div>
    </div>
    """.format(total_criancas), unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">Índice Médio de Desenvolvimento</div>
        <div class="metric-value">{:.2f}</div>
    </div>
    """.format(indice_medio), unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">Crianças com Acesso a Alimentos (Sim, sempre)</div>
        <div class="metric-value">{}</div>
        <div class="metric-subvalue">{:.1f}%</div>
    </div>
    """.format(acesso_alimentos_sim, (acesso_alimentos_sim / total_criancas) * 100 if total_criancas > 0 else 0), unsafe_allow_html=True)

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# Tabela de Indicadores por Região
st.markdown('<div class="sub-header"><span>🗺️</span> Índice de Desenvolvimento por Região</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header-description">Tabela detalhada com os indicadores de desenvolvimento infantil por região, incluindo acesso a alimentos, infraestrutura e saúde.</div>', unsafe_allow_html=True)
st.markdown('<div class="section">', unsafe_allow_html=True)
tabela_dados = df.groupby('Região').agg({
    'indice_desenvolvimento': 'mean',
    'alimentos_score': 'mean',
    'cozinha_score': 'mean',
    'tosse_score': 'mean'
}).reset_index()
tabela_dados.columns = ['Região', 'Índice de Desenvolvimento', 'Acesso a Alimentos', 'Infraestrutura (Cozinha)', 'Saúde (Ausência de Tosse)']
tabela_dados = tabela_dados.round(2)
brasil_row = pd.DataFrame([[
    'Brasil',
    df['indice_desenvolvimento'].mean(),
    df['alimentos_score'].mean(),
    df['cozinha_score'].mean(),
    df['tosse_score'].mean()
]], columns=['Região', 'Índice de Desenvolvimento', 'Acesso a Alimentos', 'Infraestrutura (Cozinha)', 'Saúde (Ausência de Tosse)'])
tabela_dados = pd.concat([brasil_row, tabela_dados], ignore_index=True)
st.dataframe(tabela_dados, use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# Análise Regional
st.markdown('<div class="sub-header"><span>🗺️</span> Análise Regional</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header-description">Visualizações que exploram as diferenças regionais no desenvolvimento infantil, com gráficos e mapas de calor por região.</div>', unsafe_allow_html=True)
st.markdown('<div class="section">', unsafe_allow_html=True)
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Índice de Desenvolvimento por Região")
    grouped_df = df.groupby('Região')['indice_desenvolvimento'].mean().reset_index()
    if len(grouped_df) == 1:
        fig = px.bar(
            grouped_df,
            x='Região',
            y='indice_desenvolvimento',
            color='Região',
            color_discrete_sequence=nutrition_palette,
            title=""
        )
        fig.update_traces(width=0.4)
        fig.update_layout(
            xaxis=dict(range=[-0.5, 0.5]),
            height=400,
            width=500,
            xaxis_title="Região",
            yaxis_title="Índice de Desenvolvimento",
            font=dict(family="Roboto", size=12),
            plot_bgcolor="white",
            margin=dict(l=20, r=20, t=40, b=20)
        )
    else:
        fig = px.bar(
            grouped_df,
            x='Região',
            y='indice_desenvolvimento',
            color='Região',
            color_discrete_sequence=nutrition_palette,
            title=""
        )
        fig.update_layout(
            xaxis_title="Região",
            yaxis_title="Índice de Desenvolvimento",
            font=dict(family="Roboto", size=12),
            plot_bgcolor="white",
            margin=dict(l=20, r=20, t=40, b=20)
        )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown("#### Mapa de Calor: Indicadores por Região")
    heatmap_data = pd.DataFrame({
        'Região': df['Região'].unique()
    })
    heatmap_data['Índice de Desenvolvimento'] = [df[df['Região'] == r]['indice_desenvolvimento'].mean() for r in heatmap_data['Região']]
    heatmap_data['Acesso a Alimentos'] = [df[df['Região'] == r]['alimentos_score'].mean() for r in heatmap_data['Região']]
    heatmap_data['Infraestrutura'] = [df[df['Região'] == r]['cozinha_score'].mean() for r in heatmap_data['Região']]
    fig_heat = px.imshow(
        heatmap_data.set_index('Região')[['Índice de Desenvolvimento', 'Acesso a Alimentos', 'Infraestrutura']],
        text_auto='.2f',
        aspect="auto",
        color_continuous_scale=px.colors.sequential.Blues
    )
    fig_heat.update_layout(
        title="",
        xaxis_title="Indicador",
        yaxis_title="Região",
        font=dict(family="Roboto", size=12)
    )
    st.plotly_chart(fig_heat, use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# Determinantes Socioeconômicos
st.markdown('<div class="sub-header"><span>👨‍👩‍👧‍👦</span> Determinantes Socioeconômicos</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header-description">Análise dos fatores socioeconômicos que impactam o desenvolvimento infantil, como renda, ocupação e escolaridade.</div>', unsafe_allow_html=True)
st.markdown('<div class="section">', unsafe_allow_html=True)
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Índice de Desenvolvimento por Ocupação")
    fig = px.box(
        df,
        x='Ocupação',
        y='indice_desenvolvimento',
        color='Ocupação',
        color_discrete_sequence=nutrition_palette,
        title=""
    )
    fig.update_layout(
        xaxis_title="Ocupação",
        yaxis_title="Índice de Desenvolvimento",
        font=dict(family="Roboto", size=12),
        plot_bgcolor="white",
        showlegend=False,
        xaxis={'visible': False}
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown("#### Acesso a Alimentos por Faixa de Renda")
    fig = px.histogram(
        df,
        x='Faixa de Renda',
        color='Alimentos Básicos',
        barmode='group',
        color_discrete_sequence=nutrition_palette,
        title=""
    )
    fig.update_layout(
        xaxis_title="Faixa de Renda",
        yaxis_title="Contagem",
        font=dict(family="Roboto", size=12),
        plot_bgcolor="white",
        xaxis={'categoryorder': 'total descending', 'tickangle': -45}
    )
    st.plotly_chart(fig, use_container_width=True)

st.markdown("#### Correlação entre Fatores Socioeconômicos e Desenvolvimento")
corr_cols = ['indice_desenvolvimento', 'alimentos_score', 'cozinha_score', 'tosse_score']
corr_data = df[corr_cols].corr()
corr_data.columns = ['Desenvolvimento', 'Alimentos', 'Cozinha', 'Tosse']
corr_data.index = ['Desenvolvimento', 'Alimentos', 'Cozinha', 'Tosse']
fig_corr = px.imshow(
    corr_data,
    text_auto='.2f',
    color_continuous_scale=px.colors.diverging.RdBu_r,
    zmin=-1, zmax=1
)
fig_corr.update_layout(
    title="",
    font=dict(family="Roboto", size=12)
)
st.plotly_chart(fig_corr, use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# Infraestrutura e Nutrição
st.markdown('<div class="sub-header"><span>🏠</span> Infraestrutura e Nutrição</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header-description">Exploração da relação entre infraestrutura domiciliar, acesso a alimentos e o desenvolvimento infantil, com foco em faixas etárias e tipos de domicílio.</div>', unsafe_allow_html=True)
st.markdown('<div class="section">', unsafe_allow_html=True)
bins = [0, 12, 24, 36, 48, 60]
labels = ["0-12m", "12-24m", "24-36m", "36-48m", "48-60m"]
df['FaixaEtaria'] = pd.cut(df['Idade em Meses'], bins=bins, labels=labels)
df_grouped = (
    df.groupby(['Região', 'FaixaEtaria', 'Tipo de Domicílio'])['indice_desenvolvimento']
    .mean()
    .reset_index(name='indice_medio')
)

st.markdown("#### Índice de Desenvolvimento (médio) por Faixa Etária, Tipo de Domicílio e Região")
fig_bar = px.bar(
    df_grouped,
    x='FaixaEtaria',
    y='indice_medio',
    color='Tipo de Domicílio',
    facet_col='Região',
    facet_col_wrap=2,
    barmode='group',
    color_discrete_sequence=nutrition_palette,
    title=""
)
fig_bar.update_layout(
    xaxis_title="Faixa Etária (meses)",
    yaxis_title="Índice de Desenvolvimento (médio)",
    font=dict(family="Roboto", size=12),
    plot_bgcolor="white",
    hovermode="x unified"
)
fig_bar.update_xaxes(categoryorder='array', categoryarray=labels)
st.plotly_chart(fig_bar, use_container_width=True)

col1, col2 = st.columns(2)
with col1:
    st.markdown("#### Distribuição de Cozinha por Tipo de Domicílio")
    fig_hist = px.histogram(
        df,
        x='Tipo de Domicílio',
        color='Possui Cozinha',
        barmode='group',
        color_discrete_sequence=nutrition_palette,
        title=""
    )
    fig_hist.update_layout(
        xaxis_title="Tipo de Domicílio",
        yaxis_title="Contagem",
        font=dict(family="Roboto", size=12),
        plot_bgcolor="white"
    )
    st.plotly_chart(fig_hist, use_container_width=True)

with col2:
    st.markdown("#### Distribuição de Tipos de Domicílio")
    domicilio_counts = df['Tipo de Domicílio'].value_counts().reset_index()
    domicilio_counts.columns = ['Tipo de Domicílio', 'Contagem']
    fig_pie = px.pie(
        domicilio_counts,
        values='Contagem',
        names='Tipo de Domicílio',
        color_discrete_sequence=nutrition_palette,
        hole=0.4,
        title=""
    )
    fig_pie.update_layout(
        font=dict(family="Roboto", size=12),
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
    )
    fig_pie.update_traces(textinfo='percent+label', pull=[0.05, 0, 0])
    st.plotly_chart(fig_pie, use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# Comparação entre Regiões e Dimensões
st.markdown('<div class="sub-header"><span>📈</span> Comparação entre Regiões e Dimensões</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header-description">Comparação detalhada entre regiões e dimensões socioeconômicas, como escolaridade, renda e etnia, para identificar disparidades no desenvolvimento infantil.</div>', unsafe_allow_html=True)
st.markdown('<div class="section">', unsafe_allow_html=True)

media_geral = df['indice_desenvolvimento'].mean()
df_regiao_plot = df.groupby('Região')['indice_desenvolvimento'].mean().reset_index(name='indice_medio')
df_escolaridade_plot = df.groupby(['Região', 'Nivel Escolaridade'])['indice_desenvolvimento'].mean().reset_index(name='indice_medio')
df_renda_plot = df.groupby(['Região', 'Faixa de Renda'])['indice_desenvolvimento'].mean().reset_index(name='indice_medio')
df_cor_plot = df.groupby(['Região', 'Cor Pessoa'])['indice_desenvolvimento'].mean().reset_index(name='indice_medio')

fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(22, 5), sharex=False, sharey=False)
sns.stripplot(data=df_regiao_plot, x='indice_medio', y='Região', ax=axes[0], color='#1E3A8A')
axes[0].set_title("Índice de Desenvolvimento por Região", fontsize=12, color='#1E3A8A')
axes[0].axvline(media_geral, linestyle='--', color='#EF4444', label=f'Média Geral = {media_geral:.2f}')
axes[0].set_xlabel("Índice de Desenvolvimento", fontsize=10)
axes[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True)
sns.stripplot(data=df_escolaridade_plot, x='indice_medio', y='Nivel Escolaridade', hue='Região', ax=axes[1])
axes[1].set_title("Escolaridade x Região", fontsize=12, color='#1E3A8A')
axes[1].axvline(media_geral, linestyle='--', color='#EF4444')
axes[1].set_xlabel("Índice de Desenvolvimento", fontsize=10)
axes[1].set_ylabel("")
axes[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True)
sns.stripplot(data=df_renda_plot, x='indice_medio', y='Faixa de Renda', hue='Região', ax=axes[2])
axes[2].set_title("Faixa de Renda x Região", fontsize=12, color='#1E3A8A')
axes[2].axvline(media_geral, linestyle='--', color='#EF4444')
axes[2].set_xlabel("Índice de Desenvolvimento", fontsize=10)
axes[2].set_ylabel("")
axes[2].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True)
sns.stripplot(data=df_cor_plot, x='indice_medio', y='Cor Pessoa', hue='Região', ax=axes[3])
axes[3].set_title("Cor da Pessoa x Região", fontsize=12, color='#1E3A8A')
axes[3].axvline(media_geral, linestyle='--', color='#EF4444')
axes[3].set_xlabel("Índice de Desenvolvimento", fontsize=10)
axes[3].set_ylabel("")
axes[3].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True)
plt.tight_layout()
st.pyplot(fig)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# Botão Fazer Análise
st.markdown('<div class="sub-header"><span>🔍</span> Análise de Predição</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header-description">Estimar a adequação da alimentação considerando aspectos socioeconômicos e estruturais.</div>', unsafe_allow_html=True)

st.markdown('<div style="text-align: center; margin: 2rem 0;">', unsafe_allow_html=True)
if st.button("Fazer Análise", key="predict_btn"):
    st.session_state.show_prediction = True
st.markdown('</div>', unsafe_allow_html=True)

# Seção de Predição (exibida apenas após clique)
if st.session_state.show_prediction:
    st.markdown('<div class="sub-header"><span></span> Predição da Qualidade da Alimentação</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header-description">Insira os dados da criança para prever a qualidade da alimentação com base em fatores socioeconômicos e de infraestrutura.</div>', unsafe_allow_html=True)
    st.markdown('<div class="section">', unsafe_allow_html=True)

    # Função para carregar o modelo treinado
    @st.cache_resource
    def load_model():
        with open("modelo_alimentos_basicos.pkl", "rb") as f:
            model = pickle.load(f)
        return model

    modelo = load_model()

    # Campos de entrada organizados em colunas
    col1, col2 = st.columns(2)
    with col1:
        nome = st.text_input("Nome da Criança:", value="")
        idade = st.number_input("Idade da Criança", value=0, min_value=0, max_value=5)
        idade_meses = st.write(f"Idade em meses: {idade * 12}")
        selecao_regiao = st.selectbox("Região em que moram", ["Norte", "Nordeste", "Sudeste", "Sul", "Centro-Oeste"])
        selecao_sexo = st.selectbox("Sexo da criança", ["Masculino", "Feminino"])
        selecao_domicilio = st.selectbox("Domicílio", ["Casa", "Apartamento", "Outros"])
        selecao_cozinha = st.selectbox("Possui Cozinha", ["Sim", "Não"])
        selecao_ocupacao = st.selectbox("Ocupação", [
            "Próprio de algum morador - já pago", "Próprio de algum morador - ainda pagando",
            "Alugado", "Cedido por empregador", "Cedido de outra forma", "Outra condição"
        ])
    with col2:
        selecao_registro = st.selectbox("Situação do Registro", ["Urbano", "Rural"])
        selecao_tosse = st.selectbox("Presença de Tosse", ["Sim", "Não", "Não sabe/ não quis responder"])
        selecao_respiracao = st.selectbox("Tipo de Respiração", ["Sim", "Não", "Não sabe/ não quis responder"])
        selecao_escolaridade = st.selectbox("Nível Escolaridade dos pais", [
            "Sem estudo", "1° ano do ensino fundamental", "1ª série/ 2°ano do ensino fundamental",
            "2ª série/ 3°ano do ensino fundamental", "3ª série/ 4°ano do ensino fundamental",
            "4ª série/ 5°ano do ensino fundamental", "5ª série/ 6°ano do ensino fundamental",
            "6ª série/ 7°ano do ensino fundamental", "7ª série/ 8°ano do ensino fundamental",
            "8ª série/ 9°ano do ensino fundamental", "1°ano do ensino médio", "2°ano do ensino médio",
            "3°ano do ensino médio", "Ensino superior incompleto", "Ensino superior completo"
        ])
        selecao_renda = st.selectbox("Faixa de Renda da casa", [
            "Sem renda", "Até R$ 1.000,00", "De R$ 1.001,00 até R$ 2.000,00",
            "De R$ 2.001,00 até R$ 3.000,00", "De R$ 3.001,00 até R$ 5.000,00",
            "De R$ 5.001,00 até R$ 10.000,00", "R$ 10.001,00 ou mais"
        ])
        selecao_cor_pessoa = st.selectbox("Cor da criança", [
            "Branca", "Preta", "Amarela (origem japonesa, chinesa, coreana etc.)",
            "Parda (mulata, cabocla, cafuza, mameluca ou mestiça)", "Indígena", "Não sabe/não quis responder"
        ])
        selecao_moradores_alimentaram_sim = st.selectbox("Moradores que Alimentaram Acabamento (Sim)", ["Sim", "Não"])
        selecao_moradores_alimentaram_nao = st.selectbox("Moradores que Alimentaram Acabamento (Não)", ["Sim", "Não"])

    # Benefícios
    beneficios_opcoes = [
        "Programa Bolsa Família (PBF)", "Benefício de Prestação Continuada (BPC/LOAS)",
        "Bolsa ou benefício da Prefeitura Municipal", "Bolsa ou benefício do Governo do Estado",
        "Pensão", "Aposentadoria", "Outro benefício"
    ]
    beneficios_selecionados = st.multiselect("Benefícios recebidos", beneficios_opcoes)
    beneficios_mapping = {
        "Programa Bolsa Família (PBF)": "A",
        "Benefício de Prestação Continuada (BPC/LOAS)": "B",
        "Bolsa ou benefício da Prefeitura Municipal": "C",
        "Bolsa ou benefício do Governo do Estado": "D",
        "Pensão": "E",
        "Aposentadoria": "F",
        "Outro benefício": "G"
    }
    beneficios_input = [1 if beneficios_mapping[ben] in beneficios_selecionados else 0 for ben in beneficios_opcoes]
    total_beneficios = len(beneficios_selecionados)

    # Mapeamentos
    mapping = {
        "Região": {"Norte": 1, "Nordeste": 2, "Sudeste": 3, "Sul": 4, "Centro-Oeste": 5},
        "Sexo": {"Masculino": 1, "Feminino": 2},
        "Domicílio": {"Casa": 1, "Apartamento": 2, "Outros": 3},
        "Cozinha": {"Sim": 1, "Não": 0},
        "Ocupação": {
            "Próprio de algum morador - já pago": 1, "Próprio de algum morador - ainda pagando": 2,
            "Alugado": 3, "Cedido por empregador": 4, "Cedido de outra forma": 5, "Outra condição": 6
        },
        "Registro": {"Urbano": 1, "Rural": 2},
        "Tosse": {"Sim": 1, "Não": 2, "Não sabe/ não quis responder": 9},
        "Respiração": {"Sim": 1, "Não": 2, "Não sabe/ não quis responder": 9},
        "Escolaridade": {
            "Sem estudo": 0, "1° ano do ensino fundamental": 1, "1ª série/ 2°ano do ensino fundamental": 2,
            "2ª série/ 3°ano do ensino fundamental": 3, "3ª série/ 4°ano do ensino fundamental": 4,
            "4ª série/ 5°ano do ensino fundamental": 5, "5ª série/ 6°ano do ensino fundamental": 6,
            "6ª série/ 7°ano do ensino fundamental": 7, "7ª série/ 8°ano do ensino fundamental": 8,
            "8ª série/ 9°ano do ensino fundamental": 9, "1°ano do ensino médio": 10, "2°ano do ensino médio": 11,
            "3°ano do ensino médio": 12, "Ensino superior incompleto": 13, "Ensino superior completo": 14
        },
        "Renda": {
            "Sem renda": 1, "Até R$ 1.000,00": 2, "De R$ 1.001,00 até R$ 2.000,00": 3,
            "De R$ 2.001,00 até R$ 3.000,00": 4, "De R$ 3.001,00 até R$ 5.000,00": 5,
            "De R$ 5.001,00 até R$ 10.000,00": 6, "R$ 10.001,00 ou mais": 7
        }
    }
    mapping_cor_pessoa = {
        "Branca": 1, "Preta": 2, "Amarela (origem japonesa, chinesa, coreana etc.)": 3,
        "Parda (mulata, cabocla, cafuza, mameluca ou mestiça)": 4, "Indígena": 5, "Não sabe/não quis responder": 9
    }
    mapping_sim_nao = {"Sim": 1, "Não": 2}
    mapping_alimentos_reverse = {
        1: "Não", 2: "Sim, raramente", 3: "Sim, às vezes", 4: "Sim, quase sempre",
        5: "Sim, sempre", 6: "Não se cozinha em casa"
    }

    # Preparar dados de entrada
    input_data = [
        idade, idade_meses,
        mapping["Região"][selecao_regiao],
        mapping["Sexo"][selecao_sexo],
        mapping["Domicílio"][selecao_domicilio],
        mapping["Cozinha"][selecao_cozinha],
        mapping["Ocupação"][selecao_ocupacao],
        mapping["Registro"][selecao_registro],
        mapping["Tosse"][selecao_tosse],
        mapping["Respiração"][selecao_respiracao],
        mapping["Escolaridade"][selecao_escolaridade],
        mapping["Renda"][selecao_renda]
    ] + beneficios_input + [
        total_beneficios,
        mapping_cor_pessoa[selecao_cor_pessoa],
        mapping_sim_nao[selecao_moradores_alimentaram_sim],
        mapping_sim_nao[selecao_moradores_alimentaram_nao]
    ]

    # Botão de predição
    if st.button("Prever Qualidade da Alimentação", key="predict_quality"):
        resultado = modelo.predict([input_data])[0]
        probabilidade = modelo.predict_proba([input_data]).max()
        st.markdown(f"""
        <div class="insight-card green">
            <div class="insight-title">Resultado da Predição</div>
            <p style="color: var(--text-dark); font-size: 1.1rem;">
                🍽️ O modelo previu a qualidade da alimentação como: <strong>{mapping_alimentos_reverse[resultado]}</strong><br>
                Confiança da predição: {probabilidade:.2%}
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# Insights e Recomendações
st.markdown('<div class="sub-header"><span>💡</span> Insights e Recomendações</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header-description">Recomendações baseadas nos dados para melhorar o desenvolvimento infantil e reduzir a desnutrição no Brasil.</div>', unsafe_allow_html=True)
st.markdown('<div class="section">', unsafe_allow_html=True)
st.markdown("""
<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 1.5rem;">
    <div style="background: var(--primary-light); padding: 1.5rem; border-radius: 12px; box-shadow: 0 2px 8px var(--shadow);">
        <h4 style="margin: 0; color: var(--primary); font-size: 1.2rem;">Disparidades Regionais</h4>
        <p style="margin: 0.5rem 0 0; color: var(--text-medium);">
            Os dados revelam diferenças significativas entre as regiões do Brasil,
            com o Norte e Nordeste apresentando indicadores mais baixos de desenvolvimento infantil.
            Políticas públicas devem priorizar estas regiões com intervenções específicas.
        </p>
    </div>
    <div style="background: var(--secondary-light); padding: 1.5rem; border-radius: 12px; box-shadow: 0 2px 8px var(--shadow);">
        <h4 style="margin: 0; color: var(--secondary); font-size: 1.2rem;">Acesso a Alimentos</h4>
        <p style="margin: 0.5rem 0 0; color: var(--text-medium);">
            O acesso a alimentos básicos está fortemente correlacionado com a renda familiar.
            Programas de transferência de renda e alimentação escolar devem ser fortalecidos.
        </p>
    </div>
    <div style="background: var(--primary-light); padding: 1.5rem; border-radius: 12px; box-shadow: 0 2px 8px var(--shadow);">
        <h4 style="margin: 0; color: var(--primary); font-size: 1.2rem;">Infraestrutura Domiciliar</h4>
        <p style="margin: 0.5rem 0 0; color: var(--text-medium);">
            A presença de cozinha adequada impacta diretamente na capacidade das famílias
            de preparar refeições nutritivas. Programas habitacionais devem considerar isso.
        </p>
    </div>
</div>
""", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="footer">
    <p style="font-size: 1rem; margin-bottom: 0.5rem;">
        Fonte: Secretarias Estaduais de Saúde, Brasil 2025
    </p>
    <p style="font-size: 0.9rem; opacity: 0.8;">
        © 2025 Análise de Desnutrição Infantil | Todos os direitos reservados
    </p>
</div>
""", unsafe_allow_html=True)