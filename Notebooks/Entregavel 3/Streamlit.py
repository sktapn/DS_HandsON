import streamlit as st
import pandas as pd
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import plotly.express as px
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier

# ----------------------------------------------------------
# Configuração da Página e CSS Customizado
# ----------------------------------------------------------
st.set_page_config(
    page_title="Análise de Desnutrição Infantil no Brasil",
    page_icon="👶",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* Variáveis de cores - Paleta suave e profissional */
    :root {
        --primary: #3a86ff;
        --primary-light: #e1ebff;
        --primary-dark: #0043a9;
        --secondary: #38b000;
        --secondary-light: #e3f5e1;
        --accent: #ff9e00;
        --accent-light: #fff4e1;
        --danger: #ef476f;
        --danger-light: #fde1e7;
        --text-dark: #2b2d42;
        --text-medium: #555b6e;
        --text-light: #8d99ae;
        --background: #f8f9fa;
        --card: #ffffff;
        --border: #e9ecef;
        --shadow: rgba(0, 0, 0, 0.05);
    }
    /* Reset e estilos base */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: var(--text-dark);
    }
    /* Estilo geral da página */
    .main {
        background-color: var(--background);
        background-image: linear-gradient(to bottom, #f8f9fa, #ffffff);
    }
    /* Cabeçalho principal */
    .main-header {
        font-size: 2.5rem;
        color: var(--primary-dark);
        text-align: center;
        margin-bottom: 1.5rem;
        font-weight: 800;
        letter-spacing: -0.025em;
        padding-bottom: 1rem;
        border-bottom: 3px solid var(--primary);
        background: linear-gradient(90deg, var(--primary-dark) 0%, var(--primary) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    /* Subtítulos */
    .sub-header {
        font-size: 1.8rem;
        color: var(--primary-dark);
        margin-top: 2rem;
        margin-bottom: 1.5rem;
        font-weight: 700;
        letter-spacing: -0.01em;
        border-bottom: 2px solid var(--primary-light);
        padding-bottom: 0.75rem;
    }
    /* Seções */
    .section {
        background-color: var(--card);
        padding: 1.8rem;
        border-radius: 16px;
        margin-bottom: 1.8rem;
        border: 1px solid var(--border);
        box-shadow: 0 4px 20px var(--shadow);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .section:hover {
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.08);
    }
    /* Cards de métricas */
    .metric-card {
        background-color: var(--card);
        padding: 1.8rem;
        border-radius: 16px;
        box-shadow: 0 4px 20px var(--shadow);
        text-align: center;
        transition: all 0.3s ease;
        border: 1px solid var(--border);
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 30px rgba(0, 0, 0, 0.1);
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: 800;
        color: var(--primary);
        margin: 0.5rem 0;
        letter-spacing: -0.03em;
    }
    .metric-label {
        font-size: 1.1rem;
        color: var(--text-medium);
        margin-top: 0.5rem;
        font-weight: 500;
    }
    /* Gráficos */
    .stPlotlyChart {
        background-color: var(--card);
        border-radius: 16px;
        padding: 1.2rem;
        box-shadow: 0 4px 20px var(--shadow);
        border: 1px solid var(--border);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .stPlotlyChart:hover {
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.08);
    }
    /* Sidebar */
    .sidebar .sidebar-content {
        background-color: var(--primary-light);
        background-image: linear-gradient(180deg, var(--primary-light) 0%, rgba(255, 255, 255, 0.95) 100%);
    }
    /* Botões */
    .stButton>button {
        background-color: var(--primary);
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.6rem 1.2rem;
        font-weight: 600;
        transition: all 0.2s;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
    }
    .stButton>button:hover {
        background-color: var(--primary-dark);
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.15);
        transform: translateY(-2px);
    }
    /* Seletores */
    .stSelectbox>div>div {
        background-color: white;
        border-radius: 8px;
        border: 1px solid var(--border);
    }
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
        background-color: var(--primary-light);
        border-radius: 12px;
        padding: 5px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 10px 16px;
        background-color: transparent;
    }
    .stTabs [aria-selected="true"] {
        background-color: white;
        color: var(--primary-dark);
        font-weight: 600;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.05);
    }
    /* Tabelas */
    .dataframe {
        border-collapse: collapse;
        width: 100%;
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
    }
    .dataframe th {
        background-color: var(--primary-light);
        color: var(--primary-dark);
        padding: 12px;
        text-align: left;
        font-weight: 600;
    }
    .dataframe td {
        padding: 10px;
        border-top: 1px solid var(--border);
    }
    .dataframe tr:nth-child(even) {
        background-color: #f8fafc;
    }
    /* Ícones e indicadores */
    .positive-indicator {
        color: var(--secondary);
        font-weight: 600;
    }
    .negative-indicator {
        color: var(--danger);
        font-weight: 600;
    }
    /* Cartão informativo */
    .info-card {
        background-color: var(--primary-light);
        border-left: 4px solid var(--primary);
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    .warning-card {
        background-color: var(--accent-light);
        border-left: 4px solid var(--accent);
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    .danger-card {
        background-color: var(--danger-light);
        border-left: 4px solid var(--danger);
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    /* Animações e transições */
    .animate-fade-in {
        animation: fadeIn 0.5s ease-in;
    }
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    /* Footer */
    .footer {
        text-align: center;
        margin-top: 3rem;
        padding: 2rem;
        background: linear-gradient(135deg, var(--primary-dark) 0%, var(--primary) 100%);
        color: white;
        border-radius: 16px;
    }
    .footer a {
        color: white;
        text-decoration: underline;
    }
    /* Expander personalizado */
    .streamlit-expanderHeader {
        font-weight: 600;
        color: var(--primary-dark);
    }
    /* Barra de progresso personalizada */
    .progress-container {
        width: 100%;
        height: 8px;
        background-color: #e9ecef;
        border-radius: 4px;
        margin-top: 10px;
        overflow: hidden;
    }
    .progress-bar {
        height: 100%;
        border-radius: 4px;
        transition: width 0.5s ease;
    }
    /* Badges */
    .badge {
        display: inline-block;
        padding: 0.25rem 0.5rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .badge-primary {
        background-color: var(--primary-light);
        color: var(--primary-dark);
    }
    .badge-secondary {
        background-color: var(--secondary-light);
        color: var(--secondary);
    }
    .badge-accent {
        background-color: var(--accent-light);
        color: var(--accent);
    }
    .badge-danger {
        background-color: var(--danger-light);
        color: var(--danger);
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
    "education": "📚"
}
nutrition_palette = ["#3a86ff", "#38b000", "#ff9e00", "#9d4edd", "#ef476f", "#073b4c"]
nutrition_cmap = LinearSegmentedColormap.from_list("nutrition_cmap", ["#3a86ff", "#ef476f"])

# ----------------------------------------------------------
# Cabeçalho e Introdução
# ----------------------------------------------------------
st.markdown(
    f'<div class="main-header animate-fade-in">{nutrition_icons["main"]} Análise de Desnutrição Infantil no Brasil</div>',
    unsafe_allow_html=True
)

st.markdown("""
<div class="section animate-fade-in">
    <div style="display: flex; align-items: center; margin-bottom: 1rem;">
        <div style="background-color: #e1ebff; border-radius: 50%; width: 40px; height: 40px; display: flex; justify-content: center; align-items: center; margin-right: 1rem;">
            <span style="font-size: 1.5rem;">📊</span>
        </div>
        <h3 style="color: #0043a9; margin: 0; font-size: 1.5rem;">Sobre este Estudo</h3>
    </div>
    <p style="font-size: 1.05rem; line-height: 1.6; color: #555b6e;">
        Esta plataforma apresenta uma análise abrangente da desnutrição infantil no Brasil,
        identificando padrões regionais, determinantes socioeconômicos e fatores de infraestrutura
        que impactam o desenvolvimento das crianças brasileiras.
    </p>
    <div class="info-card">
        <div style="display: flex; align-items: flex-start;">
            <div style="font-size: 1.5rem; margin-right: 0.75rem;">ℹ️</div>
            <div>
                <p style="margin: 0; font-weight: 600; color: #0043a9;">Por que este estudo é importante?</p>
                <p style="margin-top: 0.5rem; margin-bottom: 0; color: #555b6e;">
                    A desnutrição infantil continua sendo um desafio significativo para a saúde pública no Brasil,
                    com impactos duradouros no desenvolvimento físico e cognitivo das crianças.
                    Compreender os fatores que contribuem para este problema é essencial para desenvolver
                    políticas públicas eficazes e intervenções direcionadas.
                </p>
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ----------------------------------------------------------
# Sidebar para Filtros e Controles
# ----------------------------------------------------------
with st.sidebar:
    st.image("C:\\Users\\Alunos\\Desktop\\Entregavel\\Brasão_da_UFRR.png")
    st.title(f"{nutrition_icons['main']} Controle de Dados")
    st.markdown('<div style="border-bottom: 1px solid #e9ecef; margin-bottom: 20px;"></div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Carregar arquivo de dados", type=["csv"])

    st.markdown(
        '<p style="font-size: 0.9rem; color: #555b6e; font-weight: 500; margin-bottom: 0.5rem;">Filtros de Análise</p>',
        unsafe_allow_html=True
    )

    regioes = st.multiselect(
        "Regiões",
        options=['Norte', 'Sul', 'Sudeste', 'Centro-Oeste', 'Nordeste'],
        default=['Norte'],
        help="Escolha uma ou mais regiões para análise"
    )

    st.markdown('<p style="font-size: 0.9rem; color: #555b6e; margin-bottom: 0.25rem;">Faixa Etária (meses)</p>',
                unsafe_allow_html=True)
    faixa_etaria = st.select_slider(
        "",
        options=[0, 12, 24, 36, 48, 60],
        value=(0, 60),
        format_func=lambda x: f"{x} meses"
    )

    tipo_domicilio = st.multiselect(
        "Tipo de Domicílio",
        options=['Casa', 'Apartamento', 'Habitação em casa de cômodos'],
        default=['Casa', 'Apartamento']
    )

    acesso_alimentos = st.radio(
        "Acesso a Alimentos Básicos",
        options=["Todos", "Sim, sempre", "Sim, quase sempre", "Sim, às vezes"],
        index=0
    )

    st.markdown('<div style="border-bottom: 1px solid #e9ecef; margin: 20px 0;"></div>', unsafe_allow_html=True)

    st.markdown("""
    <div style="background-color: #e1ebff; padding: 15px; border-radius: 12px; margin-bottom: 20px;">
        <p style="font-weight: 600; color: #0043a9; margin-bottom: 8px;">Sobre os Indicadores</p>
        <ul style="margin: 0; padding-left: 20px; color: #555b6e; font-size: 0.9rem;">
            <li>Índice de Desenvolvimento: Média dos scores de alimentos, saúde e infraestrutura</li>
            <li>Score Alimentos: Disponibilidade de alimentos básicos</li>
            <li>Score Infraestrutura: Condições adequadas de moradia</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    if st.button("Atualizar Análise", key="update_btn"):
        st.success("Análise atualizada com sucesso!")

    st.markdown(
        '<div style="margin-top: 30px; text-align: center; font-size: 0.8rem; color: #8d99ae;">Versão 3.0.0</div>',
        unsafe_allow_html=True
    )


# ----------------------------------------------------------
# Função para Carregar Dados (Cache)
# ----------------------------------------------------------
@st.cache_data
def load_data(file):
    if file is not None:
        df = pd.read_csv(file)
    else:
        st.markdown("""
        <div class="warning-card">
            <div style="display: flex; align-items: flex-start;">
                <div style="font-size: 1.5rem; margin-right: 0.75rem;">⚠️</div>
                <div>
                    <p style="margin: 0; font-weight: 600; color: #ff9e00;">Dados de Exemplo</p>
                    <p style="margin-top: 0.5rem; margin-bottom: 0; color: #555b6e;">
                        Nenhum arquivo carregado. Utilizando dados simulados para demonstração.
                    </p>
                </div>
            </div>
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
            'Ocupação': np.random.choice(['Próprio de algum morador - já pago', 'Alugado', 'Cedido de outra forma'],
                                         100),
            'Situação do Registro': np.random.choice(['Urbano'], 100),
            'Presença de Tosse': np.random.choice(['Sim', 'Não'], 100),
            'Tipo de Respiração': np.random.choice(['Sim', 'Não'], 100),
            'Alimentos Básicos': np.random.choice(['Sim, sempre', 'Sim, quase sempre', 'Sim, às vezes'], 100),
            'Nivel Escolaridade': np.random.choice(
                ['1°ano do ensino médio', '3°ano do ensino médio', 'Ensino superior completo'], 100),
            'Beneficios': np.random.choice(['A', 'F', ''], 100),
            'Faixa de Renda': np.random.choice(
                ['Até R$ 1.000,00', 'De R$ 1.001,00 até R$ 2.000,00', 'De R$ 2.001,00 até R$ 3.000,00'], 100),
            'Cor Pessoa': np.random.choice(['Branca', 'Parda (mulata, cabocla, cafuza, mameluca ou mestiça)'], 100)
        })
    return df


df = load_data(uploaded_file)

# Aplicando filtros
df = df[df['Região'].isin(regioes)]
if acesso_alimentos != "Todos":
    df = df[df['Alimentos Básicos'] == acesso_alimentos]

# Pré-processamento da coluna "Idade em Meses"
df['Idade em Meses'] = df['Idade em Meses'].astype(str).str.replace(' meses', '', regex=False).str.strip()
df['Idade em Meses'] = pd.to_numeric(df['Idade em Meses'], errors='coerce')


# ----------------------------------------------------------
# Geração de Scores e Indicadores
# ----------------------------------------------------------
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

# ----------------------------------------------------------
# Criação de df_final para a Aba 4 (Agrupamento por Dimensão)
# ----------------------------------------------------------
resultados = []
# (a) Agrupamento por Região
df_regiao_group = df.groupby('Região')['indice_desenvolvimento'].mean().reset_index()
df_regiao_group['dimensao'] = 'Região'
df_regiao_group.rename(columns={'Região': 'categoria', 'indice_desenvolvimento': 'indice_medio'}, inplace=True)
resultados.append(df_regiao_group)
# (b) Agrupamento por Nível Escolaridade
df_escolaridade_group = df.groupby('Nivel Escolaridade')['indice_desenvolvimento'].mean().reset_index()
df_escolaridade_group['dimensao'] = 'Nível Escolaridade'
df_escolaridade_group.rename(columns={'Nivel Escolaridade': 'categoria', 'indice_desenvolvimento': 'indice_medio'},
                             inplace=True)
resultados.append(df_escolaridade_group)
# (c) Agrupamento por Faixa de Renda
df_renda_group = df.groupby('Faixa de Renda')['indice_desenvolvimento'].mean().reset_index()
df_renda_group['dimensao'] = 'Faixa de Renda'
df_renda_group.rename(columns={'Faixa de Renda': 'categoria', 'indice_desenvolvimento': 'indice_medio'}, inplace=True)
resultados.append(df_renda_group)
# (d) Agrupamento por Cor Pessoa
df_cor_group = df.groupby('Cor Pessoa')['indice_desenvolvimento'].mean().reset_index()
df_cor_group['dimensao'] = 'Cor Pessoa'
df_cor_group.rename(columns={'Cor Pessoa': 'categoria', 'indice_desenvolvimento': 'indice_medio'}, inplace=True)
resultados.append(df_cor_group)

df_final = pd.concat(resultados, ignore_index=True)

# ----------------------------------------------------------
# Aba de Indicadores e Visualizações (Tabs)
# ----------------------------------------------------------
tabs = st.tabs([
    f"{nutrition_icons['development']} Indicadores Regionais",
    f"{nutrition_icons['socioeconomic']} Determinantes Socioeconômicos",
    f"{nutrition_icons['infrastructure']} Infraestrutura e Nutrição",
    f"{nutrition_icons['main']} Comparação Entre Regiões",
    "🔮 Análise Preditiva"
])

# -- Aba 1: Indicadores Regionais --
with tabs[0]:
    st.markdown('<div class="sub-header animate-fade-in">Análise Regional</div>', unsafe_allow_html=True)
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown("### Índice de Desenvolvimento por Região")

    grouped_df = df.groupby('Região')['indice_desenvolvimento'].mean().reset_index()

    if len(grouped_df) == 1:
        fig = px.bar(
            grouped_df,
            x='Região',
            y='indice_desenvolvimento',
            color='Região',
            color_discrete_sequence=nutrition_palette,
            title="Índice Médio de Desenvolvimento por Região"
        )
        fig.update_traces(width=0.4)
        fig.update_layout(
            xaxis=dict(range=[-0.5, 0.5]),
            height=400,
            width=500,
            xaxis_title="Região",
            yaxis_title="Índice de Desenvolvimento",
            font=dict(family="Inter", size=12),
            plot_bgcolor="white",
            hoverlabel=dict(
                bgcolor="white",
                font_size=12,
                font_family="Inter"
            ),
            margin=dict(l=20, r=20, t=40, b=20)
        )
    else:
        fig = px.bar(
            grouped_df,
            x='Região',
            y='indice_desenvolvimento',
            color='Região',
            color_discrete_sequence=nutrition_palette,
            title="Índice Médio de Desenvolvimento por Região"
        )
        fig.update_layout(
            xaxis_title="Região",
            yaxis_title="Índice de Desenvolvimento",
            font=dict(family="Inter", size=12),
            plot_bgcolor="white",
            hoverlabel=dict(
                bgcolor="white",
                font_size=12,
                font_family="Inter"
            ),
            margin=dict(l=20, r=20, t=40, b=20)
        )
        fig.update_traces(
            marker_line_color='white',
            marker_line_width=1.5,
            opacity=0.85
        )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Mapa de Calor: Indicadores por Região")
    heatmap_data = pd.DataFrame({
        'Região': df['Região'].unique()
    })
    heatmap_data['Índice de Desenvolvimento'] = [df[df['Região'] == r]['indice_desenvolvimento'].mean() for r in
                                                 heatmap_data['Região']]
    heatmap_data['Acesso a Alimentos'] = [df[df['Região'] == r]['alimentos_score'].mean() for r in
                                          heatmap_data['Região']]
    heatmap_data['Infraestrutura'] = [df[df['Região'] == r]['cozinha_score'].mean() for r in heatmap_data['Região']]
    fig_heat = px.imshow(
        heatmap_data.set_index('Região')[['Índice de Desenvolvimento', 'Acesso a Alimentos', 'Infraestrutura']],
        text_auto='.2f',
        aspect="auto",
        color_continuous_scale=px.colors.sequential.Blues
    )
    fig_heat.update_layout(
        title="Comparativo de Indicadores por Região",
        xaxis_title="Indicador",
        yaxis_title="Região",
        font=dict(family="Inter", size=12),
        coloraxis_colorbar=dict(
            title="Valor",
            tickformat=".2f"
        )
    )
    st.plotly_chart(fig_heat, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# -- Aba 2: Determinantes Socioeconômicos --
with tabs[1]:
    st.markdown('<div class="sub-header">Determinantes Socioeconômicos</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section">', unsafe_allow_html=True)
        st.markdown("### Índice de Desenvolvimento por Ocupação")
        fig = px.box(
            df,
            x='Ocupação',
            y='indice_desenvolvimento',
            color='Ocupação',
            color_discrete_sequence=nutrition_palette,
            title="Índice de Desenvolvimento por Ocupação dos Pais"
        )
        fig.update_layout(
            xaxis_title="Ocupação",
            yaxis_title="Índice de Desenvolvimento",
            font=dict(family="Inter", size=12),
            plot_bgcolor="white",
            showlegend=False,
            xaxis={'visible': False}
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="section">', unsafe_allow_html=True)
        st.markdown("### Acesso a Alimentos por Faixa de Renda")
        fig = px.histogram(
            df,
            x='Faixa de Renda',
            color='Alimentos Básicos',
            barmode='group',
            color_discrete_sequence=nutrition_palette,
            title="Distribuição de Acesso a Alimentos por Faixa de Renda"
        )
        fig.update_layout(
            xaxis_title="Faixa de Renda",
            yaxis_title="Contagem",
            font=dict(family="Inter", size=12),
            plot_bgcolor="white",
            xaxis={'categoryorder': 'total descending', 'tickangle': -45}
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown("### Correlação entre Fatores Socioeconômicos e Desenvolvimento")
    corr_cols = ['indice_desenvolvimento', 'alimentos_score', 'cozinha_score', 'tosse_score']
    corr_data = df[corr_cols].corr()
    fig_corr = px.imshow(
        corr_data,
        text_auto='.2f',
        color_continuous_scale=px.colors.diverging.RdBu_r,
        zmin=-1, zmax=1
    )
    fig_corr.update_layout(
        title="Matriz de Correlação entre Indicadores",
        font=dict(family="Inter", size=12)
    )
    st.plotly_chart(fig_corr, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# -- Aba 3: Infraestrutura e Nutrição --
with tabs[2]:
    st.markdown('<div class="sub-header">Infraestrutura e Nutrição</div>', unsafe_allow_html=True)
    st.markdown('<div class="section">', unsafe_allow_html=True)
    bins = [0, 12, 24, 36, 48, 60]
    labels = ["0-12m", "12-24m", "24-36m", "36-48m", "48-60m"]
    df['FaixaEtaria'] = pd.cut(df['Idade em Meses'], bins=bins, labels=labels)
    df_grouped = (
        df.groupby(['Região', 'FaixaEtaria', 'Tipo de Domicílio'])['indice_desenvolvimento']
        .mean()
        .reset_index(name='indice_medio')
    )
    st.markdown("### Índice de Desenvolvimento (médio) por Faixa Etária, Tipo de Domicílio e Região")
    fig_bar = px.bar(
        df_grouped,
        x='FaixaEtaria',
        y='indice_medio',
        color='Tipo de Domicílio',
        facet_col='Região',
        facet_col_wrap=2,
        barmode='group',
        color_discrete_sequence=nutrition_palette,
        title="Infraestrutura e Nutrição: Comparação em Barras Agrupadas"
    )
    fig_bar.update_layout(
        xaxis_title="Faixa Etária (meses)",
        yaxis_title="Índice de Desenvolvimento (médio)",
        font=dict(family="Inter", size=12),
        plot_bgcolor="white",
        hovermode="x unified"
    )
    fig_bar.update_xaxes(categoryorder='array', categoryarray=labels)
    st.plotly_chart(fig_bar, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown("### Distribuição de Cozinha por Tipo de Domicílio")
    fig_hist = px.histogram(
        df,
        x='Tipo de Domicílio',
        color='Possui Cozinha',
        barmode='group',
        color_discrete_sequence=nutrition_palette,
        title="Presença de Cozinha por Tipo de Domicílio"
    )
    fig_hist.update_layout(
        xaxis_title="Tipo de Domicílio",
        yaxis_title="Contagem",
        font=dict(family="Inter", size=12),
        plot_bgcolor="white"
    )
    st.plotly_chart(fig_hist, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown("### Distribuição de Tipos de Domicílio")
    domicilio_counts = df['Tipo de Domicílio'].value_counts().reset_index()
    domicilio_counts.columns = ['Tipo de Domicílio', 'Contagem']
    fig_pie = px.pie(
        domicilio_counts,
        values='Contagem',
        names='Tipo de Domicílio',
        color_discrete_sequence=nutrition_palette,
        hole=0.4,
        title="Distribuição de Tipos de Domicílio"
    )
    fig_pie.update_layout(
        font=dict(family="Inter", size=12),
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
    )
    fig_pie.update_traces(textinfo='percent+label', pull=[0.05, 0, 0])
    st.plotly_chart(fig_pie, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# -- Aba 4: Comparação Entre Regiões e Dimensões --
with tabs[3]:
    st.markdown('<div class="sub-header">Comparação Entre Regiões e Dimensões</div>', unsafe_allow_html=True)
    st.markdown('<div class="section">', unsafe_allow_html=True)

    # Calcula a média geral para a linha de referência
    media_geral = df['indice_desenvolvimento'].mean()

    # 1) Dados para o subplot de Regiões (apenas por Região)
    df_regiao_plot = (
        df.groupby('Região')['indice_desenvolvimento']
        .mean()
        .reset_index(name='indice_medio')
    )

    # 2) Dados para o subplot de Escolaridade (Região + Nivel Escolaridade)
    df_escolaridade_plot = (
        df.groupby(['Região', 'Nivel Escolaridade'])['indice_desenvolvimento']
        .mean()
        .reset_index(name='indice_medio')
    )

    # 3) Dados para o subplot de Faixa de Renda (Região + Faixa de Renda)
    df_renda_plot = (
        df.groupby(['Região', 'Faixa de Renda'])['indice_desenvolvimento']
        .mean()
        .reset_index(name='indice_medio')
    )

    # 4) Dados para o subplot de Cor Pessoa (Região + Cor Pessoa)
    df_cor_plot = (
        df.groupby(['Região', 'Cor Pessoa'])['indice_desenvolvimento']
        .mean()
        .reset_index(name='indice_medio')
    )

    # Cria subplots: 1 linha, 4 colunas
    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(22, 5), sharex=False, sharey=False)

    # --- Subplot (a) REGIÃO ---
    sns.stripplot(
        data=df_regiao_plot,
        x='indice_medio',
        y='Região',  # Eixo Y: as regiões
        ax=axes[0],
        color='blue'
    )
    axes[0].set_title("Índice de Desenvolvimento por Região")
    axes[0].axvline(media_geral, linestyle='--', color='red', label=f'Média Geral = {media_geral:.2f}')
    axes[0].set_xlabel("Índice de Desenvolvimento")
    axes[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True)

    # --- Subplot (b) ESCOLARIDADE + REGIÃO ---
    sns.stripplot(
        data=df_escolaridade_plot,
        x='indice_medio',
        y='Nivel Escolaridade',  # Eixo Y: Escolaridade
        hue='Região',  # Cores conforme Região
        ax=axes[1]
    )
    axes[1].set_title("Escolaridade x Região")
    axes[1].axvline(media_geral, linestyle='--', color='red')
    axes[1].set_xlabel("Índice de Desenvolvimento")
    axes[1].set_ylabel("")
    axes[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True)

    # --- Subplot (c) FAIXA DE RENDA + REGIÃO ---
    sns.stripplot(
        data=df_renda_plot,
        x='indice_medio',
        y='Faixa de Renda',
        hue='Região',
        ax=axes[2]
    )
    axes[2].set_title("Faixa de Renda x Região")
    axes[2].axvline(media_geral, linestyle='--', color='red')
    axes[2].set_xlabel("Índice de Desenvolvimento")
    axes[2].set_ylabel("")
    axes[2].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True)

    # --- Subplot (d) COR DA PESSOA + REGIÃO ---
    sns.stripplot(
        data=df_cor_plot,
        x='indice_medio',
        y='Cor Pessoa',
        hue='Região',
        ax=axes[3]
    )
    axes[3].set_title("Cor da Pessoa x Região")
    axes[3].axvline(media_geral, linestyle='--', color='red')
    axes[3].set_xlabel("Índice de Desenvolvimento")
    axes[3].set_ylabel("")
    axes[3].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True)

    plt.tight_layout()
    st.pyplot(fig)

    st.markdown('</div>', unsafe_allow_html=True)

# -- Aba 5: Somente Predição --
with tabs[4]:
    st.markdown('<div class="sub-header">Predição da Qualidade da Alimentação</div>', unsafe_allow_html=True)


   # Função para carregar o modelo treinado
    @st.cache_resource
    def load_model():
        with open("modelo_alimentos_basicos.pkl", "rb") as f:
            model = pickle.load(f)
        return model
    
    modelo = load_model()

    idade = st.number_input("Idade da Criança", value=0)
    nome = st.text_input("Nome da Criança:", value="")

    idade_meses = st.number_input("Idade em Meses", value=0)
    selecao_regiao = st.selectbox("Região em que moram", ["Norte", "Nordeste", "Sudeste", "Sul", "Centro-Oeste"])
    selecao_sexo = st.selectbox("Sexo da criança", ["Masculino", "Feminino"])
    selecao_domicilio = st.selectbox("Domicílio", ["Casa", "Apartamento", "Outros"])
    selecao_cozinha = st.selectbox("Possui Cozinha", ["Sim", "Não"])
    selecao_ocupacao = st.selectbox("Ocupação", ["Próprio de algum morador - já pago", "Próprio de algum morador - ainda pagando", "Alugado", "Cedido por empregador", "Cedido de outra forma", "Outra condição"])
    selecao_registro = st.selectbox("Situação do Registro", ["Urbano", "Rural"])
    selecao_tosse = st.selectbox("Presença de Tosse", ["Sim", "Não", "Não sabe/ não quis responder"])
    selecao_respiracao = st.selectbox("Tipo de Respiração", ["Sim", "Não", "Não sabe/ não quis responder"])
    selecao_escolaridade = st.selectbox("Nível Escolaridade dos pais", ["Sem estudo", "1° ano do ensino fundamental", "1ª série/ 2°ano do ensino fundamental",
                                                                "2ª série/ 3°ano do ensino fundamental", "3ª série/ 4°ano do ensino fundamental", 
                                                                "4ª série/ 5°ano do ensino fundamental", "5ª série/ 6°ano do ensino fundamental", 
                                                                "6ª série/ 7°ano do ensino fundamental", "7ª série/ 8°ano do ensino fundamental", 
                                                                "8ª série/ 9°ano do ensino fundamental", "1°ano do ensino médio", "2°ano do ensino médio", 
                                                                "3°ano do ensino médio", "Ensino superior incompleto", "Ensino superior completo"])
    selecao_renda = st.selectbox("Faixa de Renda da casa", ["Sem renda", "Até R$ 1.000,00", "De R$ 1.001,00 até R$ 2.000,00", "De R$ 2.001,00 até R$ 3.000,00", 
                                                    "De R$ 3.001,00 até R$ 5.000,00", "De R$ 5.001,00 até R$ 10.000,00", "R$ 10.001,00 ou mais"])
    # Adicionar as features faltantes no dashboard
    selecao_cor_pessoa = st.selectbox("Cor da criança", ["Branca", "Preta", "Amarela (origem japonesa, chinesa, coreana etc.)", 
                                                    "Parda (mulata, cabocla, cafuza, mameluca ou mestiça)", 
                                                    "Indígena", "Não sabe/não quis responder"])

    selecao_moradores_alimentaram_sim = st.selectbox("Moradores que Alimentaram Acabamento (Sim)", ["Sim", "Não"])

    selecao_moradores_alimentaram_nao = st.selectbox("Moradores que Alimentaram Acabamento (Não)", ["Sim", "Não"])


    beneficios_opcoes = ["Programa Bolsa Família (PBF)", "Benefício de Prestação Continuada (BPC/LOAS)", "Bolsa ou benefício da Prefeitura Municipal", 
                         "Bolsa ou benefício do Governo do Estado", "Pensão", "Aposentadoria", "Outro benefício"]
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

    mapping = {"Região": {"Norte": 1, "Nordeste": 2, "Sudeste": 3, "Sul": 4, "Centro-Oeste": 5}, "Sexo": {"Masculino": 1, "Feminino": 2}, 
               "Domicílio": {"Casa": 1, "Apartamento": 2, "Outros": 3}, "Cozinha": {"Sim": 1, "Não": 0}, "Ocupação": {"Próprio de algum morador - já pago": 1, "Próprio de algum morador - ainda pagando": 2, 
               "Alugado": 3, "Cedido por empregador": 4, "Cedido de outra forma": 5, "Outra condição" : 6}, "Registro": {"Urbano": 1, "Rural": 2}, 
               "Tosse": {"Sim": 1, "Não": 2, "Não sabe/ não quis responder": 9}, "Respiração": {"Sim": 1, "Não": 2, "Não sabe/ não quis responder": 9}, 
               "Escolaridade": {"Sem estudo": 0,
               "1° ano do ensino fundamental": 1,
               "1ª série/ 2°ano do ensino fundamental": 2,
               "2ª série/ 3°ano do ensino fundamental": 3,
               "3ª série/ 4°ano do ensino fundamental": 4,
               "4ª série/ 5°ano do ensino fundamental": 5,
               "5ª série/ 6°ano do ensino fundamental": 6,
               "6ª série/ 7°ano do ensino fundamental": 7,
               "7ª série/ 8°ano do ensino fundamental": 8,
               "8ª série/ 9°ano do ensino fundamental": 9,
               "1°ano do ensino médio": 10,
               "2°ano do ensino médio": 11,
               "3°ano do ensino médio": 12,
               "Ensino superior incompleto": 13,
               "Ensino superior completo": 14}, "Renda": {"Sem renda": 1, "Até R$ 1.000,00": 2, "De R$ 1.001,00 até R$ 2.000,00": 3,
               "De R$ 2.001,00 até R$ 3.000,00": 4, "De R$ 3.001,00 até R$ 5.000,00": 5,
               "De R$ 5.001,00 até R$ 10.000,00": 6, "R$ 10.001,00 ou mais": 7}}
    mapping_cor_pessoa = {
    "Branca": 1,
    "Preta": 2,
    "Amarela (origem japonesa, chinesa, coreana etc.)": 3,
    "Parda (mulata, cabocla, cafuza, mameluca ou mestiça)": 4,
    "Indígena": 5,
    "Não sabe/não quis responder": 9
    }

    mapping_sim_nao = {
        "Sim": 1, 
        "Não": 2
    }

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
        total_beneficios,  # Adiciona total de benefícios
        mapping_cor_pessoa[selecao_cor_pessoa],  # Adiciona cor pessoa
        mapping_sim_nao[selecao_moradores_alimentaram_sim],  # Moradores alimentaram acabamento SIM
        mapping_sim_nao[selecao_moradores_alimentaram_nao]   # Moradores alimentaram acabamento NÃO
    ]

    mapping_alimentos_reverse = {
        1: "Não",
        2: "Sim, raramente",
        3: "Sim, às vezes",
        4: "Sim, quase sempre",
        5: "Sim, sempre",
        6: "Não se cozinha em casa"
    }

    if st.button("Prever Qualidade da Alimentação"):
        resultado = modelo.predict([input_data])[0]
        probabilidade = modelo.predict_proba([input_data]).max()

        st.success(f"🍽️ O modelo previu a qualidade da alimentação como: **{mapping_alimentos_reverse[resultado]}**")
        st.write(f"Confiança da predição: {probabilidade:.2%}")

# ----------------------------------------------------------
# Insights e Recomendações
# ----------------------------------------------------------
st.markdown('<div class="sub-header">Insights e Recomendações</div>', unsafe_allow_html=True)
st.markdown('<div class="section">', unsafe_allow_html=True)
st.markdown("""
<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 1.5rem;">
    <div style="background-color: #e1ebff; padding: 1.5rem; border-radius: 12px;">
        <div style="display: flex; align-items: center; margin-bottom: 1rem;">
            <div style="background-color: #3a86ff; border-radius: 50%; width: 36px; height: 36px; display: flex; justify-content: center; align-items: center; margin-right: 0.75rem;">
                <span style="font-size: 1.2rem; color: white;">📊</span>
            </div>
            <h4 style="margin: 0; color: #0043a9; font-size: 1.2rem;">Disparidades Regionais</h4>
        </div>
        <p style="margin: 0; color: #555b6e;">
            Os dados revelam diferenças significativas entre as regiões do Brasil,
            com o Norte e Nordeste apresentando indicadores mais baixos de desenvolvimento infantil.
            Políticas públicas devem priorizar estas regiões com intervenções específicas.
        </p>
    </div>
    <div style="background-color: #e3f5e1; padding: 1.5rem; border-radius: 12px;">
        <div style="display: flex; align-items: center; margin-bottom: 1rem;">
            <div style="background-color: #38b000; border-radius: 50%; width: 36px; height: 36px; display: flex; justify-content: center; align-items: center; margin-right: 0.75rem;">
                <span style="font-size: 1.2rem; color: white;">🍎</span>
            </div>
            <h4 style="margin: 0; color: #38b000; font-size: 1.2rem;">Acesso a Alimentos</h4>
        </div>
        <p style="margin: 0; color: #555b6e;">
            O acesso a alimentos básicos está fortemente correlacionado com a renda familiar.
            Programas de transferência de renda e alimentação escolar devem ser fortalecidos
            para garantir nutrição adequada às crianças em situação de vulnerabilidade.
        </p>
    </div>
    <div style="background-color: #fff4e1; padding: 1.5rem; border-radius: 12px;">
        <div style="display: flex; align-items: center; margin-bottom: 1rem;">
            <div style="background-color: #ff9e00; border-radius: 50%; width: 36px; height: 36px; display: flex; justify-content: center; align-items: center; margin-right: 0.75rem;">
                <span style="font-size: 1.2rem; color: white;">🏠</span>
            </div>
            <h4 style="margin: 0; color: #ff9e00; font-size: 1.2rem;">Infraestrutura Domiciliar</h4>
        </div>
        <p style="margin: 0; color: #555b6e;">
            A presença de cozinha adequada impacta diretamente na capacidade das famílias
            de preparar refeições nutritivas. Programas habitacionais devem considerar
            a adequação dos espaços para preparação de alimentos como prioridade.
        </p>
    </div>
</div>
""", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown("""
<div class="footer">
    <div style="display: flex; justify-content: center; align-items: center; gap: 10px; margin-bottom: 20px;">
        <span style="font-size: 28px;">👶</span>
        <h2 style="margin: 0; font-size: 1.8rem; font-weight: 700;">Análise de Desnutrição Infantil no Brasil</h2>
        <span style="font-size: 28px;">👶</span>
    </div>
    <p style="font-size: 1.1rem; max-width: 700px; margin: 0 auto 20px auto;">
        Trabalhando por um futuro onde todas as crianças brasileiras tenham acesso à nutrição adequada e condições para um desenvolvimento saudável.
    </p>
    <div style="display: flex; justify-content: center; gap: 20px; margin-bottom: 20px;">
        <div style="text-align: center;">
            <div style="font-size: 24px; margin-bottom: 5px;">📊</div>
            <div>Análise de Dados</div>
        </div>
        <div style="text-align: center;">
            <div style="font-size: 24px; margin-bottom: 5px;">🔍</div>
            <div>Pesquisa</div>
        </div>
        <div style="text-align: center;">
            <div style="font-size: 24px; margin-bottom: 5px;">📋</div>
            <div>Políticas Públicas</div>
        </div>
        <div style="text-align: center;">
            <div style="font-size: 24px; margin-bottom: 5px;">🤝</div>
            <div>Parcerias</div>
        </div>
    </div>
    <p style="font-size: 0.9rem; margin-top: 20px; opacity: 0.8;">© 2025 Análise de Desnutrição Infantil | Todos os direitos reservados</p>
</div>
""", unsafe_allow_html=True)
