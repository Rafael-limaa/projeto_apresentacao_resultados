import streamlit as st 
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import missingno as msno
import io



csv_file = 'student_habits_performance.csv'
df = pd.read_csv(csv_file)

df = df.rename(columns={"student_id": "ID do Estudante",
    "age": "Idade",
    "gender": "Gênero",
    "study_hours_per_day": "Horas de Estudo por Dia",
    "social_media_hours": "Horas em Mídias Sociais",
    "netflix_hours": "Horas no Netflix",
    "part_time_job": "Trabalho de Meio Período",
    "attendance_percentage": "Percentual de Frequência",
    "sleep_hours": "Horas de Sono",
    "diet_quality": "Qualidade da Dieta",
    "exercise_frequency": "Frequência de Exercícios",
    "parental_education_level": "Escolaridade dos Pais",
    "internet_quality": "Qualidade da Internet",
    "mental_health_rating": "Avaliação de Saúde Mental",
    "extracurricular_participation": "Participação em Atividades Extracurriculares",
    "exam_score": "Nota da Prova"})



# Configurações iniciais do Streamlit
st.set_page_config(page_title="Student Habits", layout="wide")

##   usei um pouco de html para evitar a quebra de texto no app ##
st.markdown(
    "<h1 style='font-sine: 36px; white-space: nowrap;'> Hábitos dos Estudantes vs. Desempenho Acadêmico </h1>",
    unsafe_allow_html=True) 

## Seleção de Dados
with st.sidebar:
 st.sidebar.header("Filtros de Seleção")
 select_genero = st.sidebar.multiselect("Gênero:", options=df['Gênero'].unique(), default=list(df['Gênero'].unique()))
 idade_min = int(df["Idade"].min())
 idade_max = int(df["Idade"].max())
 select_age = st.sidebar.slider("Idade", min_value=idade_min, max_value=idade_max, value=(idade_min, idade_max))
 select_job = st.sidebar.multiselect("Trabalho de Meio Período:", options=df['Trabalho de Meio Período'].unique(),default=list(df['Trabalho de Meio Período'].unique()))
 select_diet = st.sidebar.multiselect("Qualidade da Dieta:", options=df['Qualidade da Dieta'].unique(), default=list(df['Qualidade da Dieta'].unique()))
 frequency_min = int(df["Percentual de Frequência"].min())
 frequency_max = int(df["Percentual de Frequência"].max())
 select_frequency = st.sidebar.slider("Percentual de Frequência", min_value=frequency_min, max_value=frequency_max, value=(frequency_min, frequency_max))
 select_parental_education = st.sidebar.multiselect("Escolaridade dos Pais:", options=df['Escolaridade dos Pais'].unique(), default=list(df['Escolaridade dos Pais'].unique()))


# Filtrando os dados
filtered_df = df[
    (df['Gênero'].isin(select_genero)) &
    (df['Idade'].between(select_age[0], select_age[1])) &
    (df['Trabalho de Meio Período'].isin(select_job)) &
    (df['Qualidade da Dieta'].isin(select_diet)) &
    (df['Percentual de Frequência'].between(select_frequency[0], select_frequency[1])) &
    (df['Escolaridade dos Pais'].isin(select_parental_education))
]



st.write("As Analises a seguir refletem uma banco de dados simulado que comtempla cerca de mil registros com hábitos sociais e como eles afetam o desempenho educacional.")



## Análises exploratória 

st.markdown("### Visão Geral do Dataset")

st.subheader("Primeiras linhas")
st.dataframe(filtered_df.head(10))


st.subheader("Últimas linhas")
st.dataframe(filtered_df.tail(10))


# st.dataframe(filtered_df.dtypes)

st.subheader("Informações sobre data type e valores nulos")
buffer = io.StringIO()
filtered_df.info(buf=buffer)
s = buffer.getvalue()
st.code(s)


st.subheader("Dimensões")
st.write(f"{filtered_df.shape[0]} linhas × {filtered_df.shape[1]} colunas")

st.subheader("Resumo estatístico")
st.dataframe(filtered_df.describe())






# Explicação do Streamlit
st.markdown("## Sobre o Streamlit")
st.write("""
    **Streamlit** é uma biblioteca de Python de código aberto que permite criar e compartilhar aplicativos de dados 
    interativos de forma fácil e rápida. Ela transforma scripts em uma interface de usuário web amigável e intuitiva 
    sem a necessidade de conhecimento em desenvolvimento web. Abaixo estão os principais conceitos utilizados neste exemplo:

    - `st.title()`: Adiciona um título ao seu aplicativo.
    - `st.markdown()`: Permite adicionar textos em formato Markdown.
    - `st.dataframe()`: Exibe um DataFrame do Pandas.
    - `st.sidebar`: Permite adicionar componentes de entrada e seleção na barra lateral do aplicativo.
    - `st.multiselect()`: Adiciona uma caixa de seleção múltipla.
    - `st.date_input()`: Adiciona um componente de seleção de data.
    - `st.bar_chart()`: Cria um gráfico de barras.
    - `st.line_chart()`: Cria um gráfico de linha.
    """)

