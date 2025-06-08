import streamlit as st 
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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

##  KPIs
media_nota = filtered_df["Nota da Prova"].mean()
mediana_nota = filtered_df["Nota da Prova"].median()
media_hora_estudo = filtered_df["Horas de Estudo por Dia"].mean()
mediana_hora_estudo = filtered_df["Horas de Estudo por Dia"].median()
media_hora_netflix = filtered_df["Horas no Netflix"].mean()
mediana_hora_netflix = filtered_df["Horas no Netflix"].median()
quantidade_estudate = filtered_df['ID do Estudante'].count()


kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric(label="Notas por Média e Mediana", value=f"{media_nota:.2f} - {mediana_nota:.2f}")
kpi2.metric(label="Horas de Estudo por Média e Mediana", value=f"{media_hora_estudo:.2f} - {mediana_hora_estudo:.2f}") 
kpi3.metric(label="Horas de Netflix por Média e Mediana", value=f"{media_hora_netflix:.2f} - {mediana_hora_netflix:.2f}")
kpi4.metric(label="Quantidade de Estudante", value=quantidade_estudate)




st.write("As Analises a seguir refletem uma banco de dados simulado que comtempla cerca de mil registros com hábitos sociais e como eles afetam o desempenho educacional.")

st.markdown("### Nossos dados são compostos por essas #features")
st.dataframe(filtered_df.head(10))

st.write("### Podemos observar a distruição do dataframe por gênero e destaca uma pequena concetração classificado como outros")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Por Escolaridade dos Pais")
    qtd_estudates_escolaridade_pais = filtered_df.groupby('Escolaridade dos Pais')['ID do Estudante'].count().reset_index()
    fig_product = px.bar(qtd_estudates_escolaridade_pais, x='Escolaridade dos Pais', y='ID do Estudante', title="Estudate por Formação do Pais",
                         labels={'Escolaridade dos Pais': 'Escolaridade dos Pais', 'ID do Estudante': 'Quantidade'})
    st.plotly_chart(fig_product, use_container_width=True)


with col2:
    st.markdown("### Por Gênero do Estudates")
    qtd_estudates_genero = filtered_df.groupby('Gênero')['ID do Estudante'].count().reset_index()
    fig_product = px.pie(qtd_estudates_genero, values='ID do Estudante', title="Estudate por Formação do Pais",
                         labels={'Gênero': 'Gênero dos Estudates', 'ID do Estudante': 'Quantidade'})
    st.plotly_chart(fig_product, use_container_width=True)




col3, col4 = st.columns(2)

with col3:
    st.markdown("### Por Qualidade da Internet")
    qtd_estudates_internet = filtered_df.groupby('Qualidade da Internet')['ID do Estudante'].count().reset_index()
    fig_product = px.pie(qtd_estudates_internet, values='ID do Estudante', title="Estudate por Qualidade da Internet",
                         labels={'Qualidade da Internet': 'Qualidade da Internet', 'ID do Estudante': 'Quantidade'})
    st.plotly_chart(fig_product, use_container_width=True)

with col4:
    st.markdown("### Por Qualidade da Dieta")
    qtd_estudates_dieta = filtered_df.groupby('Qualidade da Dieta')['ID do Estudante'].count().reset_index()
    fig_product = px.bar(qtd_estudates_dieta, x='Qualidade da Dieta', y='ID do Estudante', title="Estudate por Qualidade da Dieta",
                         labels={'Qualidade da Dieta': 'Qualidade da Dieta', 'ID do Estudante': 'Quantidade'})
    st.plotly_chart(fig_product, use_container_width=True)

# Aqui mostro um gráfico de barras do top 10 estudantes com mais horas de estudo
st.markdown("## Média de Horas de Estudo por Dia")
avg = filtered_df.groupby('ID do Estudante')['Horas de Estudo por Dia'].mean().sort_values(ascending=False).head(10)
fig, ax = plt.subplots(figsize=(10, 8))
avg.sort_values().plot(kind='barh', ax=ax, color='skyblue')
ax.set_xlabel('Média de Horas de Estudo por dia')
ax.set_ylabel("ID do Estudante")
ax.set_title("Top 10 Estudantes por Horas de Estudo")
st.pyplot(fig)



##  Foi usando uma tecnica para verificar correlação entre dados númericos
st.markdown("### Correlação entre váriaveis númericas")
df_number = filtered_df.select_dtypes(include='number')
st.dataframe(df_number.corr().style.background_gradient(cmap='coolwarm'))

## Tecnica de dispersão

st.subheader("Dispersão: Horas de Estudo x Nota da Prova")
fig, ax = plt.subplots(figsize=(10,6))
sns.scatterplot(data=filtered_df, x='Horas de Estudo por Dia', y='Nota da Prova', hue='Escolaridade dos Pais', palette='Set1',ax=ax)
ax.set_title('Horas de Estudo vs Nota da Prova')
ax.set_xlabel('Horas de Estudo por Dia')
ax.set_ylabel('Nota da Prova')
st.pyplot(fig)



# Dividindo os dados em treino e teste

X = df_number.drop(columns=['Nota da Prova'])
y = df_number['Nota da Prova']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# Modelo de ML

model = RandomForestRegressor(n_estimators=10, random_state=42)
model.fit(X_train, y_train)


# Previsão
y_pred = model.predict(X_test)


# Métricas de Avaliação

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.markdown("### Avaliação do Modelo")
st.write(f"**Mean Absolute Error (MAE):** {mae:,.2f}")
st.write(f"**Mean Squared Error (MSE):** {mse:,.2f}")
st.write(f"**R-squared (R2):** {r2:.2f}")

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

