import streamlit as st



st.set_page_config(
    page_title="Sejam Bem Vindo!",
    page_icon=":earth:"
)

st.sidebar.success("Escolha uma das opções:")


st.markdown(
    "<h2 style='text-align: center;'>Projeto Final para Disciplina de Comunicação e Apresentação de Resultados</h2>",
    unsafe_allow_html=True
)

st.markdown(
    "<h2 style='text-align: center;'>Tema: Hábitos dos alunos vs. desempenho acadêmico </h2>",
    unsafe_allow_html=True
)

st.subheader("Sobre o Conjunto de dados:")
st.write("Este é um conjunto de dados simulado que explora como os hábitos de vida afetam o desempenho acadêmico dos alunos. Com 1.000 registros sintéticos de alunos e mais de 15 recursos, incluindo horas de estudo, padrões de sono, uso de mídias sociais, qualidade da dieta, saúde mental e notas de exames finais")

st.subheader("Objetivo:")
st.write("###### Este projeto tem como objetivo analisar os hábitos e costumes sociais, buscando compreender de que forma esses comportamentos podem impactar o desempenho educacional.")

st.markdown( "### Grupo:\n"
    "Paulo Rafael  \n"
    "Paulo Passos  \n"
    "Washington França  \n"
    "Henrique Frazão"
)
