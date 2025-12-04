import streamlit as st
import pandas as pd
import plotly.express as px

# Configuración de página
st.set_page_config(page_title="VI Trucks - Monitor", page_icon="🚛", layout="wide")

# Cargar datos
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('simulacion_piloto_60dias_CPG.csv')
        df['Fecha Ingreso'] = pd.to_datetime(df['Fecha Ingreso'])
        return df
    except FileNotFoundError:
        return pd.DataFrame()

df = load_data()

# Título
st.title("🚛 Monitor CPG Chile - En Vivo")

if df.empty:
    st.error("No se encontró el archivo CSV. Asegúrate de subirlo al repositorio.")
else:
    # Sidebar Filtros
    st.sidebar.header("Filtros")
    materiales = ['Todos'] + sorted(list(df['Material (IA Class)'].unique()))
    opcion = st.sidebar.selectbox("Material:", materiales)

    # Filtrar
    if opcion != 'Todos':
        df_view = df[df['Material (IA Class)'] == opcion]
    else:
        df_view = df

    # KPIs
    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric("Camiones", len(df_view))
    kpi2.metric("Volumen Total", f"{df_view['Vol. IA (m³)'].sum():,.0f} m³")
    kpi3.metric("Precisión Promedio", f"{df_view['Precisión (%)'].mean():.1f}%")

    # Gráficos
    col1, col2 = st.columns(2)
    
    # Gráfico 1
    daily = df_view.groupby('Fecha Ingreso')['Vol. IA (m³)'].sum().reset_index()
    fig1 = px.line(daily, x='Fecha Ingreso', y='Vol. IA (m³)', title='Tendencia Diaria')
    col1.plotly_chart(fig1, use_container_width=True)

    # Gráfico 2
    fig2 = px.scatter(df_view, x='Vol. Declarado (m³)', y='Precisión (%)', color='Empresa', title='Precisión vs Carga')
    fig2.add_hline(y=90, line_dash="dash", line_color="red")
    col2.plotly_chart(fig2, use_container_width=True)

    # Alertas
    st.subheader("⚠️ Alertas de Contaminación (>2%)")
    alertas = df_view[df_view['Contaminación (%)'] > 2.0][['Fecha Ingreso','Patente','Empresa','Contaminación (%)']]
    if not alertas.empty:
        st.dataframe(alertas.sort_values('Contaminación (%)', ascending=False), use_container_width=True)
    else:
        st.success("Sin alertas críticas.")
