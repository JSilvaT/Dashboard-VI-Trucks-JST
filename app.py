import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(
    page_title="VI Trucks JST - Dashboard",
    page_icon="🚚",
    layout="wide"
)

# --- TÍTULO Y DESCRIPCIÓN ---
st.title("🚚 VI Trucks JST: Monitor de Visión Artificial")
st.markdown("""
**Cliente:** CPG Chile | **Versión:** MVP Piloto 1.0
Este dashboard visualiza en tiempo real la precisión del sistema de perfilometría LiDAR 2D.
""")

# --- MÓDULO DE CARGA DE DATOS ---
@st.cache_data # Esto hace que la app sea rápida, cargando datos una sola vez
def cargar_datos():
    try:
        # Asegúrate de subir el archivo .csv junto con este script
        df = pd.read_csv('simulacion_piloto_60dias_CPG.csv')
        return df
    except FileNotFoundError:
        return None

df = cargar_datos()

if df is None:
    st.error("⚠️ Error: No se encontró el archivo 'simulacion_piloto_60dias_CPG.csv'. Por favor cárguelo en el repositorio.")
else:
    # --- SIDEBAR (FILTROS) ---
    st.sidebar.header("Filtros de Visualización")
    materiales = df['Material (IA Class)'].unique()
    seleccion_material = st.sidebar.multiselect("Filtrar por Material", materiales, default=materiales)
    
    # Filtrar el dataframe según la selección
    df_filtered = df[df['Material (IA Class)'].isin(seleccion_material)]

    # --- KPIs PRINCIPALES (MÉTRICAS) ---
    st.markdown("### 📊 KPIs Operativos (Tiempo Real)")
    
    col1, col2, col3 = st.columns(3)
    
    precision_promedio = df_filtered['Precisión (%)'].mean()
    total_camiones = len(df_filtered)
    fallos = len(df_filtered[df_filtered['Precisión (%)'] < 90])
    
    col1.metric("Precisión Global", f"{precision_promedio:.2f}%", "Meta: >90%")
    col2.metric("Camiones Procesados", f"{total_camiones}", "Unidades")
    col3.metric("Tasa de Fallos (<90%)", f"{fallos}", f"-{(fallos/total_camiones)*100:.1f}%", delta_color="inverse")

    st.divider()

    # --- VISUALIZACIÓN GRÁFICA ---
    col_izq, col_der = st.columns(2)

    with col_izq:
        st.subheader("A. Correlación Manual vs IA")
        fig1 = plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df_filtered, x='Vol. Declarado (m³)', y='Vol. IA (m³)', 
                        hue='Material (IA Class)', palette='viridis', s=100, alpha=0.7)
        plt.plot([10, 20], [10, 20], 'r--', lw=2, label='Ideal')
        plt.legend()
        st.pyplot(fig1)

    with col_der:
        st.subheader("B. Precisión por Material")
        fig2 = plt.figure(figsize=(10, 6))
        sns.boxplot(data=df_filtered, x='Material (IA Class)', y='Precisión (%)', palette='Set2')
        plt.axhline(90, color='red', linestyle='--', label='Meta KPI')
        st.pyplot(fig2)

    st.subheader("C. Estabilidad del Sistema (Histograma)")
    fig3 = plt.figure(figsize=(12, 4))
    sns.histplot(df_filtered['Precisión (%)'], bins=30, kde=True, color='green')
    plt.axvline(90, color='red', linestyle='--', label='Umbral 90%')
    st.pyplot(fig3)

    # --- TABLA DE DATOS ---
    with st.expander("Ver Datos Crudos"):
        st.dataframe(df_filtered)

    # --- PIE DE PÁGINA ---
    st.caption("Sistema desarrollado por Jorge Silva Tapia para Proyecto IDA300 - UNAB")
