import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(
    page_title="VI Trucks JST - Dashboard Pro",
    page_icon="🚚",
    layout="wide"
)

# --- ESTILOS CSS PERSONALIZADOS ---
st.markdown("""
    <style>
    .metric-card {
        background-color: #f0f2f6;
        border-left: 5px solid #ff4b4b;
        padding: 15px;
        border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- TÍTULO Y DESCRIPCIÓN ---
st.title("🚚 VI Trucks JST: Centro de Control de Visión Artificial")
st.markdown("""
**Cliente:** CPG Chile | **Versión:** Piloto Avanzado 2.0
Monitoreo en tiempo real de la precisión volumétrica con filtros dinámicos de operación.
""")

# --- MÓDULO DE CARGA DE DATOS ---
@st.cache_data
def cargar_datos():
    try:
        df = pd.read_csv('simulacion_piloto_60dias_CPG.csv')
        # Convertir fecha a objeto datetime real para poder filtrar
        df['Fecha Ingreso'] = pd.to_datetime(df['Fecha Ingreso'])
        return df
    except FileNotFoundError:
        return None

df = cargar_datos()

if df is None:
    st.error("⚠️ Error: No se encontró el archivo 'simulacion_piloto_60dias_CPG.csv'.")
else:
    # ==========================================
    # BARRA LATERAL (FILTROS INTELIGENTES)
    # ==========================================
    st.sidebar.header("🔍 Filtros de Operación")
    
    # 1. Filtro de Fechas (Semana a Semana)
    min_date = df['Fecha Ingreso'].min()
    max_date = df['Fecha Ingreso'].max()
    
    fechas_sel = st.sidebar.date_input(
        "Rango de Fechas",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )

    # 2. Filtro de Materiales
    st.sidebar.divider()
    materiales = df['Material (IA Class)'].unique()
    seleccion_material = st.sidebar.multiselect(
        "Tipo de Material", 
        options=materiales, 
        default=materiales
    )
    
    # 3. Filtro de Empresas (NUEVO VALOR)
    empresas = df['Empresa'].unique()
    seleccion_empresa = st.sidebar.multiselect(
        "Empresa Transportista",
        options=empresas,
        default=empresas
    )

    # --- APLICAR FILTROS AL DATAFRAME ---
    # Validar que las fechas sean una tupla (inicio, fin) para evitar errores
    if isinstance(fechas_sel, tuple) and len(fechas_sel) == 2:
        start_date, end_date = fechas_sel
        mask_date = (df['Fecha Ingreso'].dt.date >= start_date) & (df['Fecha Ingreso'].dt.date <= end_date)
    else:
        mask_date = pd.Series([True] * len(df)) # Si no hay rango completo, muestra todo

    df_filtered = df[
        mask_date & 
        df['Material (IA Class)'].isin(seleccion_material) &
        df['Empresa'].isin(seleccion_empresa)
    ]

    # ==========================================
    # DASHBOARD PRINCIPAL
    # ==========================================
    
    if df_filtered.empty:
        st.warning("⚠️ No hay datos con los filtros seleccionados. Intenta ampliar el rango.")
    else:
        # --- KPIs SUPERIORES ---
        st.markdown("### 📊 Indicadores Clave de Desempeño (KPIs)")
        col1, col2, col3, col4 = st.columns(4)
        
        precision_prom = df_filtered['Precisión (%)'].mean()
        total_ops = len(df_filtered)
        # Calcular volumen total movido en el periodo
        volumen_total = df_filtered['Vol. IA (m³)'].sum()
        tasa_fallos = (len(df_filtered[df_filtered['Precisión (%)'] < 90]) / total_ops) * 100
        
        col1.metric("Precisión Promedio", f"{precision_prom:.2f}%", delta="Meta > 90%")
        col2.metric("Total Camiones", f"{total_ops}", delta="Operaciones")
        col3.metric("Volumen Procesado", f"{volumen_total:,.0f} m³", delta="Acumulado")
        col4.metric("Tasa de Error", f"{tasa_fallos:.1f}%", delta_color="inverse")

        st.divider()

        # --- FILA 1: EVOLUCIÓN TEMPORAL Y CORRELACIÓN ---
        col_izq, col_der = st.columns([2, 1])

        with col_izq:
            st.subheader("📈 Evolución de Precisión (Tendencia Diaria)")
            # Agrupar por día para ver la tendencia
            daily_trend = df_filtered.groupby('Fecha Ingreso')['Precisión (%)'].mean().reset_index()
            
            fig_trend = plt.figure(figsize=(10, 4))
            sns.lineplot(data=daily_trend, x='Fecha Ingreso', y='Precisión (%)', marker='o', color='green', linewidth=2)
            plt.axhline(90, color='red', linestyle='--', label='Meta 90%')
            plt.title("¿Estamos mejorando semana a semana?")
            plt.xticks(rotation=45)
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.legend()
            st.pyplot(fig_trend)

        with col_der:
            st.subheader("🎯 Exactitud por Material")
            fig_box = plt.figure(figsize=(5, 4))
            sns.boxplot(data=df_filtered, x='Material (IA Class)', y='Precisión (%)', palette='Set2')
            plt.axhline(90, color='red', linestyle='--')
            plt.xticks(rotation=45)
            st.pyplot(fig_box)

        # --- FILA 2: ANÁLISIS POR EMPRESA (NUEVO) ---
        st.subheader("🏢 Desempeño por Empresa Contratista")
        
        # Tabla interactiva ordenada por precisión (Ranking)
        ranking_empresas = df_filtered.groupby('Empresa')[['Precisión (%)', 'Vol. IA (m³)']].mean().sort_values('Precisión (%)', ascending=False)
        st.dataframe(ranking_empresas.style.highlight_between(left=0, right=90, subset=['Precisión (%)'], color='#ffcccc'), use_container_width=True)

        st.caption("Nota: Las celdas rojas indican promedio bajo el 90%.")
