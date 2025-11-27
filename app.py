import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, date

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="VI Trucks - Analytics", page_icon="🚛", layout="wide")

# --- CARGA DE DATOS ---
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('simulacion_piloto_60dias_CPG.csv')
        df['Fecha Ingreso'] = pd.to_datetime(df['Fecha Ingreso'])
        return df
    except FileNotFoundError:
        return pd.DataFrame()

df = load_data()

# --- HEADER ---
st.title("🚛 Dashboard Inteligente - CPG Chile")
st.markdown("---")

if df.empty:
    st.error("❌ No se encontró el archivo CSV. Sube 'simulacion_piloto_60dias_CPG.csv' al repositorio.")
else:
    # --- BARRA LATERAL (FILTROS + DINERO) ---
    st.sidebar.header("🔍 Filtros de Análisis")
    
    # 1. Filtro de Fechas
    min_date = df['Fecha Ingreso'].min().date()
    max_date = df['Fecha Ingreso'].max().date()
    date_range = st.sidebar.date_input("Período:", value=(min_date, max_date), min_value=min_date, max_value=max_date)
    
    if len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date, end_date = min_date, max_date

    # 2. Filtros Categoricos
    all_companies = sorted(df['Empresa'].unique())
    sel_companies = st.sidebar.multiselect("Empresas:", all_companies, default=all_companies)
    
    all_materials = sorted(df['Material (IA Class)'].unique())
    sel_materials = st.sidebar.multiselect("Materiales:", all_materials, default=all_materials)

    # 3. MÓDULO FINANCIERO (NUEVO 💰)
    st.sidebar.markdown("---")
    st.sidebar.header("💰 Parámetros ROI")
    precio_m3 = st.sidebar.number_input(
        "Precio por m³ (CLP):", 
        value=12000, 
        step=500,
        help="Valor promedio para calcular ahorro por ajuste de volumen."
    )

    # --- LÓGICA DE FILTRADO ---
    mask = (
        (df['Fecha Ingreso'].dt.date >= start_date) &
        (df['Fecha Ingreso'].dt.date <= end_date) &
        (df['Empresa'].isin(sel_companies)) &
        (df['Material (IA Class)'].isin(sel_materials))
    )
    df_filtered = df[mask].copy() # Usamos .copy() para evitar warnings

    # --- CÁLCULOS FINANCIEROS ---
    # Calculamos la diferencia: Lo que declararon - Lo que realmente traían (IA)
    # Si la diferencia es positiva, significa que declararon más de lo real = Ahorro para CPG
    df_filtered['Dif_Volumen'] = df_filtered['Vol. Declarado (m³)'] - df_filtered['Vol. IA (m³)']
    df_filtered['Ahorro_CLP'] = df_filtered['Dif_Volumen'].apply(lambda x: x * precio_m3 if x > 0 else 0)

    # --- PANEL DE CONTROL ---
    if df_filtered.empty:
        st.warning("⚠️ No hay datos para los filtros seleccionados.")
    else:
        st.subheader(f"Resumen del Período: {start_date} al {end_date}")
        
        # KPIs (Ahora son 5)
        k1, k2, k3, k4, k5 = st.columns(5)
        
        total_camiones = len(df_filtered)
        vol_total = df_filtered['Vol. IA (m³)'].sum()
        precision = df_filtered['Precisión (%)'].mean()
        rechazos = (len(df_filtered[df_filtered['Contaminación (%)'] > 2.0]) / total_camiones) * 100
        ahorro_total = df_filtered['Ahorro_CLP'].sum()
        
        k1.metric("🚛 Camiones", total_camiones)
        k2.metric("📦 Volumen (m³)", f"{vol_total:,.0f}")
        k3.metric("🎯 Precisión IA", f"{precision:.1f}%")
        k4.metric("⚠️ Tasa Rechazo", f"{rechazos:.1f}%")
        
        # EL KPI DE DINERO
        k5.metric(
            "💰 Ahorro Estimado", 
            f"${ahorro_total:,.0f}", 
            delta="Dinero Recuperado", 
            delta_color="normal"
        )

        # --- GRÁFICOS ---
        c1, c2 = st.columns(2)
        with c1:
            # Evolución Volumen
            daily = df_filtered.groupby('Fecha Ingreso')['Vol. IA (m³)'].sum().reset_index()
            fig1 = px.bar(daily, x='Fecha Ingreso', y='Vol. IA (m³)', title="📈 Volumen Diario", color_discrete_sequence=['#3182bd'])
            st.plotly_chart(fig1, use_container_width=True)
            
        with c2:
            # Ranking Financiero por Empresa (Quién infla más el volumen)
            roi_empresa = df_filtered.groupby('Empresa')['Ahorro_CLP'].sum().reset_index().sort_values('Ahorro_CLP', ascending=True)
            fig2 = px.bar(roi_empresa, x='Ahorro_CLP', y='Empresa', orientation='h', 
                          title="💸 Ahorro Generado por Empresa (Auditado)",
                          color='Ahorro_CLP', color_continuous_scale='Greens')
            st.plotly_chart(fig2, use_container_width=True)

        # --- DETALLE DE REGISTROS ---
        with st.expander("📝 Ver Detalle de Registros y Ahorros"):
            cols = ['Fecha Ingreso', 'Patente', 'Empresa', 'Material (IA Class)', 
                    'Vol. Declarado (m³)', 'Vol. IA (m³)', 'Contaminación (%)', 'Ahorro_CLP']
            st.dataframe(df_filtered[cols].sort_values('Fecha Ingreso', ascending=False), use_container_width=True)