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
    # --- BARRA LATERAL DE FILTROS (EL VALOR AGREGADO) ---
    st.sidebar.header("🔍 Filtros de Análisis")
    
    # 1. Filtro de Fechas (Semana a Semana)
    min_date = df['Fecha Ingreso'].min().date()
    max_date = df['Fecha Ingreso'].max().date()
    
    date_range = st.sidebar.date_input(
        "Seleccionar Período:",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    
    # Manejo de error si el usuario selecciona solo una fecha
    if len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date, end_date = min_date, max_date

    # 2. Filtro de Empresa (Multiselect)
    all_companies = sorted(df['Empresa'].unique())
    selected_companies = st.sidebar.multiselect(
        "Filtrar por Empresa:",
        options=all_companies,
        default=all_companies # Por defecto todas seleccionadas
    )

    # 3. Filtro de Material (Multiselect)
    all_materials = sorted(df['Material (IA Class)'].unique())
    selected_materials = st.sidebar.multiselect(
        "Filtrar por Material:",
        options=all_materials,
        default=all_materials
    )

    # --- LÓGICA DE FILTRADO ---
    mask = (
        (df['Fecha Ingreso'].dt.date >= start_date) &
        (df['Fecha Ingreso'].dt.date <= end_date) &
        (df['Empresa'].isin(selected_companies)) &
        (df['Material (IA Class)'].isin(selected_materials))
    )
    df_filtered = df[mask]

    # --- PANEL DE CONTROL (RESULTADOS FILTRADOS) ---
    if df_filtered.empty:
        st.warning("⚠️ No hay datos para los filtros seleccionados.")
    else:
        # KPIs Dinámicos
        st.subheader(f"Resumen del Período: {start_date} al {end_date}")
        
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        
        total_camiones = len(df_filtered)
        volumen_total = df_filtered['Vol. IA (m³)'].sum()
        precision_prom = df_filtered['Precisión (%)'].mean()
        # Nuevo KPI: Tasa de Rechazo (Contaminación > 2%)
        rechazos = len(df_filtered[df_filtered['Contaminación (%)'] > 2.0])
        tasa_rechazo = (rechazos / total_camiones) * 100
        
        kpi1.metric("🚛 Camiones", total_camiones)
        kpi2.metric("📦 Volumen (m³)", f"{volumen_total:,.0f}")
        kpi3.metric("🎯 Precisión IA", f"{precision_prom:.1f}%")
        kpi4.metric("⚠️ Tasa Contaminación", f"{tasa_rechazo:.1f}%", delta_color="inverse")

        # --- GRÁFICOS AVANZADOS ---
        col_graf1, col_graf2 = st.columns(2)

        with col_graf1:
            # Evolución Semanal/Diaria (Dependiendo del filtro)
            # Agrupamos por fecha para ver la curva
            daily_trend = df_filtered.groupby('Fecha Ingreso')['Vol. IA (m³)'].sum().reset_index()
            fig_trend = px.bar(
                daily_trend, 
                x='Fecha Ingreso', 
                y='Vol. IA (m³)',
                title="📈 Evolución de Volumen en el Período",
                color='Vol. IA (m³)',
                color_continuous_scale='Blues'
            )
            st.plotly_chart(fig_trend, use_container_width=True)

        with col_graf2:
            # Comparativa por Empresa (Nuevo Gráfico de Valor)
            company_perf = df_filtered.groupby('Empresa')[['Precisión (%)', 'Contaminación (%)']].mean().reset_index()
            fig_comp = px.scatter(
                company_perf,
                x='Precisión (%)',
                y='Contaminación (%)',
                color='Empresa',
                size='Contaminación (%)', # Burbuja más grande = más sucio
                title="🏢 Ranking de Empresas (Calidad vs Precisión)",
                hover_data=['Empresa']
            )
            # Líneas de referencia
            fig_comp.add_hline(y=2.0, line_dash="dash", line_color="red", annotation_text="Límite Contaminación")
            fig_comp.add_vline(x=90.0, line_dash="dash", line_color="green", annotation_text="Meta Precisión")
            st.plotly_chart(fig_comp, use_container_width=True)

        # --- DETALLE DE DATA (TABLA INTERACTIVA) ---
        with st.expander("📝 Ver Detalle de Registros (Click para desplegar)"):
            # AQUI AGREGAMOS 'Vol. Declarado (m³)' A LA LISTA DE COLUMNAS
            cols_to_show = [
                'Fecha Ingreso', 
                'Hora Ingreso', 
                'Patente', 
                'Empresa', 
                'Material (IA Class)', 
                'Vol. Declarado (m³)',  # <--- ¡Nueva columna recuperada!
                'Vol. IA (m³)', 
                'Precisión (%)', 
                'Contaminación (%)'
            ]
            
            st.dataframe(
                df_filtered[cols_to_show].sort_values(by='Fecha Ingreso', ascending=False),
                use_container_width=True
            )

