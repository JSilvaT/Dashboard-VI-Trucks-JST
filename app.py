import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, date

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="VI Trucks - Auditoría 360", page_icon="🚛", layout="wide")

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
st.title("🚛 Dashboard de Control - CPG Chile")
st.markdown("---")

if df.empty:
    st.error("❌ No se encontró el archivo CSV. Sube 'simulacion_piloto_60dias_CPG.csv' al repositorio.")
else:
    # --- BARRA LATERAL (FILTROS + PARÁMETROS) ---
    st.sidebar.header("🔍 Filtros de Análisis")
    
    # 1. Filtro de Fechas
    min_date = df['Fecha Ingreso'].min().date()
    max_date = df['Fecha Ingreso'].max().date()
    date_range = st.sidebar.date_input("Período:", value=(min_date, max_date), min_value=min_date, max_value=max_date)
    
    if len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date, end_date = min_date, max_date

    # 2. Filtros Categóricos
    all_companies = sorted(df['Empresa'].unique())
    sel_companies = st.sidebar.multiselect("Empresas:", all_companies, default=all_companies)
    
    all_materials = sorted(df['Material (IA Class)'].unique())
    sel_materials = st.sidebar.multiselect("Materiales:", all_materials, default=all_materials)

    # 3. MÓDULO FINANCIERO (AUDITORÍA)
    st.sidebar.markdown("---")
    st.sidebar.header("💰 Parámetros Económicos")
    precio_m3 = st.sidebar.number_input(
        "Precio Promedio (CLP/m³):", 
        value=12000, 
        step=500,
        help="Valor base para calcular impacto financiero (Ahorro o Merma)."
    )

    # --- LÓGICA DE FILTRADO ---
    mask = (
        (df['Fecha Ingreso'].dt.date >= start_date) &
        (df['Fecha Ingreso'].dt.date <= end_date) &
        (df['Empresa'].isin(sel_companies)) &
        (df['Material (IA Class)'].isin(sel_materials))
    )
    df_filtered = df[mask].copy()

    # --- CÁLCULOS FINANCIEROS (EL CEREBRO DEL NEGOCIO) ---
    # Diferencia = Declarado (Papel) - IA (Realidad)
    df_filtered['Dif_Volumen'] = df_filtered['Vol. Declarado (m³)'] - df_filtered['Vol. IA (m³)']

    # CASO A: AHORRO (El camión traía MENOS de lo declarado)
    # Evitamos pagar por "aire".
    df_filtered['Ahorro_CLP'] = df_filtered['Dif_Volumen'].apply(lambda x: x * precio_m3 if x > 0 else 0)

    # CASO B: FUGA (El camión traía MÁS de lo declarado)
    # Estamos regalando material o recibiendo sobrecarga gratis (riesgo operativo).
    df_filtered['Fuga_CLP'] = df_filtered['Dif_Volumen'].apply(lambda x: abs(x) * precio_m3 if x < 0 else 0)

    # --- PANEL DE CONTROL (KPIs) ---
    if df_filtered.empty:
        st.warning("⚠️ No hay datos para los filtros seleccionados.")
    else:
        st.subheader(f"Resumen de Operaciones: {start_date} al {end_date}")
        
        # Ahora usamos 6 Columnas para incluir la Fuga
        k1, k2, k3, k4, k5, k6 = st.columns(6)
        
        total_camiones = len(df_filtered)
        vol_total = df_filtered['Vol. IA (m³)'].sum()
        precision = df_filtered['Precisión (%)'].mean()
        rechazos = (len(df_filtered[df_filtered['Contaminación (%)'] > 2.0]) / total_camiones) * 100
        ahorro_total = df_filtered['Ahorro_CLP'].sum()
        fuga_total = df_filtered['Fuga_CLP'].sum()
        
        k1.metric("🚛 Camiones", total_camiones)
        k2.metric("📦 Vol. Real (m³)", f"{vol_total:,.0f}")
        k3.metric("🎯 Precisión IA", f"{precision:.1f}%")
        k4.metric("⚠️ Tasa Rechazo", f"{rechazos:.1f}%")
        
        # LOS KPIS FINANCIEROS
        k5.metric(
            "💰 Ahorro (Aire)", 
            f"${ahorro_total:,.0f}", 
            delta="Retenido", 
            delta_color="normal",
            help="Dinero ahorrado al detectar carga MENOR a la declarada."
        )
        k6.metric(
            "💸 Fuga (Merma)", 
            f"${fuga_total:,.0f}", 
            delta="-Desviación", 
            delta_color="inverse",
            help="Valor del material excedente no declarado (Merma)."
        )

        # --- GRÁFICOS ---
        c1, c2 = st.columns(2)
        with c1:
            # Evolución Volumen
            daily = df_filtered.groupby('Fecha Ingreso')['Vol. IA (m³)'].sum().reset_index()
            fig1 = px.bar(daily, x='Fecha Ingreso', y='Vol. IA (m³)', title="📈 Flujo Diario de Material (m³)")
            st.plotly_chart(fig1, use_container_width=True)
            
        with c2:
            # Gráfico de Dispersión Financiera (Quién nos ahorra vs quién nos cuesta)
            # Eje X = Ahorro, Eje Y = Fuga
            roi_empresa = df_filtered.groupby('Empresa')[['Ahorro_CLP', 'Fuga_CLP']].sum().reset_index()
            fig2 = px.scatter(roi_empresa, x='Ahorro_CLP', y='Fuga_CLP', color='Empresa', size='Ahorro_CLP',
                              title="🏢 Análisis Financiero (Ahorro vs Fuga)",
                              labels={'Ahorro_CLP': 'Ahorro Generado ($)', 'Fuga_CLP': 'Fuga Detectada ($)'})
            st.plotly_chart(fig2, use_container_width=True)

        # --- DETALLE TABULAR ---
        with st.expander("📝 Ver Detalle de Registros (Auditoría Financiera)"):
            cols = ['Fecha Ingreso', 'Patente', 'Empresa', 'Material (IA Class)', 
                    'Vol. Declarado (m³)', 'Vol. IA (m³)', 'Ahorro_CLP', 'Fuga_CLP']
            st.dataframe(df_filtered[cols].sort_values('Fecha Ingreso', ascending=False), use_container_width=True)