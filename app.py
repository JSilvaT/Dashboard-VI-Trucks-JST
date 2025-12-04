import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="VI Trucks - Analytics Pro", page_icon="🚛", layout="wide")

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
st.title("🚛 Dashboard VI Trucks - Análisis & Finanzas")
st.markdown("---")

if df.empty:
    st.error("❌ Error: No se encuentra 'simulacion_piloto_60dias_CPG.csv'")
else:
    # --- SIDEBAR (FILTROS + FINANZAS) ---
    st.sidebar.header("🔍 Configuración")
    
    # Filtros de Fecha
    min_date = df['Fecha Ingreso'].min().date()
    max_date = df['Fecha Ingreso'].max().date()
    date_range = st.sidebar.date_input("Período:", value=(min_date, max_date), min_value=min_date, max_value=max_date)
    start_date, end_date = date_range if len(date_range) == 2 else (min_date, max_date)
    
    # Filtros Categoría
    sel_companies = st.sidebar.multiselect("Empresas:", sorted(df['Empresa'].unique()), default=sorted(df['Empresa'].unique()))
    sel_materials = st.sidebar.multiselect("Materiales:", sorted(df['Material (IA Class)'].unique()), default=sorted(df['Material (IA Class)'].unique()))
    
    # Módulo Financiero
    st.sidebar.markdown("---")
    st.sidebar.header("💰 Parámetros ROI")
    precio_m3 = st.sidebar.number_input("Precio m³ (CLP):", value=12000, step=500)

    # Filtrado
    mask = (
        (df['Fecha Ingreso'].dt.date >= start_date) &
        (df['Fecha Ingreso'].dt.date <= end_date) &
        (df['Empresa'].isin(sel_companies)) &
        (df['Material (IA Class)'].isin(sel_materials))
    )
    df_filtered = df[mask].copy()

    # Cálculos Financieros
    df_filtered['Dif_Volumen'] = df_filtered['Vol. Declarado (m³)'] - df_filtered['Vol. IA (m³)']
    df_filtered['Ahorro_CLP'] = df_filtered['Dif_Volumen'].apply(lambda x: x * precio_m3 if x > 0 else 0)
    df_filtered['Error_Abs'] = abs(df_filtered['Vol. IA (m³)'] - df_filtered['Vol. Declarado (m³)'])

    # --- KPIs PRINCIPALES ---
    if not df_filtered.empty:
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("🚛 Camiones", len(df_filtered))
        k2.metric("📦 Volumen (m³)", f"{df_filtered['Vol. IA (m³)'].sum():,.0f}")
        k3.metric("🎯 Precisión Prom.", f"{df_filtered['Precisión (%)'].mean():.1f}%")
        k4.metric("⚠️ Rechazos (>2%)", f"{(len(df_filtered[df_filtered['Contaminación (%)'] > 2.0])/len(df_filtered)*100):.1f}%")
        k5.metric("💰 Ahorro Est.", f"${df_filtered['Ahorro_CLP'].sum():,.0f}", delta="ROI Positivo")

        st.markdown("### 📊 Análisis Técnico Detallado (Los 4 Gráficos)")
        
        # --- FILA 1 DE GRÁFICOS ---
        row1_col1, row1_col2 = st.columns(2)
        
        with row1_col1:
            # 1. HISTOGRAMA COMPARATIVO (Declarado vs IA)
            # Truco para superponer en Plotly:
            fig1 = go.Figure()
            fig1.add_trace(go.Histogram(x=df_filtered['Vol. Declarado (m³)'], name='Declarado', opacity=0.75, marker_color='skyblue'))
            fig1.add_trace(go.Histogram(x=df_filtered['Vol. IA (m³)'], name='IA (Real)', opacity=0.75, marker_color='orange'))
            fig1.update_layout(title="1. Distribución de Carga: Declarado vs IA", barmode='overlay')
            st.plotly_chart(fig1, use_container_width=True)
            
        with row1_col2:
            # 2. BOXPLOT DE PRECISIÓN (Variabilidad por Material)
            fig2 = px.box(df_filtered, x='Material (IA Class)', y='Precisión (%)', color='Material (IA Class)',
                          title="2. Variabilidad de Precisión por Material")
            fig2.add_hline(y=90, line_dash="dash", line_color="red", annotation_text="Meta 90%")
            st.plotly_chart(fig2, use_container_width=True)

        # --- FILA 2 DE GRÁFICOS ---
        row2_col1, row2_col2 = st.columns(2)
        
        with row2_col1:
            # 3. SERIE DE TIEMPO (Tendencia Diaria)
            daily_stats = df_filtered.groupby('Fecha Ingreso')['Precisión (%)'].mean().reset_index()
            fig3 = px.line(daily_stats, x='Fecha Ingreso', y='Precisión (%)', markers=True,
                           title="3. Evolución de la Precisión Diaria", line_shape='spline')
            fig3.add_hline(y=90, line_dash="dash", line_color="red")
            st.plotly_chart(fig3, use_container_width=True)
            
        with row2_col2:
            # 4. SCATTER PLOT (Correlación Error vs Volumen)
            fig4 = px.scatter(df_filtered, x='Vol. Declarado (m³)', y='Error_Abs', color='Material (IA Class)',
                              title="4. Correlación: Volumen de Carga vs Error (m³)",
                              size='Contaminación (%)', hover_data=['Empresa'])
            st.plotly_chart(fig4, use_container_width=True)

        # --- BONUS: GRÁFICO FINANCIERO ---
        st.markdown("### 💸 Visión Financiera")
        roi_empresa = df_filtered.groupby('Empresa')['Ahorro_CLP'].sum().reset_index().sort_values('Ahorro_CLP', ascending=True)
        fig5 = px.bar(roi_empresa, x='Ahorro_CLP', y='Empresa', orientation='h', 
                      title="Ranking de Ahorro Generado por Empresa", color='Ahorro_CLP', color_continuous_scale='Greens')
        st.plotly_chart(fig5, use_container_width=True)

        # --- TABLA DETALLE ---
        with st.expander("📝 Ver Detalle de Registros"):
            st.dataframe(df_filtered.sort_values('Fecha Ingreso', ascending=False), use_container_width=True)
            
    else:
        st.warning("⚠️ No hay datos para mostrar.")
