import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from datetime import timedelta

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="VI Trucks - AI Suite", page_icon="🚛", layout="wide")

# --- CARGA Y LIMPIEZA DE DATOS (ETL) ---
@st.cache_data
def load_data():
    try:
        # 1. Carga
        df = pd.read_csv('simulacion_piloto_60dias_CPG.csv')
        # 2. Transformación (Limpieza)
        df['Fecha Ingreso'] = pd.to_datetime(df['Fecha Ingreso'])
        # Eliminamos nulos si existieran (Buenas prácticas)
        df.dropna(inplace=True)
        return df
    except FileNotFoundError:
        return pd.DataFrame()

df = load_data()

# --- HEADER ---
st.title("🚛 VI Trucks - Plataforma de Inteligencia Artificial")
st.markdown("Integración de Monitoreo, Finanzas y Modelos Predictivos.")
st.markdown("---")

if df.empty:
    st.error("❌ Error: No se encuentra el dataset. Sube el CSV al repositorio.")
else:
    # --- TABS PRINCIPALES (Organización por Módulos) ---
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard Operativo", "🤖 Clustering (K-Means)", "🔮 Predicciones", "📋 Calidad de Datos"])

    # ==============================================================================
    # TAB 1: DASHBOARD OPERATIVO + FINANZAS (Lo que ya tenías mejorado)
    # ==============================================================================
    with tab1:
        # SIDEBAR (Solo visible aquí o global, lo dejamos global por simplicidad)
        st.sidebar.header("🔍 Configuración General")
        
        # Filtros de Fecha
        min_date = df['Fecha Ingreso'].min().date()
        max_date = df['Fecha Ingreso'].max().date()
        date_range = st.sidebar.date_input("Período Análisis:", value=(min_date, max_date))
        start_date, end_date = date_range if len(date_range) == 2 else (min_date, max_date)
        
        # Filtros Categoría
        sel_companies = st.sidebar.multiselect("Empresas:", sorted(df['Empresa'].unique()), default=sorted(df['Empresa'].unique()))
        
        # Módulo Financiero
        st.sidebar.markdown("---")
        st.sidebar.header("💰 Parámetros Económicos")
        precio_m3 = st.sidebar.number_input("Precio m³ (CLP):", value=12000, step=500)

        # Filtrado
        mask = (
            (df['Fecha Ingreso'].dt.date >= start_date) &
            (df['Fecha Ingreso'].dt.date <= end_date) &
            (df['Empresa'].isin(sel_companies))
        )
        df_filtered = df[mask].copy()

        # Cálculos
        df_filtered['Dif_Volumen'] = df_filtered['Vol. Declarado (m³)'] - df_filtered['Vol. IA (m³)']
        df_filtered['Ahorro_CLP'] = df_filtered['Dif_Volumen'].apply(lambda x: x * precio_m3 if x > 0 else 0)
        
        # KPIs
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("🚛 Camiones", len(df_filtered))
        k2.metric("📦 Volumen (m³)", f"{df_filtered['Vol. IA (m³)'].sum():,.0f}")
        k3.metric("🎯 Precisión", f"{df_filtered['Precisión (%)'].mean():.1f}%")
        k4.metric("⚠️ Rechazos", f"{(len(df_filtered[df_filtered['Contaminación (%)'] > 2.0])/len(df_filtered)*100):.1f}%")
        k5.metric("💰 Ahorro Est.", f"${df_filtered['Ahorro_CLP'].sum():,.0f}")

        # Gráficos con LÍNEAS DE TENDENCIA
        c1, c2 = st.columns(2)
        with c1:
            # Scatter con Tendencia (OLS)
            fig_scatter = px.scatter(df_filtered, x="Vol. Declarado (m³)", y="Precisión (%)", color="Material (IA Class)",
                                     trendline="ols", title="Correlación: Carga vs Precisión (con Tendencia)")
            st.plotly_chart(fig_scatter, use_container_width=True)
        with c2:
             # Histograma
            fig_hist = px.histogram(df_filtered, x="Vol. IA (m³)", color="Material (IA Class)", title="Distribución de Volúmenes")
            st.plotly_chart(fig_hist, use_container_width=True)

    # ==============================================================================
    # TAB 2: CLUSTERING K-MEANS (¡NUEVO!)
    # ==============================================================================
    with tab2:
        st.subheader("🤖 Segmentación Inteligente de Camiones (K-Means)")
        st.markdown("""
        Este modelo de IA agrupa los camiones automáticamente basándose en 3 variables clave:
        **Precisión**, **Contaminación** y **Volumen IA**. Permite detectar patrones de comportamiento anómalos.
        """)
        
        col_k1, col_k2 = st.columns([1, 3])
        
        with col_k1:
            n_clusters = st.slider("Número de Grupos (Clusters):", 2, 5, 3)
            btn_run = st.button("Ejecutar Modelo K-Means")
            
        if btn_run:
            # Preparar datos para el modelo
            features = df_filtered[['Vol. IA (m³)', 'Precisión (%)', 'Contaminación (%)']].dropna()
            
            # Entrenar K-Means
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            df_filtered['Cluster'] = kmeans.fit_predict(features)
            df_filtered['Cluster'] = df_filtered['Cluster'].astype(str) # Para que Plotly lo vea como categoría
            
            with col_k2:
                # Visualización 3D o 2D
                fig_cluster = px.scatter_3d(df_filtered, x='Vol. IA (m³)', y='Precisión (%)', z='Contaminación (%)',
                                            color='Cluster', title=f"Mapa 3D de Comportamiento ({n_clusters} Grupos)",
                                            hover_data=['Empresa', 'Patente'])
                st.plotly_chart(fig_cluster, use_container_width=True)
            
            st.success("✅ Modelo ejecutado exitosamente. Gire el gráfico 3D para explorar los grupos.")
            st.dataframe(df_filtered.groupby('Cluster')[['Vol. IA (m³)', 'Precisión (%)', 'Contaminación (%)']].mean())

    # ==============================================================================
    # TAB 3: PREDICCIONES (FORECASTING) (¡NUEVO!)
    # ==============================================================================
    with tab3:
        st.subheader("🔮 Proyección de Flujo (Próximos 7 días)")
        st.markdown("Modelo de Regresión Lineal para estimar el volumen de carga futura basado en el histórico.")
        
        # Preparar datos temporales
        daily_vol = df.groupby('Fecha Ingreso')['Vol. IA (m³)'].sum().reset_index()
        daily_vol['Dia_Num'] = np.arange(len(daily_vol)) # Convertir fecha a número para regresión
        
        # Entrenar Regresión Lineal
        X = daily_vol[['Dia_Num']]
        y = daily_vol['Vol. IA (m³)']
        model = LinearRegression()
        model.fit(X, y)
        
        # Predecir futuro (7 días)
        future_days = 7
        last_day_num = daily_vol['Dia_Num'].max()
        future_X = np.arange(last_day_num + 1, last_day_num + 1 + future_days).reshape(-1, 1)
        future_pred = model.predict(future_X)
        
        # Crear DataFrame futuro para graficar
        last_date = daily_vol['Fecha Ingreso'].max()
        future_dates = [last_date + timedelta(days=i) for i in range(1, future_days + 1)]
        df_future = pd.DataFrame({'Fecha Ingreso': future_dates, 'Vol. IA (m³)': future_pred, 'Tipo': 'Predicción'})
        daily_vol['Tipo'] = 'Histórico'
        
        # Unir y graficar
        df_forecast = pd.concat([daily_vol, df_future])
        
        fig_forecast = px.line(df_forecast, x='Fecha Ingreso', y='Vol. IA (m³)', color='Tipo', 
                               markers=True, title="Pronóstico de Volumen de Carga")
        fig_forecast.add_vline(x=last_date, line_dash="dash", line_color="green", annotation_text="Hoy")
        st.plotly_chart(fig_forecast, use_container_width=True)
        
        st.info(f"Tendencia calculada: El volumen varía aproximadamente {model.coef_[0]:.2f} m³ por día.")

    # ==============================================================================
    # TAB 4: CALIDAD DE DATOS (ESTADÍSTICAS)
    # ==============================================================================
    with tab4:
        st.subheader("📋 Resumen Estadístico (Data Quality)")
        st.markdown("Análisis descriptivo de las variables numéricas del dataset.")
        
        # Describe
        st.dataframe(df_filtered.describe().T, use_container_width=True)
        
        # Validación de nulos
        nulls = df.isnull().sum()
        if nulls.sum() == 0:
            st.success("✅ Dataset Limpio: No se detectaron valores nulos (NaN).")
        else:
            st.warning("⚠️ Se detectaron valores nulos:")
            st.write(nulls)