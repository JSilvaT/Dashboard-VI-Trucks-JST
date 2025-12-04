import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import datetime

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(
    page_title="VI Trucks JST - IA Analytics",
    page_icon="🧠",
    layout="wide"
)

# --- ESTILOS CSS PERSONALIZADOS ---
st.markdown("""
    <style>
    .metric-card { background-color: #f9f9f9; border-left: 5px solid #4CAF50; padding: 15px; border-radius: 5px; box-shadow: 2px 2px 5px rgba(0,0,0,0.1); }
    </style>
    """, unsafe_allow_html=True)

# --- ENCABEZADO ---
st.title("🧠 VI Trucks JST: Plataforma de Inteligencia Artificial")
st.markdown("**Cliente:** CPG Chile | **Módulo:** Analytics Avanzado (Clustering & Predicción) | **Versión:** 3.1 (Proyección Mensual)")

# --- 1. MÓDULO DE INGESTA Y ETL (EXTRACT, TRANSFORM, LOAD) ---
@st.cache_data
def cargar_y_limpiar_datos():
    try:
        # EXTRACT: Carga de datos crudos
        df = pd.read_csv('simulacion_piloto_60dias_CPG.csv')
        
        # TRANSFORM: Limpieza y Casteo de Tipos
        # Convertir fechas a objetos datetime reales
        df['Fecha Ingreso'] = pd.to_datetime(df['Fecha Ingreso'])
        # Imputación de nulos (si existieran) con 0 o la media
        df.fillna(0, inplace=True)
        
        return df
    except FileNotFoundError:
        return None

df = cargar_y_limpiar_datos()

if df is None:
    st.error("⚠️ ERROR CRÍTICO: No se encontró el archivo 'simulacion_piloto_60dias_CPG.csv'. Asegúrese de que esté en la misma carpeta.")
else:
    # --- BARRA LATERAL (FILTROS GLOBALES) ---
    st.sidebar.header("🔍 Filtros de Operación")
    
    # Filtro de Fechas Dinámico
    min_date, max_date = df['Fecha Ingreso'].min(), df['Fecha Ingreso'].max()
    fechas = st.sidebar.date_input("Rango de Análisis", (min_date, max_date), min_value=min_date, max_value=max_date)
    
    # Filtro de Materiales
    todos_materiales = df['Material (IA Class)'].unique()
    materiales_sel = st.sidebar.multiselect("Tipo de Material", todos_materiales, default=todos_materiales)
    
    # Filtro de Empresas
    todas_empresas = df['Empresa'].unique()
    empresas_sel = st.sidebar.multiselect("Empresa Transportista", todas_empresas, default=todas_empresas)

    # APLICACIÓN DE FILTROS (MÁSCARA)
    if isinstance(fechas, tuple) and len(fechas) == 2:
        mask = (
            (df['Fecha Ingreso'].dt.date >= fechas[0]) & 
            (df['Fecha Ingreso'].dt.date <= fechas[1]) & 
            (df['Material (IA Class)'].isin(materiales_sel)) &
            (df['Empresa'].isin(empresas_sel))
        )
        df_filtered = df[mask].copy()
    else:
        df_filtered = df.copy()

    # --- PESTAÑAS DE NAVEGACIÓN ---
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard Operativo", "🧹 Calidad de Datos (ETL)", "🤖 Clustering (IA)", "🔮 Predicciones (30 Días)"])

    # ==========================================
    # TAB 1: DASHBOARD OPERATIVO
    # ==========================================
    with tab1:
        st.subheader("Estado Actual de la Operación (KPIs)")
        
        if df_filtered.empty:
            st.warning("No hay datos para los filtros seleccionados.")
        else:
            col1, col2, col3, col4 = st.columns(4)
            
            # Cálculo de métricas
            prec_media = df_filtered['Precisión (%)'].mean()
            vol_total = df_filtered['Vol. IA (m³)'].sum()
            camiones = len(df_filtered)
            fallos = len(df_filtered[df_filtered['Precisión (%)'] < 90])
            tasa_fallos = (fallos / camiones) * 100 if camiones > 0 else 0

            # Despliegue de métricas
            col1.metric("Precisión Global", f"{prec_media:.2f}%", delta="Meta > 90%")
            col2.metric("Volumen Procesado", f"{vol_total:,.0f} m³")
            col3.metric("Flujo de Camiones", f"{camiones}", delta="Unidades")
            col4.metric("Tasa de Error", f"{tasa_fallos:.1f}%", delta_color="inverse")
            
            st.divider()

            # Gráficos Operativos
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("##### Evolución Diaria de Precisión")
                daily_acc = df_filtered.groupby('Fecha Ingreso')['Precisión (%)'].mean().reset_index()
                fig_evo = plt.figure(figsize=(10, 4))
                sns.lineplot(data=daily_acc, x='Fecha Ingreso', y='Precisión (%)', marker='o', color='green')
                plt.axhline(90, color='red', linestyle='--', label='Meta 90%')
                plt.grid(True, alpha=0.3)
                st.pyplot(fig_evo)
            
            with c2:
                st.markdown("##### Ranking de Precisión por Empresa")
                ranking = df_filtered.groupby('Empresa')['Precisión (%)'].mean().sort_values().reset_index()
                fig_rank = plt.figure(figsize=(10, 4))
                sns.barplot(data=ranking, x='Precisión (%)', y='Empresa', palette='viridis')
                plt.axvline(90, color='red', linestyle='--')
                st.pyplot(fig_rank)

    # ==========================================
    # TAB 2: CALIDAD DE DATOS (ETL AUDIT)
    # ==========================================
    with tab2:
        st.subheader("🧹 Auditoría de Calidad de Datos (Data Health)")
        st.info("Validación técnica de integridad de los datos antes del procesamiento IA.")
        
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("#### 1. Integridad de Datos (Nulos)")
            nulos = df_filtered.isnull().sum()
            if nulos.sum() == 0:
                st.success("✅ INTEGRIDAD OK: No se detectaron valores nulos.")
                st.dataframe(nulos, width=400)
            else:
                st.error(f"⚠️ ALERTA: Se detectaron {nulos.sum()} valores perdidos.")
                st.dataframe(nulos[nulos > 0])
        
        with col_b:
            st.markdown("#### 2. Estadística Descriptiva")
            st.dataframe(df_filtered[['Vol. Declarado (m³)', 'Vol. IA (m³)', 'Precisión (%)']].describe())

    # ==========================================
    # TAB 3: CLUSTERING (APRENDIZAJE NO SUPERVISADO)
    # ==========================================
    with tab3:
        st.subheader("🤖 Segmentación Inteligente de Camiones")
        st.markdown("Algoritmo **K-Means** para agrupar camiones por comportamiento (Precisión vs Contaminación).")

        if len(df_filtered) > 10:
            # Scaling
            X = df_filtered[['Precisión (%)', 'Contaminación (%)']].copy()
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Selector de K
            k = st.slider("Número de Grupos (K)", 2, 5, 3)
            
            # Modelo K-Means
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            df_filtered['Cluster'] = kmeans.fit_predict(X_scaled)
            
            # Visualización
            col_k1, col_k2 = st.columns([3, 1])
            with col_k1:
                fig_k, ax_k = plt.subplots(figsize=(10, 6))
                sns.scatterplot(data=df_filtered, x='Precisión (%)', y='Contaminación (%)', 
                                hue='Cluster', palette='deep', s=100, style='Material (IA Class)', ax=ax_k)
                plt.title(f"Segmentación de la Flota en {k} Perfiles")
                st.pyplot(fig_k)
            
            with col_k2:
                st.markdown("#### Perfiles Identificados")
                for i in range(k):
                    cluster_data = df_filtered[df_filtered['Cluster'] == i]
                    p_mean = cluster_data['Precisión (%)'].mean()
                    c_mean = cluster_data['Contaminación (%)'].mean()
                    st.success(f"**Grupo {i}**\n- Precisión: {p_mean:.1f}%\n- Contam.: {c_mean:.1f}%")
        else:
            st.warning("Datos insuficientes para Clustering (>10 registros requeridos).")

    # ==========================================
    # TAB 4: PREDICCIONES (REGRESIÓN LINEAL - 30 DÍAS)
    # ==========================================
    with tab4:
        st.subheader("🔮 Proyección Mensual de Demanda (30 Días)")
        st.markdown("Modelo de **Regresión Lineal** para estimar el volumen de carga del próximo mes calendario.")

        # Agrupar datos por día
        daily_vol = df.groupby('Fecha Ingreso')['Vol. IA (m³)'].sum().reset_index()
        
        if len(daily_vol) > 5:
            # Ingeniería de Características
            daily_vol['Dia_Num'] = (daily_vol['Fecha Ingreso'] - daily_vol['Fecha Ingreso'].min()).dt.days
            
            # Entrenamiento
            X_reg = daily_vol[['Dia_Num']]
            y_reg = daily_vol['Vol. IA (m³)']
            model = LinearRegression()
            model.fit(X_reg, y_reg)
            
            # --- PREDICCIÓN EXTENDIDA (30 DÍAS) ---
            last_day_num = daily_vol['Dia_Num'].max()
            # Creamos un rango de 1 a 30 días en el futuro
            future_days_num = np.array(range(last_day_num + 1, last_day_num + 31)).reshape(-1, 1)
            future_vol = model.predict(future_days_num)
            
            # Fechas futuras (30 días)
            last_date = daily_vol['Fecha Ingreso'].max()
            future_dates = [last_date + datetime.timedelta(days=i) for i in range(1, 31)]
            
            # DataFrame Futuro
            df_future = pd.DataFrame({
                'Fecha Ingreso': future_dates,
                'Vol. IA (m³)': future_vol,
                'Tipo': 'Proyección (30 Días)'
            })
            daily_vol['Tipo'] = 'Histórico Real'
            
            # Unir datasets
            df_combined = pd.concat([daily_vol, df_future])

            # Métricas de la Proyección
            vol_proyectado_mes = df_future['Vol. IA (m³)'].sum()
            tendencia_txt = "Creciente" if model.coef_[0] > 0 else "Decreciente"

            col_p1, col_p2 = st.columns(2)
            col_p1.metric("Volumen Total Estimado (30 días)", f"{vol_proyectado_mes:,.0f} m³")
            col_p2.metric("Tendencia de Carga", tendencia_txt, f"{model.coef_[0]:.2f} m³/día")

            # Visualización
            fig_pred, ax_p = plt.subplots(figsize=(12, 5))
            
            # Graficar Histórico vs Predicción
            sns.lineplot(









