import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta  # <--- AQUÍ ESTABA EL DETALLE

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
st.markdown("**Cliente:** CPG Chile | **Módulo:** Analytics Avanzado (Clustering & Predicción) | **Versión:** 3.0")

# --- 1. MÓDULO DE INGESTA Y ETL (EXTRACT, TRANSFORM, LOAD) ---
@st.cache_data
def cargar_y_limpiar_datos():
    try:
        # EXTRACT: Carga de datos crudos
        df = pd.read_csv('simulacion_piloto_60dias_CPG.csv')
        
        # TRANSFORM: Limpieza y Casteo de Tipos
        # Convertir fechas a objetos datetime reales
        df['Fecha Ingreso'] = pd.to_datetime(df['Fecha Ingreso'])
        # Imputación de nulos (si existieran) con 0 o la media, para evitar errores en ML
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
        df_filtered = df.copy() # Fallback por seguridad

    # --- PESTAÑAS DE NAVEGACIÓN (LA ESTRUCTURA DE LA APP) ---
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard Operativo", "🧹 Calidad de Datos (ETL)", "🤖 Clustering (IA)", "🔮 Predicciones Futuras"])

    # ==========================================
    # TAB 1: DASHBOARD OPERATIVO (Resumen Ejecutivo)
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
        st.info("Este módulo garantiza la transparencia del proceso. Validamos que los datos no tengan errores antes de aplicar IA.")
        
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("#### 1. Integridad de Datos (Nulos)")
            nulos = df_filtered.isnull().sum()
            if nulos.sum() == 0:
                st.success("✅ INTEGRIDAD OK: No se detectaron valores nulos en el dataset.")
                st.dataframe(nulos, width=400)
            else:
                st.error(f"⚠️ ALERTA: Se detectaron {nulos.sum()} valores perdidos.")
                st.dataframe(nulos[nulos > 0])
        
        with col_b:
            st.markdown("#### 2. Estadística Descriptiva Automática")
            st.markdown("Resumen estadístico de las variables numéricas clave:")
            st.dataframe(df_filtered[['Vol. Declarado (m³)', 'Vol. IA (m³)', 'Precisión (%)']].describe())

    # ==============================================================================
    # TAB 3: PREDICCIONES (FORECASTING) - VISUALMENTE MEJORADO
    # ==============================================================================
    with tab3:
        st.subheader("🔮 Proyección de Flujo (Próximos 7 días)")
        st.markdown("Proyección basada en tendencia lineal y media móvil de los últimos registros.")
        
        # 1. Preparar datos
        daily_vol = df.groupby('Fecha Ingreso')['Vol. IA (m³)'].sum().reset_index()
        daily_vol['Dia_Num'] = np.arange(len(daily_vol)) 
        
        # 2. Entrenar Modelo (Regresión Lineal Simple)
        X = daily_vol[['Dia_Num']]
        y = daily_vol['Vol. IA (m³)']
        
        if len(daily_vol) > 1:
            model = LinearRegression()
            model.fit(X, y)
            
            # 3. Predecir Futuro
            future_days = 7
            last_day_num = daily_vol['Dia_Num'].max()
            future_X = np.arange(last_day_num + 1, last_day_num + 1 + future_days).reshape(-1, 1)
            future_pred = model.predict(future_X)
            
            # Generar fechas futuras
            last_date = daily_vol['Fecha Ingreso'].max()
            future_dates = [last_date + timedelta(days=i) for i in range(1, future_days + 1)]
            
            # --- TRUCO VISUAL: CONECTAR LAS LÍNEAS ---
            # Agregamos el último punto real como el primer punto de la predicción
            # para que no quede un hueco en el gráfico.
            last_real_val = daily_vol.iloc[-1]['Vol. IA (m³)']
            
            # Fechas: [Última Real, Futuro 1, Futuro 2...]
            plot_dates = [last_date] + future_dates
            # Valores: [Último Real, Pred 1, Pred 2...]
            plot_vals = [last_real_val] + list(future_pred.flatten())
            
            df_future = pd.DataFrame({
                'Fecha Ingreso': plot_dates, 
                'Vol. IA (m³)': plot_vals, 
                'Tipo': 'Predicción'
            })
            
            daily_vol['Tipo'] = 'Histórico'
            
            # Unir para graficar
            df_forecast = pd.concat([daily_vol, df_future])
            
            # 4. Graficar
            fig_forecast = px.line(df_forecast, x='Fecha Ingreso', y='Vol. IA (m³)', color='Tipo', 
                                   markers=True, title="Pronóstico de Volumen de Carga",
                                   color_discrete_map={"Histórico": "#1f77b4", "Predicción": "#ff7f0e"}) # Azul y Naranja
            
            # Línea vertical de "Hoy"
            # Usamos el truco numérico que sí funciona
            fecha_numerica = last_date.timestamp() * 1000
            fig_forecast.add_vline(x=fecha_numerica, line_dash="dash", line_color="green", annotation_text="Hoy")
            
            st.plotly_chart(fig_forecast, use_container_width=True)
            
            # 5. Interpretación Inteligente
            tendencia = model.coef_[0]
            
            # Lógica para que el texto tenga sentido de negocio
            col_res1, col_res2 = st.columns(2)
            with col_res1:
                st.info(f"**Tasa de Variación:** {tendencia:.2f} m³/día")
            
            with col_res2:
                if abs(tendencia) < 0.5:
                    st.success("✅ **Diagnóstico:** Operación Estable. El flujo se mantiene constante sin desviaciones críticas.")
                elif tendencia > 0:
                    st.success("📈 **Diagnóstico:** Tendencia al Alza. Se proyecta un aumento en la carga de trabajo.")
                else:
                    st.warning("📉 **Diagnóstico:** Tendencia a la Baja. Posible disminución de actividad.")
                    
        else:
            st.warning("⚠️ No hay suficientes datos históricos para generar una predicción.")

    # ==========================================
    # TAB 4: PREDICCIONES (REGRESIÓN LINEAL)
    # ==========================================
    with tab4:
        st.subheader("🔮 Proyección de Volúmenes (Forecasting)")
        st.markdown("Modelo supervisado de **Regresión Lineal** para estimar la carga de trabajo de los próximos 7 días.")

        # Agrupar datos por día
        daily_vol = df.groupby('Fecha Ingreso')['Vol. IA (m³)'].sum().reset_index()
        
        if len(daily_vol) > 5:
            # Ingeniería de Características (Día numérico)
            daily_vol['Dia_Num'] = (daily_vol['Fecha Ingreso'] - daily_vol['Fecha Ingreso'].min()).dt.days
            
            # Entrenamiento del Modelo
            X_reg = daily_vol[['Dia_Num']]
            y_reg = daily_vol['Vol. IA (m³)']
            model = LinearRegression()
            model.fit(X_reg, y_reg)
            
            # Predicción (Próximos 7 días)
            last_day_num = daily_vol['Dia_Num'].max()
            future_days_num = np.array(range(last_day_num + 1, last_day_num + 8)).reshape(-1, 1)
            future_vol = model.predict(future_days_num)
            
            # Crear fechas futuras
            last_date = daily_vol['Fecha Ingreso'].max()
            future_dates = [last_date + datetime.timedelta(days=i) for i in range(1, 8)]
            
            # DataFrame de Futuro
            df_future = pd.DataFrame({
                'Fecha Ingreso': future_dates,
                'Vol. IA (m³)': future_vol,
                'Tipo': 'Proyección (IA)'
            })
            daily_vol['Tipo'] = 'Histórico Real'
            
            # Unir para graficar
            df_combined = pd.concat([daily_vol, df_future])

            # Visualización
            fig_pred, ax_p = plt.subplots(figsize=(12, 5))
            sns.lineplot(data=df_combined, x='Fecha Ingreso', y='Vol. IA (m³)', hue='Tipo', style='Tipo', markers=True, ax=ax_p)
            
            # Línea de Tendencia General
            x_trend = np.linspace(0, last_day_num + 7, 100).reshape(-1, 1)
            y_trend = model.predict(x_trend)
            trend_dates = [daily_vol['Fecha Ingreso'].min() + datetime.timedelta(days=int(d)) for d in x_trend.flatten()]
            ax_p.plot(trend_dates, y_trend, color='red', linestyle='--', alpha=0.5, label='Tendencia Lineal')
            
            plt.title(f"Proyección de Demanda para la Semana del {future_dates[0].strftime('%d-%m')}")
            plt.legend()
            st.pyplot(fig_pred)

            st.caption("Nota: La proyección asume condiciones operativas similares a los últimos 60 días.")
        else:
            st.warning("No hay suficientes días de datos para generar una predicción fiable.")

# --- PIE DE PÁGINA ---
st.divider()
st.caption("Sistema de Visión Artificial 'VI Trucks JST' | Desarrollado para CPG Chile | Proyecto IDA300 - UNAB")







