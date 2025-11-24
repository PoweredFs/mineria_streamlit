import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, confusion_matrix, classification_report, 
                            roc_auc_score)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import warnings
warnings.filterwarnings('ignore')

# ============ CONFIGURACIÓN DE PÁGINA ============
st.set_page_config(
    page_title="Predicción de Tipo de Inscripción",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============ SIDEBAR - NAVEGACIÓN ============
st.sidebar.title("🎓 ML - Tipo de Inscripción")
st.sidebar.write("Predicción del tipo de inscripción de estudiantes")

page = st.sidebar.radio(
    "Selecciona una sección:",
    ["🏠 Inicio", "📊 Datos", "🤖 Modelos", "🔮 Predicción", "📈 Métricas"]
)

# ============ CARGAR Y CACHEAR DATOS ============
@st.cache_data
def cargar_datos():
    """Cargar datos del Excel"""
    try:
        df = pd.read_excel("data.xlsx")
        return df
    except FileNotFoundError:
        st.error("❌ Archivo 'data.xlsx' no encontrado")
        return None

@st.cache_resource
def preparar_modelos(df):
    """Entrenar modelos y retornar objetos"""
    
    df_copy = df.copy()
    
    # Label Encoders
    le_tipo = LabelEncoder()
    le_programa_consec = LabelEncoder()
    le_programa = LabelEncoder()
    le_departamento = LabelEncoder()
    
    # Codificar variables
    df_copy["TIPO DE INSCRIPCION"] = le_tipo.fit_transform(df_copy["TIPO DE INSCRIPCION"])
    df_copy["PROGRAMA CONSECUTIVO"] = le_programa_consec.fit_transform(df_copy["PROGRAMA CONSECUTIVO"].astype(str))
    df_copy["PROGRAMA"] = le_programa.fit_transform(df_copy["PROGRAMA"].astype(str))
    df_copy["DEPARTAMENTO"] = le_departamento.fit_transform(df_copy["DEPARTAMENTO"].astype(str))
    
    # Preparar datos
    X = df_copy[["PROGRAMA CONSECUTIVO", "PROGRAMA", "DEPARTAMENTO"]]
    y = df_copy["TIPO DE INSCRIPCION"]
    
    # Division train-test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Random Forest
    model_rf = RandomForestClassifier(n_estimators=200, random_state=42)
    model_rf.fit(X_train, y_train)
    
    # Gradient Boosting
    model_gb = GradientBoostingClassifier(random_state=42)
    model_gb.fit(X_train, y_train)
    
    return {
        'rf': model_rf,
        'gb': model_gb,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'le_tipo': le_tipo,
        'le_programa_consec': le_programa_consec,
        'le_programa': le_programa,
        'le_departamento': le_departamento,
        'df': df,
        'prog_consec_encoded': dict(zip(le_programa_consec.classes_, le_programa_consec.transform(le_programa_consec.classes_))),
        'programa_encoded': dict(zip(le_programa.classes_, le_programa.transform(le_programa.classes_))),
        'departamento_encoded': dict(zip(le_departamento.classes_, le_departamento.transform(le_departamento.classes_)))
    }

# ============ FUNCIÓN PARA CALCULAR MÉTRICAS ============
def calcular_metricas(y_true, y_pred):
    """Calcular todas las métricas"""
    return {
        'accuracy': accuracy_score(y_true, y_pred) * 100,
        'precision': precision_score(y_true, y_pred, average='macro', zero_division=0) * 100,
        'recall': recall_score(y_true, y_pred, average='macro', zero_division=0) * 100,
        'f1': f1_score(y_true, y_pred, average='macro', zero_division=0) * 100,
    }

# ============ CARGAR DATOS ============
df = cargar_datos()

if df is not None:
    modelos = preparar_modelos(df)
    
    # ============ PÁGINA: INICIO ============
    if page == "🏠 Inicio":
        st.title("🎓 Predicción de Tipo de Inscripción")
        
        st.markdown("""
        ### Bienvenido a la Aplicación de Machine Learning
        
        Esta aplicación utiliza **dos algoritmos de clasificación** para predecir el tipo de inscripción
        de estudiantes basándose en su programa, consecutivo y departamento.
        
        #### 🤖 Algoritmos Implementados:
        
        **1. Random Forest Classifier**
        - Ensemble de múltiples árboles de decisión (200 árboles)
        - Captura relaciones complejas en los datos
        - Muy robusto y generalizable
        
        **2. Gradient Boosting Classifier**
        - Construcción secuencial de árboles
        - Corrección de errores de forma progresiva
        - Excelente rendimiento predictivo
        
        #### 📊 Funcionalidades:
        
        - 📈 Visualización de datos
        - 🤖 Entrenamiento de modelos
        - 🔮 Predicciones interactivas
        - 📉 Métricas de desempeño
        - 🎯 Matrices de confusión
        """)
        
        # Estadísticas generales
        st.subheader("📊 Estadísticas Generales")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total de Registros", len(df))
        
        with col2:
            st.metric("Características", len(["PROGRAMA CONSECUTIVO", "PROGRAMA", "DEPARTAMENTO"]))
        
        with col3:
            st.metric("Clases Predichas", len(modelos['le_tipo'].classes_))
        
        with col4:
            st.metric("Datos de Prueba", len(modelos['y_test']))
        
        # Distribución de tipo de inscripción
        st.subheader("Distribución de Tipo de Inscripción")
        
        tipo_dist = df["TIPO DE INSCRIPCION"].value_counts()
        
        fig, ax = plt.subplots(figsize=(10, 5))
        tipo_dist.plot(kind='bar', ax=ax, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        ax.set_title('Distribución de Tipo de Inscripción', fontsize=14, fontweight='bold')
        ax.set_xlabel('Tipo de Inscripción')
        ax.set_ylabel('Cantidad de Estudiantes')
        plt.xticks(rotation=45)
        st.pyplot(fig)
    
    # ============ PÁGINA: DATOS ============
    elif page == "📊 Datos":
        st.title("📊 Exploración de Datos")
        
        st.subheader("Vista previa del dataset")
        st.dataframe(df.head(10), use_container_width=True)
        
        st.subheader("Información del Dataset")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**Total de registros:** {len(df)}")
            st.write(f"**Columnas totales:** {df.shape[1]}")
        
        with col2:
            st.write(f"**Variable objetivo:** TIPO DE INSCRIPCION")
            st.write(f"**Características:** PROGRAMA CONSECUTIVO, PROGRAMA, DEPARTAMENTO")
        
        st.subheader("Valores Únicos")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write(f"**PROGRAMA CONSECUTIVO:** {df['PROGRAMA CONSECUTIVO'].nunique()} valores únicos")
        
        with col2:
            st.write(f"**PROGRAMA:** {df['PROGRAMA'].nunique()} valores únicos")
        
        with col3:
            st.write(f"**DEPARTAMENTO:** {df['DEPARTAMENTO'].nunique()} valores únicos")
        
        st.subheader("Clases Objetivo")
        st.write(f"**Tipos de Inscripción:** {', '.join(modelos['le_tipo'].classes_)}")
    
    # ============ PÁGINA: MODELOS ============
    elif page == "🤖 Modelos":
        st.title("🤖 Información de Modelos")
        
        # Realizar predicciones
        y_pred_rf = modelos['rf'].predict(modelos['X_test'])
        y_pred_gb = modelos['gb'].predict(modelos['X_test'])
        
        # Calcular métricas
        metricas_rf = calcular_metricas(modelos['y_test'], y_pred_rf)
        metricas_gb = calcular_metricas(modelos['y_test'], y_pred_gb)
        
        # Tabs para cada modelo
        tab1, tab2 = st.tabs(["🔴 Random Forest", "🔵 Gradient Boosting"])
        
        with tab1:
            st.subheader("Random Forest Classifier")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Accuracy", f"{metricas_rf['accuracy']:.2f}%")
            with col2:
                st.metric("Precision", f"{metricas_rf['precision']:.2f}%")
            with col3:
                st.metric("Recall", f"{metricas_rf['recall']:.2f}%")
            with col4:
                st.metric("F1-Score", f"{metricas_rf['f1']:.2f}%")
            
            # Matriz de confusión RF
            st.subheader("Matriz de Confusión")
            cm_rf = confusion_matrix(modelos['y_test'], y_pred_rf)
            
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', ax=ax, 
                       xticklabels=modelos['le_tipo'].classes_,
                       yticklabels=modelos['le_tipo'].classes_,
                       cbar_kws={'label': 'Cantidad'})
            ax.set_title('Matriz de Confusión - Random Forest', fontweight='bold', fontsize=14)
            ax.set_ylabel('Verdadero')
            ax.set_xlabel('Predicho')
            st.pyplot(fig)
            
            # Reporte
            st.subheader("Reporte Detallado")
            report_rf = classification_report(modelos['y_test'], y_pred_rf, 
                                             target_names=modelos['le_tipo'].classes_,
                                             output_dict=True)
            report_df_rf = pd.DataFrame(report_rf).transpose()
            st.dataframe(report_df_rf, use_container_width=True)
        
        with tab2:
            st.subheader("Gradient Boosting Classifier")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Accuracy", f"{metricas_gb['accuracy']:.2f}%")
            with col2:
                st.metric("Precision", f"{metricas_gb['precision']:.2f}%")
            with col3:
                st.metric("Recall", f"{metricas_gb['recall']:.2f}%")
            with col4:
                st.metric("F1-Score", f"{metricas_gb['f1']:.2f}%")
            
            # Matriz de confusión GB
            st.subheader("Matriz de Confusión")
            cm_gb = confusion_matrix(modelos['y_test'], y_pred_gb)
            
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(cm_gb, annot=True, fmt='d', cmap='Greens', ax=ax,
                       xticklabels=modelos['le_tipo'].classes_,
                       yticklabels=modelos['le_tipo'].classes_,
                       cbar_kws={'label': 'Cantidad'})
            ax.set_title('Matriz de Confusión - Gradient Boosting', fontweight='bold', fontsize=14)
            ax.set_ylabel('Verdadero')
            ax.set_xlabel('Predicho')
            st.pyplot(fig)
            
            # Reporte
            st.subheader("Reporte Detallado")
            report_gb = classification_report(modelos['y_test'], y_pred_gb, 
                                             target_names=modelos['le_tipo'].classes_,
                                             output_dict=True)
            report_df_gb = pd.DataFrame(report_gb).transpose()
            st.dataframe(report_df_gb, use_container_width=True)
    
    # ============ PÁGINA: PREDICCIÓN ============
    elif page == "🔮 Predicción":
        st.title("🔮 Realizar Predicción")
        
        st.write("Ingresa los datos para predecir el tipo de inscripción")
        
        # Crear formulario
        with st.form("prediction_form"):
            st.subheader("Selecciona los valores")
            
            # Programa Consecutivo
            prog_consec_options = sorted(modelos['df']['PROGRAMA CONSECUTIVO'].unique())
            prog_consec_input = st.selectbox(
                "PROGRAMA CONSECUTIVO",
                prog_consec_options
            )
            
            # Programa
            programa_options = sorted(modelos['df']['PROGRAMA'].unique())
            programa_input = st.selectbox(
                "PROGRAMA",
                programa_options
            )
            
            # Departamento
            departamento_options = sorted(modelos['df']['DEPARTAMENTO'].unique())
            departamento_input = st.selectbox(
                "DEPARTAMENTO",
                departamento_options
            )
            
            submit = st.form_submit_button("🔮 Realizar Predicción")
        
        if submit:
            try:
                # Convertir valores a sus códigos
                prog_consec_encoded = modelos['prog_consec_encoded'].get(prog_consec_input, 0)
                programa_encoded = modelos['programa_encoded'].get(programa_input, 0)
                departamento_encoded = modelos['departamento_encoded'].get(departamento_input, 0)
                
                # Crear fila para predecir
                row = [[prog_consec_encoded, programa_encoded, departamento_encoded]]
                
                # Predicciones
                pred_rf = modelos['rf'].predict(row)[0]
                pred_gb = modelos['gb'].predict(row)[0]
                
                # Probabilidades
                prob_rf = modelos['rf'].predict_proba(row)[0]
                prob_gb = modelos['gb'].predict_proba(row)[0]
                
                # Decodificar predicciones
                pred_label_rf = modelos['le_tipo'].inverse_transform([pred_rf])[0]
                pred_label_gb = modelos['le_tipo'].inverse_transform([pred_gb])[0]
                
                # Mostrar resultados
                st.subheader("📋 Resultados de Predicción")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### 🔴 Random Forest")
                    st.markdown(f"**Predicción:** {pred_label_rf}")
                    st.markdown(f"**Confianza:** {max(prob_rf)*100:.2f}%")
                    
                    st.write("**Probabilidades:**")
                    for i, clase in enumerate(modelos['le_tipo'].classes_):
                        st.write(f"- {clase}: {prob_rf[i]*100:.2f}%")
                
                with col2:
                    st.markdown("### 🔵 Gradient Boosting")
                    st.markdown(f"**Predicción:** {pred_label_gb}")
                    st.markdown(f"**Confianza:** {max(prob_gb)*100:.2f}%")
                    
                    st.write("**Probabilidades:**")
                    for i, clase in enumerate(modelos['le_tipo'].classes_):
                        st.write(f"- {clase}: {prob_gb[i]*100:.2f}%")
                
                # Consenso
                if pred_label_rf == pred_label_gb:
                    st.success(f"✅ CONSENSO: Ambos modelos predicen **{pred_label_rf}**")
                else:
                    st.warning(f"⚠️ DESACUERDO: RF predice {pred_label_rf}, GB predice {pred_label_gb}")
            
            except Exception as e:
                st.error(f"Error en la predicción: {str(e)}")
    
    # ============ PÁGINA: MÉTRICAS ============
    elif page == "📈 Métricas":
        st.title("📈 Validación de Métricas")
        
        # Realizar predicciones
        y_pred_rf = modelos['rf'].predict(modelos['X_test'])
        y_pred_gb = modelos['gb'].predict(modelos['X_test'])
        
        # Calcular métricas
        metricas_rf = calcular_metricas(modelos['y_test'], y_pred_rf)
        metricas_gb = calcular_metricas(modelos['y_test'], y_pred_gb)
        
        # Tabla comparativa
        st.subheader("Comparación de Métricas")
        
        metricas_df = pd.DataFrame({
            'Métrica': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
            'Random Forest': [f"{metricas_rf['accuracy']:.2f}%", 
                             f"{metricas_rf['precision']:.2f}%",
                             f"{metricas_rf['recall']:.2f}%",
                             f"{metricas_rf['f1']:.2f}%"],
            'Gradient Boosting': [f"{metricas_gb['accuracy']:.2f}%",
                                 f"{metricas_gb['precision']:.2f}%",
                                 f"{metricas_gb['recall']:.2f}%",
                                 f"{metricas_gb['f1']:.2f}%"]
        })
        
        st.dataframe(metricas_df, use_container_width=True)
        
        # Gráfico comparativo
        st.subheader("Gráfico Comparativo")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = np.arange(len(metricas_df))
        width = 0.35
        
        valores_rf = [metricas_rf['accuracy'], metricas_rf['precision'], 
                     metricas_rf['recall'], metricas_rf['f1']]
        valores_gb = [metricas_gb['accuracy'], metricas_gb['precision'], 
                     metricas_gb['recall'], metricas_gb['f1']]
        
        ax.bar(x - width/2, valores_rf, width, label='Random Forest', color='#FF6B6B')
        ax.bar(x + width/2, valores_gb, width, label='Gradient Boosting', color='#4ECDC4')
        
        ax.set_ylabel('Porcentaje (%)', fontweight='bold')
        ax.set_title('Comparación de Métricas - Ambos Modelos', fontweight='bold', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(['Accuracy', 'Precision', 'Recall', 'F1-Score'])
        ax.legend()
        ax.axhline(y=85, color='r', linestyle='--', linewidth=2, label='Umbral 85%')
        ax.set_ylim([0, 110])
        
        st.pyplot(fig)
        
        # Validación de requisitos
        st.subheader("✅ Validación de Requisitos (>85%)")
        
        umbral = 85
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Random Forest")
            if metricas_rf['accuracy'] >= umbral:
                st.success(f"✅ Accuracy: {metricas_rf['accuracy']:.2f}% >= {umbral}%")
            else:
                st.warning(f"⚠️ Accuracy: {metricas_rf['accuracy']:.2f}% < {umbral}%")
            
            if metricas_rf['precision'] >= umbral:
                st.success(f"✅ Precision: {metricas_rf['precision']:.2f}% >= {umbral}%")
            else:
                st.warning(f"⚠️ Precision: {metricas_rf['precision']:.2f}% < {umbral}%")
        
        with col2:
            st.markdown("### Gradient Boosting")
            if metricas_gb['accuracy'] >= umbral:
                st.success(f"✅ Accuracy: {metricas_gb['accuracy']:.2f}% >= {umbral}%")
            else:
                st.warning(f"⚠️ Accuracy: {metricas_gb['accuracy']:.2f}% < {umbral}%")
            
            if metricas_gb['precision'] >= umbral:
                st.success(f"✅ Precision: {metricas_gb['precision']:.2f}% >= {umbral}%")
            else:
                st.warning(f"⚠️ Precision: {metricas_gb['precision']:.2f}% < {umbral}%")

else:
    st.error("No se pudieron cargar los datos. Verifica que el archivo 'data.xlsx' esté en el directorio correcto.")