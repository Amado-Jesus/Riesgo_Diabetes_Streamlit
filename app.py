import streamlit as st
import pandas as pd
import joblib

# ------------------------------
# Configuración de la app
# ------------------------------
st.set_page_config(
    page_title="Detección de Riesgo de Diabetes",
    layout="centered"
)

st.title("🩺 Detección de Riesgo de Diabetes")
st.write(
    "Aplicación clínica para la predicción automática del riesgo de diabetes "
    "a partir de variables metabólicas."
)

# ------------------------------
# Cargar modelo
# ------------------------------
@st.cache_resource
def cargar_modelo():
    return joblib.load("clf.pkl")

modelo = cargar_modelo()

# ------------------------------
# Subida de archivo
# ------------------------------
archivo = st.file_uploader(
    "📂 Cargue el archivo CSV del paciente",
    type=["csv"]
)

if archivo is not None:
    df = pd.read_csv(archivo)

    st.subheader("📊 Vista previa de los datos")
    st.dataframe(df.head())

    # ------------------------------
    # Verificación de columnas
    # ------------------------------
    columnas_esperadas = ['Genero', 'Edad', 
                          'Urea', 'Cr', 
                          'HbA1c', 'Chol', 
                          'TG', 'HDL', 'LDL',
                            'VLDL', 'IMC']

    if not all(col in df.columns for col in columnas_esperadas):
        st.error(
            "❌ El archivo no contiene todas las columnas requeridas.\n\n"
            f"Columnas esperadas:\n{columnas_esperadas}"
        )
    else:
        # ------------------------------
        # Predicción
        # ------------------------------
        st.subheader("🔮 Predicción del modelo")

        predicciones = modelo.predict(df[columnas_esperadas])

        df_resultado = df.copy()
        df_resultado["riesgo_diabetes"] = predicciones
        df_resultado["riesgo_diabetes"] = df_resultado["riesgo_diabetes"].map({
            0: "Riesgo bajo",
            1: "Riesgo moderado",
            2:"Riesgo alto"
        })

        st.success("✅ Predicciones generadas correctamente")

        st.dataframe(df_resultado.head())

        # ------------------------------
        # Descargar resultados
        # ------------------------------
        csv_resultado = df_resultado.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="⬇️ Descargar archivo etiquetado",
            data=csv_resultado,
            file_name="predicciones_riesgo_diabetes.csv",
            mime="text/csv"
        )

# ------------------------------
# Footer
# ------------------------------
st.markdown("---")
st.caption(
    "⚠️ Esta herramienta es de apoyo clínico y no sustituye el juicio médico."
)