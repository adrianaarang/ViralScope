"""
preprocessing.py

Funciones reutilizables para limpieza, feature engineering y construcción
del pipeline de preprocesamiento (sin fuga de datos).
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer


def cargar_datos(path: str) -> pd.DataFrame:
    """Carga el CSV crudo y devuelve un DataFrame."""
    df = pd.read_csv(path)
    return df


def resumen_inicial(df: pd.DataFrame) -> None:
    """Imprime un resumen rápido: shape, tipos, nulos, duplicados."""
    print(f"Shape: {df.shape}")
    print("\nTipos de datos:")
    print(df.dtypes)
    print("\nValores nulos por columna:")
    print(df.isnull().sum())
    print(f"\nFilas duplicadas: {df.duplicated().sum()}")


def extraer_features_temporales(df: pd.DataFrame, col_fecha: str) -> pd.DataFrame:
    """
    Extrae hora del día y día de la semana a partir de una columna de
    fecha/hora de publicación. No modifica el DataFrame original.
    """
    df = df.copy()
    df[col_fecha] = pd.to_datetime(df[col_fecha], errors="coerce")
    df["hora_publicacion"] = df[col_fecha].dt.hour
    df["dia_semana"] = df[col_fecha].dt.day_name()
    return df


def calcular_longitud_caption(df: pd.DataFrame, col_caption: str) -> pd.DataFrame:
    """Añade columna con la longitud (en caracteres) del caption."""
    df = df.copy()
    df["longitud_caption"] = df[col_caption].fillna("").astype(str).str.len()
    return df


def calcular_num_hashtags(df: pd.DataFrame, col_hashtags: str) -> pd.DataFrame:
    """
    Añade columna con el número de hashtags, asumiendo que vienen como
    texto separado por espacios o comas en col_hashtags.
    """
    df = df.copy()

    def contar(valor):
        if pd.isna(valor):
            return 0
        texto = str(valor)
        separador = "," if "," in texto else " "
        return len([h for h in texto.split(separador) if h.strip()])

    df["num_hashtags"] = df[col_hashtags].apply(contar)
    return df


def construir_pipeline_preprocesamiento(
    features_numericas: list[str], features_categoricas: list[str]
) -> ColumnTransformer:
    """
    Construye un ColumnTransformer que:
    - Imputa y escala las variables numéricas.
    - Imputa y aplica One-Hot Encoding a las variables categóricas.

    Se ajusta SIEMPRE solo sobre el conjunto de entrenamiento (fit en train,
    transform en train y test) para evitar data leakage.
    """
    transformador_numerico = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    transformador_categorico = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocesador = ColumnTransformer(transformers=[
        ("num", transformador_numerico, features_numericas),
        ("cat", transformador_categorico, features_categoricas),
    ])

    return preprocesador
