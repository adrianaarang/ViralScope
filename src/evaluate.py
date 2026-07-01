"""
evaluate.py

Funciones reutilizables para evaluar modelos de regresión y detectar
overfitting comparando train vs test.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def calcular_metricas(y_real, y_pred) -> dict:
    """Devuelve MAE, RMSE y R² para un conjunto de predicciones."""
    mae = mean_absolute_error(y_real, y_pred)
    rmse = np.sqrt(mean_squared_error(y_real, y_pred))
    r2 = r2_score(y_real, y_pred)
    return {"MAE": mae, "RMSE": rmse, "R2": r2}


def comparar_train_test(modelo, X_train, y_train, X_test, y_test, nombre_modelo: str) -> pd.DataFrame:
    """
    Calcula métricas en train y test para un modelo ya entrenado y las
    devuelve en un DataFrame, útil para detectar overfitting (gran
    diferencia entre métricas de train y test).
    """
    pred_train = modelo.predict(X_train)
    pred_test = modelo.predict(X_test)

    metricas_train = calcular_metricas(y_train, pred_train)
    metricas_test = calcular_metricas(y_test, pred_test)

    resultado = pd.DataFrame({
        "modelo": [nombre_modelo, nombre_modelo],
        "conjunto": ["train", "test"],
        **{k: [metricas_train[k], metricas_test[k]] for k in metricas_train},
    })
    return resultado


def tabla_comparativa_modelos(resultados: list[pd.DataFrame]) -> pd.DataFrame:
    """Concatena varios resultados de comparar_train_test en una sola tabla."""
    return pd.concat(resultados, ignore_index=True)
