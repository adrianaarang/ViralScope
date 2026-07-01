# Instagram Reach Predictor

**Proyecto I — Módulo III: Clasificación y Regresión**
Cliente: agencia de marketing digital (vía **DataScope Solutions**)

## Contexto de negocio

Nuestro cliente gestiona contenido en Instagram para varias marcas y quiere
saber **qué variables predicen mejor el alcance (reach) de una publicación**,
para poder planificar tipo de contenido, horario y formato con criterio
basado en datos, en lugar de intuición.

## Pregunta de negocio

> ¿Podemos predecir el alcance (reach) de una publicación de Instagram a
> partir de sus características (tipo de post, hashtags, hora de
> publicación, longitud del caption, etc.) sin recurrir a métricas que solo
> se conocen *después* de publicar (likes, comentarios)?

Es un problema de **regresión** (el target, `reach`, es una variable numérica
continua).

## Dataset

[Instagram Analytics Dataset](https://www.kaggle.com/datasets/kundanbedmutha/instagram-analytics-dataset)
(Kaggle, 29.999 publicaciones, 23 columnas, sin nulos).

> ⚠️ El CSV no se versiona en el repo (ver `.gitignore`); colócalo en
> `data/raw/instagram_analytics.csv` antes de ejecutar el notebook.

## 🔎 Resultado principal (resumen ejecutivo)

Tras EDA, feature engineering y entrenar 3 modelos (baseline, Regresión
Lineal, Random Forest) con validación cruzada, **ninguno supera al
baseline**: R² ≈ 0 (ligeramente negativo en test) en todos los casos.

Las variables conocidas *antes de publicar* (tipo de cuenta, tipo de
contenido, categoría, hora, día, nº de seguidores, longitud del caption,
nº de hashtags) **no están correlacionadas de forma relevante con el
reach** en este dataset. Solo las métricas *posteriores* a la publicación
(impressions, likes, saves, shares, comments) explican el reach — y esas no
son usables como predictoras sin caer en data leakage.

**Conclusión para el cliente:** con los datos actuales no es posible
predecir el alcance antes de publicar; el hallazgo en sí es valioso porque
indica que optimizar "hora ideal" o "nº de hashtags" probablemente no es la
palanca real, y que hace falta enriquecer el dataset (historial de cuenta,
señales de contenido, audiencia) antes de construir un modelo en
producción. Detalle completo en la sección 7-8 del notebook.

## Estructura del repositorio

```
instagram-reach-predictor/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/            # CSV original de Kaggle (no versionado)
│   └── processed/      # Datos limpios/transformados (no versionado)
├── notebooks/
│   └── 01_eda_preprocesamiento_modelado.ipynb
└── src/
    ├── preprocessing.py  # funciones reutilizables de limpieza/encoding
    └── evaluate.py        # funciones de evaluación y reporting de métricas
```

## Cómo empezar

1. Crear entorno e instalar dependencias:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```
2. El dataset (`Instagram_Analytics.csv`) ya está incluido en
   `data/raw/instagram_analytics.csv` en este paquete.
3. Abrir y ejecutar `notebooks/01_eda_preprocesamiento_modelado.ipynb` de
   arriba a abajo (ya viene ejecutado con outputs reales, pero puedes
   re-ejecutarlo).

## Metodología

1. **EDA**: shape, tipos, nulos, distribución del target, correlaciones de
   variables pre-publicación vs reach, medias por categoría.
2. **Detección de data leakage**: se excluyen del feature set las métricas
   que son *consecuencia* del alcance (likes, comments, shares, saves,
   impressions, engagement_rate, followers_gained, performance_bucket_label).
3. **Features usadas**: `follower_count`, `post_hour`, `caption_length`,
   `hashtags_count` (numéricas) + `account_type`, `media_type`,
   `content_category`, `traffic_source`, `has_call_to_action`,
   `day_of_week` (categóricas) — todas ya presentes en el dataset, no
   requirieron feature engineering adicional.
4. **Preprocesamiento**: `ColumnTransformer` + `Pipeline` de scikit-learn
   (sin fuga de datos: scaler/encoder se ajustan solo sobre train).
5. **Modelado**: baseline (media), Regresión Lineal, Random Forest
   Regressor — comparación + validación cruzada (5-fold).
6. **Evaluación**: MAE, RMSE, R², chequeo de overfitting (train vs test).
7. **Análisis crítico**: el dataset no contiene señal predictiva relevante
   en las variables pre-publicación (ver resumen ejecutivo arriba).


## Licencia de datos

Dataset de uso educativo según licencia original de Kaggle.
