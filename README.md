# AgroInsight India: predicción y segmentación del rendimiento agrícola

Proyecto Integrador de Ciencia de Datos, Maestría en Inteligencia Artificial, Universidad de La Salle (2026).

Dataset: [Crop Yield in Indian States](https://www.kaggle.com/datasets/akshatgupta7/crop-yield-in-indian-states-dataset) (Kaggle, CC BY 4.0), registros 2015-2020.

## Resultados principales
| Tarea | Modelo | Métrica (prueba) |
|---|---|---|
| Regresión de `Yield` | XGBoost | MAE 1.42 t/ha, R² log 0.940 |
| Clasificación Bajo/Medio/Alto | XGBoost | F1 macro 0.694, AUC 0.861 |
| Segmentación | K-Means (K = 4) | Silueta 0.274 |

## Estructura
- `data/`: `raw/dataset_raw.csv` y particiones en `processed/`
- `notebooks/`: los 4 cuadernos (ejecutar en orden en Google Colab)
- `src/crop_features.py`: transformaciones propias (necesarias para cargar los `.pkl`)
- `models/`: preprocesador y modelos entrenados
- `reports/`: métricas en JSON y figuras
- `dashboard/`: aplicación Streamlit (`app.py`) y sus datos (`assets/`)

## Ejecutar el dashboard
```bash
pip install -r dashboard/requirements.txt
streamlit run dashboard/app.py
```
En Streamlit Community Cloud: *New app* → este repositorio → archivo principal `dashboard/app.py`,
con la misma versión de Python usada en Colab (3.13).

## Reproducir todo el proyecto
`pip install -r requirements.txt` y ejecutar los cuadernos 01 → 04 en orden. Semilla global: 42.
