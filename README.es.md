# Regresión Lineal — Predicción de Primas de Seguro de Salud

> Pipeline de regresión para predecir primas anuales de seguro a partir de datos demográficos de pacientes: EDA completo que revela el impacto desproporcionado del tabaquismo, ingeniería de características de interacción y un modelo de Regresión Lineal que alcanza R² ≈ 0,80.

---

## Problema

Una compañía de seguros quiere predecir la prima anual de seguro de salud (`charges`) para cada cliente basándose en sus datos demográficos y de salud. Se utiliza un dataset ensamblado por un equipo de médicos a partir de datos del sector para entrenar el modelo. Este es un problema de regresión supervisada.

## Dataset

- **Fuente:** Dataset Medical Insurance Cost (~1.338 filas)
- **Target:** `charges` — prima anual de seguro en USD (continua)
- **Características:**

| Característica | Tipo | Descripción |
|---|---|---|
| age | Numérica | Edad del beneficiario principal |
| sex | Categórica | Género del beneficiario principal |
| bmi | Numérica | Índice de masa corporal |
| children | Numérica | Número de dependientes |
| smoker | Categórica | ¿Es fumador/a la persona? |
| region | Categórica | Región residencial de EE.UU. (NE / SE / SW / NW) |

## Pipeline de EDA y Preprocesamiento

| Paso | Acción |
|---|---|
| Duplicados | 1 duplicado eliminado |
| Nulos | Ninguno encontrado |
| Manejo de outliers | BMI limitado al umbral superior del IQR (muy pocos valores extremos eliminados) |
| Codificación | `sex` y `smoker` → pd.factorize(); `region` → one-hot (drop_first para evitar la trampa de variables ficticias) |
| Característica de interacción | `bmi_smoker = bmi × smoker` — el análisis multivariante mostró que los fumadores con alto BMI forman un clúster diferenciado de altas primas |
| Escalado | MinMaxScaler en todas las columnas de características |
| Selección de características | SelectKBest (f_regression, k=7) — división antes de la selección para prevenir fuga de datos |
| División | 80/20 entrenamiento/prueba |

**Hallazgo clave del EDA:** `smoker` es con diferencia la característica predictiva individual más importante. Los fumadores pagan primas dramáticamente más altas — la distribución de charges tiene un pronunciado pico secundario impulsado completamente por este grupo. El término de interacción `bmi_smoker` captura que este efecto se amplifica con un BMI alto.

**Correlación con charges (aproximada):**

| Característica | Señal |
|---|---|
| smoker | Muy fuerte |
| age | Moderada positiva |
| bmi | Moderada positiva (amplificada por la interacción con smoker) |
| children | Débil positiva |
| region | Negligible |
| sex | Casi nula |

## Resultados del Modelo

- **Modelo:** `sklearn.linear_model.LinearRegression`
- **Puntuación R²:** ≈ **0,80** (80% de la varianza en las primas de seguro explicada)
- **Nota:** La Regresión Lineal no tiene hiperparámetros que optimizar — la puntuación R² es la evaluación final.

## Conclusiones Clave

- **Una característica domina:** El estado de fumador por sí solo representa la mayor parte de la varianza explicable. Cualquier modelo que no capture esta distinción fallará gravemente.
- **Las características de interacción importan:** `bmi_smoker` es más predictivo que BMI solo porque la relación no es aditiva — un no fumador con BMI alto paga mucho menos que un fumador con BMI alto. Los modelos lineales necesitan que esto se ingenierie explícitamente; los modelos basados en árboles lo encontrarían automáticamente.
- **R² ≈ 0,80 tiene un techo:** El 20% restante de la varianza probablemente proviene del historial de reclamaciones, condiciones preexistentes y factores de estilo de vida no capturados en estas 6 características — no es un fallo del modelo.

## Stack Tecnológico

`Python` · `scikit-learn` · `pandas` · `NumPy` · `Matplotlib` · `Seaborn`

## Ejecutar Localmente

```bash
git clone https://github.com/matthewkane-ml/ML_Linear_Regression_MTK.git
cd ML_Linear_Regression_MTK
pip install -r requirements.txt
python src/app.py
```

## Próximos Pasos

- Probar **regresión Ridge o Lasso** para añadir regularización y comparar si penalizar los coeficientes débiles mejora la generalización
- Ingenierizar una interacción **smoker × age** para probar si la penalización por tabaquismo crece con la edad
- Comparar con un **Gradient Boosting Regressor** — los métodos basados en árboles encuentran relaciones no lineales e interacciones automáticamente, lo que probablemente empujaría el R² por encima de 0,85

---

**Autor:** Matthew Kane — [LinkedIn](https://www.linkedin.com/in/thomas-k-392094410/) · [Portafolio GitHub](https://github.com/matthewkane-ml)
