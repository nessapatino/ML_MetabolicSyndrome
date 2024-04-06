![](./img/imagen_readme.webp)

# Modelo de Clasificación para el Síndrome Metabólico

## Descripción
Este proyecto contiene un modelo de clasificación supervisada diseñado para identificar pacientes con Síndrome Metabólico. Utilizando datos clínicos, el modelo clasifica a los individuos en dos categorías: Clase 0 (pacientes sanos) y Clase 1 (pacientes con Síndrome Metabólico).

## Modelos Baseline
Se han evaluado varios modelos de clasificación como línea base, incluyendo:

- Regresión Logística
- Árbol de Decisión
- Random Forest
- CatBoost
- XGBoost
- LightGBM
- SVC (Support Vector Classifier)
- KNN (K-Nearest Neighbors)


Inicialmente, el modelo SVC mostró la mejor métrica de desempeño.

## Optimización de Hiperparámetros
Tras un proceso de optimización de hiperparámetros, el modelo XGBoost emergió como el más eficaz, alcanzando un recall de 0,97 sobre el conjunto de prueba. Esto indica una alta capacidad del modelo para identificar correctamente a los pacientes con Síndrome Metabólico.

## Resultados
- Alta precisión para la clase 0 (Pacientes Sanos): El modelo tiene una precisión de 0.97 para predecir la clase 0. Esto significa que es muy efectivo en identificar a los pacientes sanos, con muy pocos falsos positivos.

- Recall destacado para la clase 1 (Pacientes Enfermos): El recall para la clase 1 es 0.96, indicando que el modelo es altamente sensible y capaz de identificar a la mayoría de los pacientes con síndrome metabólico.

## Contenido

El análisis exploratorio de datos incluye:

    ML_MetabolicSyndrome/
    │
    ├── presentacion/
    │   └── Presentacion_ML_sindrome_metabolico.pptx
    │
    ├── src/
    │    │
    │    ├── data/
    │    │   └── MetabolicSyndrome.csv
    │    │
    │    ├── notebooks/
    │    │   ├── 1_Modelo_sin_cat.ipynb
    │    │   ├── 2_Modelo_no_supervisado.ipynb
    │    │   ├── 3_Modelo_cat_features.ipynb
    │    │   ├── 4_Modelo_DL.ipynb
    │    │   └── 5_Pipeline.ipynb
    │    │    
    │    ├── results_notbook/
    │    │   └── Project_resumen.py
    │    │
    │    └── utils/
    │        ├── funciones_toolbox_ml_final.py
    │        └── modulos.py
    │
    │
    └── README.md

## Requisitos

Para ejecutar los Notebooks o utilizar los scripts, se recomienda tener instalado:

- Python (preferiblemente la versión 3.11) con las bibliotecas de análisis de datos como Pandas, NumPy, Matplotlib/Seaborn y bibliotecas para la construcción de modelos de machine learning como Sci-kit Learn y TensorFlow.

## Uso

1. Clona este repositorio en tu máquina local:

    ```bash
    git clone https://github.com/nessapatino/ML_MetabolicSyndrome

2. Abre los Notebooks utilizando Jupyter Notebook o JupyterLab:

    ```bash
    jupyter notebook
    ```

3. Explora los archivos `.ipynb` para acceder al código utilizado en el análisis exploratorio de datos y modelos.

4. Para cualquier pregunta, colaboración o comentario, no dudes en ponerte en contacto conmigo:

    - **Nombre**: Vanessa Patiño Del Hoyo
    - **Correo electrónico**: vanepatino1991@gmail.com
    - [**LinkedIn**](https://wwww.linkedin.com/in/vanessapatino)