## Trabajo práctico labo3 - Maestría Ciencia de Datos 2025

Este repositorio documenta el desarrollo completo del trabajo práctico para la materia Laboratorio de Implementación 3, correspondiente a la Maestría en Ciencia de Datos. El proyecto se centra en resolver un problema de predicción de demanda, con el objetivo de estimar las ventas en toneladas (tn) para un conjunto de productos durante el período de febrero de 2020.

El flujo de trabajo abarca desde el análisis exploratorio de datos (EDA) y el preprocesamiento, pasando por una extensa etapa de ingeniería de características (feature engineering), hasta la experimentación con diversos modelos de predicción. Se exploraron desde técnicas de baseline como medias móviles y regresión lineal para segmentos específicos de productos, hasta modelos más complejos como LightGBM (LGBM), optimizado con Optuna.

El repositorio está estructurado para reflejar este proceso, con notebooks dedicadas a cada etapa. El resultado final es un modelo híbrido que combina las fortalezas de las diferentes técnicas para generar la predicción final.


## Estructura del proyecto

```text
.
├── data/
│   ├── predict/
│   │       ├── final/                          # predicciones obtenidas por productos
│   │       └── raw/                            # productos a predecir
│   ├── external/                               # datos externos: series de ipc / dólar
│   ├── raw/                                    # datos originales sin procesar
│   ├── interim/                                # datos intermedios procesados
│   └── processed/                              # datos procesados para modelar
├── noteoobks/      
│   ├── 0_edas/                                 # exploración del dataset
│   ├── 1_armado_base_completa/                 # armado de la base completa (rellenar ceros)
│   ├── 2_feature_engineering/                  # crear nuevas variables a partir de lags y datos externos
│   ├── 3_post_proc_train_valid_pred_split/     # post-procesamiento, definir respuesta y particionar dataset
│   ├── 4_modelado/                             
│   │          ├──series_tiempo                 # auto-arima
│   │          ├──medias                        # predicciones a partir de medias con meses previos
│   │          ├──regresion_lineal              # modelo regresión lineal
│   │          └──lgbm                          # modelo LGBM
│   └──5_modelo_final/                          # obtener predicciones finales
└── src/                                        # funciones a utilizar en notebooks/scripts
    ├── data_exploration/                       
    └── utils/                                  
```

---

## Ambiente de desarrollo

Este proyecto utiliza **Conda** para gestionar el entorno base y **Poetry** para la gestión de dependencias y empaquetado.

### 1. Crear y activar entorno Conda

```bash
conda create -n labo_3_env python=3.11
conda activate labo_3_env
```


### 2. Instalar Poetry

```bash
pip install poetry==2.1.3
```

### 3. Instalar dependencias del proyecto

Asegúrate de estar en la raíz del proyecto y con el ambiente `labo_3_env` activado:

```bash
poetry install
```

---

## Datasets necesarios para la ejecución del código

Los únicos datasets que no se generan con este repositorio y son necesarios para ejecutar las notebooks/scripts son:

* `data/raw/sell-in.txt`
* `data/raw/tb_productos.txt`
* `data/raw/tb_stocks.txt`
* `data/predict/raw/product_id_apredecir201912.txt`
* `data/external/series_ipc_dolar.csv`


## Datasets Generados según notebook

* `notebooks/1_armado_base_completa/armado_base_v2.ipynb`
    
    Genera el dataset:
        
        - data/interim/sell_in_completo.feather

* `notebooks/2_feature_engineering/feature_engineering.ipynb`
    
    Genera el dataset:
        
        - data/interim/sell_in_completo_con_fe.feather

* `notebooks\3_post_procesamiento_train_validation_predict_split\post_procesamiento_y_separar_dataset.ipynb`
    
    Genera los datasets:
        
        - data/processed/df_train.feather
        - data/processed/df_validation.feather
        - data/processed/df_predict.feather

## Predicciones obtenidas según notebooks de modelado

* `notebooks/4_modelado/medias/medias/medias.ipynb`

    - **data/predict/final/product_id_clase_3_20250613.csv**: predecir 202002 a partir de 201912.
    - **data/predict/final/product_id_clase_3_ultimos_3_meses_20250613.csv**: predecir 202002 a partir del promedio de los últimos 3 meses.
    - **data/predict/final/product_id_clase_3_ultimos_12_meses_20250613.csv**: predecir 202002 a partir del promedio de los últimos 12 meses.
    - **data/predict/final/product_id_clase_3_ultimos_24_meses_20250613.csv**: predecir 202002 a partir del promedio de los últimos 24 meses.
    - **data/predict/final/product_id_clase_3_ultimos_32_meses_20250613.csv**: predecir 202002 a partir del promedio de los últimos 32 meses.
    - **data/predict/final/product_id_clase_3_salteando_2_18_meses_20250613.csv**: predecir 202002 a partir del promedio de los últimos 18 meses salteando de a 2 meses.
    - **data/predict/final/product_id_clase_3_salteando_2_12_meses_20250613.csv**: predecir 202002 a partir del promedio de los últimos 12 meses salteando de a 2 meses.

* `notebooks/4_modelado/regresion_lineal/`:

    1. `modelo_regresion_2018_para_predecir_con_2019_v3.ipynb`: no genera ningún archivo sino que obtiene los coeficiones del modelo de regresión lineal.
    2. `medias_v3.ipynb`
        - **data/predict/final/product_id_clase_6_modelo_reg_simple_v1_magicos.csv**: predecir 202002 a partir del modelo de regresión lineal para los productos "mágicos" y utilizar la media de los últimos 12 meses para el resto de los productos.

* `notebooks/4_modelado/series_tiempo/auto_arima_script_v2.py`

    - **data/predict/final/auto_arima_predictions_statsforecast_v3_202002.csv**: predecir 202002 a parir del auto-arima.

* `notebooks/4_modelado/lgbm/lgbm_optuna_y_ensamble_semillas.ipynb`

    - **data/predict/final/lgbm_optuna_y_ensamble_semillas_{time_tag}.csv**: predecir 202002 a partir del modelado con LGBM.
    - **data/predict/final/lgbm_optuna_y_ensamble_semillas_top20_prod_y_medias_{time_tag}.csv**: predecir 202002 a partir del modelado con LGBM para el top de productos que acumulan el 20% de tn, y para el resto predicción según el archivo product_id_clase_3_ultimos_12_meses_20250613.csv.
    - **data/predict/final/lgbm_optuna_y_ensamble_semillas_top20_prod_y_magicos_{time_tag}.csv**: predecir 202002 a partir del modelado con LGBM para el top de productos que acumulan el 20% de tn, y para el resto predicción según el archivo product_id_clase_6_modelo_reg_simple_v1_magicos.csv.
    - **data/predict/final/lgbm_optuna_y_ensamble_semillas_top30_prod_y_medias_{time_tag}.csv**: predecir 202002 a partir del modelado con LGBM para el top de productos que acumulan el 30% de tn, y para el resto predicción según el archivo product_id_clase_3_ultimos_12_meses_20250613.csv.
    - **data/predict/final/lgbm_optuna_y_ensamble_semillas_top15_prod_y_medias_{time_tag}.csv**: predecir 202002 a partir del modelado con LGBM para el top de productos que acumulan el 15% de tn, y para el resto predicción según el archivo product_id_clase_3_ultimos_12_meses_20250613.csv.
    - **data/predict/final/lgbm_optuna_y_ensamble_semillas_magicos_medias_{time_tag}.csv**: predecir 202002 a partir del modelado con LGBM para los productos mágicos, y para el resto predicción según el archivo product_id_clase_6_modelo_reg_simple_v1_magicos.csv.

* `notebooks/5_modelo_final`

    - **data/predict/final/predicciones_final_{time_tag}.csv**: archivo de prediccion final seleccionado, misma alternativa que data/predict/final/lgbm_optuna_y_ensamble_semillas_top20_prod_y_magicos_{time_tag}.csv