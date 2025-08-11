import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA, SeasonalNaive
from pathlib import Path
import logging
import sys

from src.utils.utils import get_base_dir
base_dir = get_base_dir()
base_dir

# ================================
# Configuración de logging
# ================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ================================
# Configuración
# ================================
FORECAST_HORIZON = 2  # Porque queremos predecir hasta 202002

# ================================
# Cargar y preparar los datos
# ================================

# Paths
DATA_PATH_SELL_IN = base_dir / "data/raw/sell-in.txt"
PREDICT_FILE = base_dir / "data/predict/raw/product_id_apredecir201912.txt"
OUTPUT_FILE = base_dir / "data/predict/final/auto_arima_predictions_statsforecast_v3_202002.csv"


def main():
    try:
        # Cargar ventas
        logger.info("Cargando datos de ventas desde '%s'...", DATA_PATH_SELL_IN)
        sell_in = pd.read_csv(
            DATA_PATH_SELL_IN, sep="\t", encoding="utf-8"
        ).drop_duplicates()
        logger.info("Datos de ventas cargados: %d filas.", sell_in.shape[0])
    except Exception as e:
        logger.error("Error al cargar datos de ventas: %s", e)
        sys.exit(1)

    # Agrupar por periodo y producto
    data_baseline = (
        sell_in.groupby(["periodo", "product_id"])
        .agg({"tn": "sum"})
        .reset_index(drop=False)
    )
    logger.info("Datos agrupados: %d filas.", data_baseline.shape[0])

    try:
        # Leer archivo de productos a predecir
        logger.info("Leyendo productos a predecir desde '%s'...", PREDICT_FILE)
        df_pred = pd.read_csv(PREDICT_FILE, sep="\t", encoding="utf-8")
        logger.info("Productos a predecir: %d.", df_pred.shape[0])
    except Exception as e:
        logger.error("Error al leer productos a predecir: %s", e)
        sys.exit(1)

    # Filtrar productos relevantes
    product_ids = df_pred["product_id"].unique()
    df = data_baseline[data_baseline["product_id"].isin(product_ids)].reset_index(
        drop=True
    )
    logger.info("Filtrado de productos relevantes: %d filas.", df.shape[0])

    # Convertir 'periodo' a datetime
    try:
        df["periodo"] = pd.to_datetime(df["periodo"].astype(str), format="%Y%m")
    except Exception as e:
        logger.error("Error al convertir 'periodo' a datetime: %s", e)
        sys.exit(1)

    # Renombrar columnas para StatsForecast
    df_sf = df.rename(columns={"product_id": "unique_id", "periodo": "ds", "tn": "y"})

    # ================================
    # Crear y entrenar el modelo
    # ================================

    logger.info(
        "Entrenando modelos AutoARIMA, ETS y SeasonalNaive para %d series...",
        df_sf["unique_id"].nunique(),
    )
    models = [
        AutoARIMA(season_length=6),
        SeasonalNaive(season_length=6),
    ]
    sf = StatsForecast(models=models, freq="MS", n_jobs=-1)

    try:
        forecast = sf.forecast(df=df_sf, h=FORECAST_HORIZON)
        logger.info(
            "Predicción realizada para horizonte de %d meses con todos los modelos.",
            FORECAST_HORIZON,
        )
    except Exception as e:
        logger.error("Error durante la predicción: %s", e)
        sys.exit(1)

    # ================================
    # Filtrar predicciones para 202002
    # ================================

    n_series = df_sf["unique_id"].nunique()
    dates = pd.date_range(start="2020-01-01", periods=FORECAST_HORIZON, freq="MS")
    forecast["ds"] = dates.tolist() * n_series

    forecast_202002 = forecast[forecast["ds"] == "2020-02-01"]
    # Guardar predicciones de cada modelo en archivos separados
    modelos = ["AutoARIMA", "SeasonalNaive"]
    for modelo in modelos:
        out_file = OUTPUT_FILE.replace(".csv", f"_{modelo}.csv")
        df_modelo = forecast_202002[["unique_id", modelo]].copy()
        df_modelo.columns = ["product_id", "tn"]
        try:
            df_modelo.to_csv(out_file, index=False)
            logger.info(f"Predicciones del modelo {modelo} guardadas en: {out_file}")
        except Exception as e:
            logger.error(f"Error al guardar resultados del modelo {modelo}: %s", e)
            sys.exit(1)


# ================================
# Protección para multiprocessing en Windows
# ================================
if __name__ == "__main__":
    main()  # Esto previene errores de multiprocessing en Windows

# ejecutar notebook desde terminal con "poetry run notebooks\4_modelado\series_tiempo\auto_arima_script_v2.py"