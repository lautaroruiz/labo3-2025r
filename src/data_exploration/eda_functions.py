import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from typing import Tuple, Dict


def load_and_clean_data(base_dir) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Carga y limpia los datos de sell_in, tb_productos y tb_stocks."""
    data_path_sell_in = base_dir / "data" / "raw" / "sell-in.txt"
    data_path_tb_productos = base_dir / "data" / "raw" / "tb_productos.txt"
    data_path_tb_stocks = base_dir / "data" / "raw" / "tb_stocks.txt"

    sell_in = pd.read_csv(
        data_path_sell_in, sep="\t", encoding="utf-8"
    ).drop_duplicates()
    tb_productos = pd.read_csv(
        data_path_tb_productos, sep="\t", encoding="utf-8"
    ).drop_duplicates()
    tb_stocks = pd.read_csv(
        data_path_tb_stocks, sep="\t", encoding="utf-8"
    ).drop_duplicates()
    return sell_in, tb_productos, tb_stocks


def show_basic_info(df_dict: Dict[str, pd.DataFrame], n: int = 5):
    """Muestra información básica y los primeros registros de cada DataFrame."""
    for name, df in df_dict.items():
        print(f"Datos {name}: {df.shape}")
        display(name)
        display(df.head(n))


def analyze_duplicates(df: pd.DataFrame, cols: list = None):
    """Analiza duplicados en el DataFrame, opcionalmente por columnas."""
    if cols:
        print(df[cols].drop_duplicates().shape)
    else:
        print(df.drop_duplicates().shape)


def plot_grouped_data(
    df: pd.DataFrame,
    agg_dict: dict,
    group_col: str = "periodo",
    title_prefix: str = "",
    by_product_id: int = None,
    by_customer_id: int = None,
    return_data: bool = False,
):
    """Agrupa y grafica datos por una columna, usando agregaciones dadas."""
    # Filtrar por producto o cliente si se especifica
    if by_product_id is not None:
        title_prefix = f"Producto {by_product_id} - {title_prefix}"
        df = df[df["product_id"] == by_product_id]
    if by_customer_id is not None:
        title_prefix = f"Cliente {by_customer_id} - {title_prefix}"
        df = df[df["customer_id"] == by_customer_id]
    # Agrupar y agregar datos
    data = (
        df.groupby(group_col)
        .agg(agg_dict)
        .reset_index(drop=False)
        .sort_values(by=group_col, ascending=True)
    )
    data["fecha"] = pd.to_datetime(data[group_col].astype(str), format="%Y%m")
    data = data.set_index("fecha")
    data = data.drop(columns=[group_col])
    data.plot(
        subplots=True,
        layout=(-1, 2),
        figsize=(12, 6),
        title=[f"{title_prefix} {col}" for col in data.columns],
    )
    plt.tight_layout()
    plt.show()
    if return_data:
        print(f"Datos agrupados por {group_col}:")
        display(data.head())
        return data


def analyze_top_products(
    df: pd.DataFrame,
    product_col: str = "product_id",
    value_col: str = "tn",
    top_n: int = 10,
):
    """Analiza y grafica los productos top por suma de value_col."""
    top_products = df.groupby(product_col)[value_col].sum().sort_values(ascending=False)
    top_products.head(top_n).plot(kind="bar")
    plt.title(f"Top {top_n} productos por {value_col} acumulado")
    plt.show()
    top_products = top_products.reset_index(drop=False)
    top_products["tn_cumsum"] = (
        top_products["tn"].cumsum().round(2) / top_products["tn"].sum().round(2) * 100
    )
    return top_products


def analyze_top_customers(
    df: pd.DataFrame,
    customer_col: str = "customer_id",
    value_col: str = "tn",
    top_n: int = 10,
):
    """Analiza y grafica los clientes top por suma de value_col."""
    top_customers = (
        df.groupby(customer_col)[value_col].sum().sort_values(ascending=False)
    )
    top_customers.head(top_n).plot(kind="bar")
    plt.title(f"Top {top_n} clientes por {value_col} acumulado")
    plt.show()
    top_customers = top_customers.reset_index(drop=False)
    top_customers["tn_cumsum"] = (
        top_customers["tn"].cumsum().round(2) / top_customers["tn"].sum().round(2) * 100
    )
    return top_customers


def describe_column_uniques(df: pd.DataFrame):
    """Describe valores únicos y frecuencias para columnas tipo object."""
    for col in df.columns:
        if df[col].dtype == "object":
            print(f"Columna '{col}' tiene {df[col].nunique()} valores únicos.")
            print(df[col].value_counts(dropna=False, normalize=True))
            print("\n")


def plot_histogram(df: pd.DataFrame, col: str, quantile: float = 0.9, bins: int = 20):
    """Grafica histograma de una columna, filtrando por un cuantil."""
    limite = df[col] < df[col].quantile(quantile)
    df[limite][col].hist(bins=bins, figsize=(12, 6), grid=True)
    plt.title(f"Histograma de {col} (por debajo del {quantile*100:.0f} percentil)")
    plt.show()


def plot_top_products_over_time(
    df: pd.DataFrame, response: str = "tn", product_ids: list = [], n: int = 10
):
    """Grafica las toneladas de los n productos más vendidos por período."""
    if product_ids:
        df_top_products = df[df["product_id"].isin(product_ids)]
    else:
        top_products = df.groupby("product_id")[response].sum().nlargest(n).index
        df_top_products = df[df["product_id"].isin(top_products)]
    df_top_products_agg = (
        df_top_products.groupby(["periodo", "product_id"])[response].sum().reset_index()
    )
    df_top_products_agg["fecha"] = pd.to_datetime(
        df_top_products_agg["periodo"].astype(str), format="%Y%m"
    )

    plt.figure(figsize=(12, 5))
    sns.lineplot(
        data=df_top_products_agg, x="fecha", y=response, hue="product_id", marker="o"
    )
    plt.title(f"Productos - {response} por período")
    plt.ylabel(f"{response}")
    plt.xlabel("Período")
    plt.legend(title="Product ID")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_top_customers_over_time(
    df: pd.DataFrame, response: str = "tn", customer_ids: list = [], n: int = 10
):
    """Grafica las toneladas de los n clientes más vendidos por período."""
    if customer_ids:
        df_top_customers = df[df["customer_id"].isin(customer_ids)]
    else:
        top_customers = df.groupby("customer_id")[response].sum().nlargest(n).index
        df_top_customers = df[df["customer_id"].isin(top_customers)]
    df_top_customers_agg = (
        df_top_customers.groupby(["periodo", "customer_id"])[response]
        .sum()
        .reset_index()
    )
    df_top_customers_agg["fecha"] = pd.to_datetime(
        df_top_customers_agg["periodo"].astype(str), format="%Y%m"
    )

    plt.figure(figsize=(12, 5))
    sns.lineplot(
        data=df_top_customers_agg, x="fecha", y=response, hue="customer_id", marker="o"
    )
    plt.title(f"Clientes - {response} por período")
    plt.ylabel(f"{response}")
    plt.xlabel("Período")
    plt.legend(title="Customer ID")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
