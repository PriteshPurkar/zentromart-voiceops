"""
Loads the ZentroMart CSV datasets.
All IDs, pincodes, and phones are loaded strictly as strings (dtype=str).
"""
from pathlib import Path
import pandas as pd

from src.config import PATHS

# Cache to avoid reloading the same CSV multiple times
_cache: dict[str, pd.DataFrame] = {}

def load_csv(name: str) -> pd.DataFrame:
    """
    Load a CSV from the data directory.
    Example: load_csv("orders") -> returns orders.csv as a DataFrame.
    """
    if name in _cache:
        return _cache[name]
    
    path = PATHS.data / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")
        
    # dtype=str is CRITICAL so IDs like '0001234' are not converted to ints
    df = pd.read_csv(path, dtype=str)
    
    # Replace empty strings with None (pandas NaN)
    df = df.replace({"": None})
    
    _cache[name] = df
    return df

# Convenience getters
def get_customers() -> pd.DataFrame:
    return load_csv("customers")

def get_orders() -> pd.DataFrame:
    return load_csv("orders")

def get_order_items() -> pd.DataFrame:
    return load_csv("order_items")

def get_products() -> pd.DataFrame:
    return load_csv("products")

def get_pincodes() -> pd.DataFrame:
    return load_csv("pincodes")

def get_holidays() -> pd.DataFrame:
    return load_csv("holidays")

def get_support_logs() -> pd.DataFrame:
    return load_csv("support_logs")