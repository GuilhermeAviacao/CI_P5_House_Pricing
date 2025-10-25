import pandas as pd

def load_house_data():
    """Full house prices dataset"""
    df = pd.read_csv("outputs/datasets/collection/HouseFeaturesPrices.csv")
    return df