import importlib
import pandas as pd
import json


def load_test_data():
    """
    Loads example data provided with sparca for testing and demonstration purposes.
    Test data acquired from
     "Zieba, M., Tomczak, S. K., & Tomczak, J. M. (2016). Ensemble Boosted Trees with Synthetic Features 
     Generation in Application to Bankruptcy Prediction. Expert Systems with Applications."
     http://archive.ics.uci.edu/ml/datasets/polish+companies+bankruptcy+data

    Returns
    -------
    test_data: pandas.DataFrame
        DataFrame containing test data
    test_data_info: dict
        Essential information about test data set
    """

    with importlib.resources.path('sparca', 'test_data') as data_root:
        data_path = f'{data_root}/sparca_test_data_sample.csv'
        json_path = f'{data_root}/sparca_test_data_info.json'

    test_data = pd.read_csv(data_path)

    with open(json_path) as info_json:
        data_info = json.load(info_json)

    return test_data, data_info

    

    