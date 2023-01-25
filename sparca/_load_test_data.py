import importlib
import pandas as pd
import json


def load_test_data():

    with importlib.resources.path('sparca', 'test_data') as data_root:
        data_path = f'{data_root}/sparca_test_data_sample.csv'
        json_path = f'{data_root}/sparca_test_data_info.json'

    test_data = pd.read_csv(data_path)

    with open(json_path) as info_json:
        data_info = json.load(info_json)

    return test_data, data_info

    

    