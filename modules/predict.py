# <YOUR_IMPORTS>
import glob
import json
import logging
import os

import dill
import joblib
import pandas as pd
from datetime import datetime

path = os.environ.get('PROJECT_PATH', '.')

def predict():
    # <YOUR_CODE>
    with open(f'{path}/data/models/cars_pipe_202306270946.pkl', 'rb') as file:
        model = dill.load(file)
    df_pred = pd.DataFrame(columns=['card_id', 'pred'])
    for filename in glob.glob(f'{path}/data/test/*.json'):
        with open(filename) as fin:

            logging.info(f'Prediction for test - {filename}')
            form = json.load(fin)
            df = pd.DataFrame.from_dict([form])
            y = model.predict(df)
            x = {'car_id': df.id, 'pred': y}
            df1 = pd.DataFrame(x)
            df_pred = pd.concat([df_pred, df1], axis=0)

    logging.info(f'Prediction result')
    df_pred.to_csv(f'{path}/data/predictions/pred_{datetime.now().strftime("%Y%m%d%H%M")}.csv')
    pass


if __name__ == '__main__':
    predict()
