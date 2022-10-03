# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 21:40:41 2020
@author: win10
"""

# 1. Library imports
import uvicorn
from fastapi import FastAPI
from model import BankClient
import numpy as np
import pickle
import pandas as pd
# 2. Create the app object
app = FastAPI()
pickle_in = open("LR_SMOTE.pkl", "rb")
classifier = pickle.load(pickle_in)


# 3. Index route, opens automatically on http://127.0.0.1:8000
@app.get('/')
def index():
    return {'message': 'Hello, World'}


# 4. Route with a single parameter, returns the parameter within a message
#    Located at: http://127.0.0.1:8000/AnyNameHere
@app.get('/{name}')
def get_name(name: str):
    return {'Welcome To Krish Youtube Channel': f'{name}'}


# 3. Expose the prediction functionality, make a prediction from the passed
#    JSON data and return the predicted Bank Note with the confidence
@app.post('/predict')
def predict_defautclient(data: BankClient):
    data = data.dict()
    EXT_SOURCE_3=data['EXT_SOURCE_3'] #0.156898
    OBS_60_CNT_SOCIAL_CIRCLE=data['OBS_60_CNT_SOCIAL_CIRCLE'] #0.08
    EXT_SOURCE_2=data['EXT_SOURCE_2'] #0.307507
    OBS_30_CNT_SOCIAL_CIRCLE=data['OBS_30_CNT_SOCIAL_CIRCLE'] #0.08
    AMT_REQ_CREDIT_BUREAU_YEAR=data['AMT_REQ_CREDIT_BUREAU_YEAR'] #0.04
    CNT_CHILDREN=data['CNT_CHILDREN'] #0.0
    CNT_FAM_MEMBERS=data['CNT_FAM_MEMBERS'] #0.0
    EXT_SOURCE_1=data['EXT_SOURCE_1'] #0.06535
    PAYMENT_RATE=data['PAYMENT_RATE'] #0.376626
    FLAG_PHONE=data['FLAG_PHONE'] #1.0
    data_in = [[
        EXT_SOURCE_3, OBS_60_CNT_SOCIAL_CIRCLE, EXT_SOURCE_2,
        OBS_30_CNT_SOCIAL_CIRCLE, AMT_REQ_CREDIT_BUREAU_YEAR, CNT_CHILDREN,
        CNT_FAM_MEMBERS, EXT_SOURCE_1, PAYMENT_RATE, FLAG_PHONE
    ]]
    prediction = classifier.predict(data_in)
    probability = classifier.predict_proba(data_in)

    # if (prediction[0] > 0.5):
    #     prediction = "Default client predicted"
    # else:
    #     prediction = "Good client predicted"
    return {'prediction': str(prediction[0]), 'probability': str(probability)}


# 5. Run the API with uvicorn
#    Will run on http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)

#uvicorn app:app --reload
# requirement uvlopp

# # 1. Library imports
# import uvicorn  #ASGI
# from fastapi import FastAPI, HTTPException
# # from BankNotes import BankNote
# from model import BankClient #InfoClient, ClientModel
# import numpy as np
# import pickle
# import pandas as pd

# from contextlib import contextmanager
# # from P7 import params, utils, preprocess

# # 2. Create the app object
# app = FastAPI()

# # 3. Load the model
# model = pd.read_pickle(r'../model/LR_SMOTE.pkl')

# # 3. Index route, opens automatically on http://127.0.0.1:8000
# @app.get('/')
# def index():
#     '''
#     This is a first docstring.
#     '''
#     return {'message': 'API is running well'}

# #  3. Expose the prediction functionality, make a prediction from the passed
# @app.post('/predict')
# def predict_client(EXT_SOURCE_3=0.156898,
#                    OBS_60_CNT_SOCIAL_CIRCLE=0.08,
#                    EXT_SOURCE_2=0.307507,
#                    OBS_30_CNT_SOCIAL_CIRCLE=0.08,
#                    AMT_REQ_CREDIT_BUREAU_YEAR=0.04,
#                    CNT_CHILDREN=0.0,
#                    CNT_FAM_MEMBERS=0.0,
#                    EXT_SOURCE_1=0.06535,
#                    PAYMENT_RATE=0.376626,
#                    FLAG_PHONE=1.0):
#     '''
#     This is a first docstring.
#     '''
#     prediction = 0

#     # prediction = model.predict([[EXT_SOURCE_3, OBS_60_CNT_SOCIAL_CIRCLE,
#     #                            EXT_SOURCE_2, OBS_30_CNT_SOCIAL_CIRCLE,
#     #                            AMT_REQ_CREDIT_BUREAU_YEAR, CNT_CHILDREN,
#     #                            CNT_FAM_MEMBERS, EXT_SOURCE_1, PAYMENT_RATE,
#     #                            FLAG_PHONE]])
#     # probability = model.predict_proba(data_in)

#     if(prediction[0]>0.5):
#         prediction="Fake note"
#     else:
#         prediction="Its a Bank note"
#     # return {'prediction': str(prediction)}
#     return {'message': 'API is running well not'}


#     # return {'prediction': prediction[0]}
#     # return {'prediction': str(prediction[0]), 'probability': str(probability)}

# # 5. Run the API with uvicorn
# #    Will run on http://127.0.0.1:8000
# if __name__ == '__main__':
#     uvicorn.run(app, host='127.0.0.1', port=8000)
