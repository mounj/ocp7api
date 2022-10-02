# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 21:51:19 2020
@author: win10
"""
from pydantic import BaseModel


# # 2. Class which describes Bank Notes measurements
# class BankNote(BaseModel):
#     EXT_SOURCE_3: float
#     OBS_60_CNT_SOCIAL_CIRCLE: float
#     EXT_SOURCE_2: float
#     OBS_30_CNT_SOCIAL_CIRCLE: float
#     AMT_REQ_CREDIT_BUREAU_YEAR: float
#     CNT_CHILDREN: float
#     CNT_FAM_MEMBERS: float
#     EXT_SOURCE_1: float
#     PAYMENT_RATE: float
#     FLAG_PHONE: float

# from pydantic import BaseModel


# 2. Class which describes Bank Client Information
class BankClient(BaseModel):
    EXT_SOURCE_3: float
    OBS_60_CNT_SOCIAL_CIRCLE: float
    EXT_SOURCE_2: float
    OBS_30_CNT_SOCIAL_CIRCLE: float
    AMT_REQ_CREDIT_BUREAU_YEAR: float
    CNT_CHILDREN: float
    CNT_FAM_MEMBERS: float
    EXT_SOURCE_1: float
    PAYMENT_RATE: float
    FLAG_PHONE: float
