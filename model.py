# -*- coding: utf-8 -*-
"""
Création du modèle
"""
from pydantic import BaseModel

class BankClient(BaseModel):
    EXT_SOURCE_3 = 0.156898
    OBS_60_CNT_SOCIAL_CIRCLE=0.08
    EXT_SOURCE_2=0.307507
    OBS_30_CNT_SOCIAL_CIRCLE=0.08
    AMT_REQ_CREDIT_BUREAU_YEAR=0.04
    CNT_CHILDREN=0.0
    CNT_FAM_MEMBERS=0.0
    EXT_SOURCE_1=0.06535
    PAYMENT_RATE=0.376626
    FLAG_PHONE=1.0
