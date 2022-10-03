import streamlit as st
from streamlit_shap import st_shap
import shap
import pandas as pd
# pour réparer le problème avec XGBOOST !!!!!!
from pandas import MultiIndex, Int16Dtype
import numpy as np
import matplotlib
#import seaborn as sns
import requests
import json
import pickle
import os
from sklearn.preprocessing import StandardScaler
import io
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objs as go
import streamlit.components.v1 as components
import xgboost as xgb
from PIL import Image
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

st.sidebar.title("Prêt à dépenser")
#st.write ('---debug chargement image ')
########################################################
# Loading images to the website
########################################################
image = Image.open("images/credit.jpg")
st.sidebar.image(image)

#st.write ('---debug chargement du modèle ')
# Chargement du modèle
#current_path = os.getcwd()
#credit_path = os.path.join(current_path, 'classifier.pkl')
#with open(credit_path, 'rb') as handle:
#    model = pickle.load(handle)
model = xgb.XGBClassifier()
model.load_model('classifier_xgb_opt.json')

#st.write ('---debug chargement des fonctions')
#def st_shap(plot, height=None):
#    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
#    components.html(shap_html, height=height)


def prediction(X):
    prediction = model.predict(X)
    return prediction


def impPlot(imp, name):
    figure = px.bar(imp,
                    x=imp.values,
                    y=imp.keys(),
                    labels={
                        'x': 'Importance Value',
                        'index': 'Columns'
                    },
                    text=np.round(imp.values, 2),
                    title=name + ' Feature Selection Plot',
                    width=1000,
                    height=600)
    figure.update_layout({
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(0, 0, 0, 0)',
    })
    st.plotly_chart(figure)


def chargement_data(path):
    dataframe = pd.read_csv(path)
    liste_id = dataframe['SK_ID_CURR'].tolist()
    return dataframe, liste_id


#st.write ('---debug lecture df1')
# Pour alimenter le modèle avec les informations du client - les variables sont encodées !!!!!!
examples_file = 'df1.csv'
dataframe, liste_id = chargement_data(examples_file)

#st.write ('---debug les pages')


def main_page():

    st.sidebar.markdown("# Calcul du risque")
    #st.write ('---Calcul du risque')

    st.title('Calcul du risque de remboursement de prêt')

    st.subheader(
        "Prédictions de scoring client et positionnement dans l'ensemble des clients"
    )

    # Affichage 1ère fois

    if 'client' not in st.session_state:
        st.session_state.client = 0
    else:
        # Retour pagination
        id_input = st.session_state.client
        #st.write ('---debug client retour pagination main ' ,id_input)

    id_input = st.selectbox(
        'Choisissez le client que vous souhaitez visualiser', liste_id)
    st.session_state.client = id_input

    client_infos = dataframe[dataframe['SK_ID_CURR'] == id_input].drop(
        ['SK_ID_CURR'], axis=1)
    client_infos.to_dict(orient='records')

    result = ""

    X1 = dataframe[dataframe['SK_ID_CURR'] == id_input]
    X = X1[[
        'CODE_GENDER', 'AGE', 'CNT_CHILDREN', 'DEF_30_CNT_SOCIAL_CIRCLE',
        'NAME_EDUCATION_TYPE_High education',
        'NAME_EDUCATION_TYPE_Low education',
        'NAME_EDUCATION_TYPE_Medium education',
        'ORGANIZATION_TYPE_Construction', 'ORGANIZATION_TYPE_Electricity',
        'ORGANIZATION_TYPE_Government/Industry', 'ORGANIZATION_TYPE_Medicine',
        'ORGANIZATION_TYPE_Other/Construction/Agriculture',
        'ORGANIZATION_TYPE_School', 'ORGANIZATION_TYPE_Services',
        'ORGANIZATION_TYPE_Trade/Business',
        'OCCUPATION_TYPE_Accountants/HR staff/Managers',
        'OCCUPATION_TYPE_Core/Sales staff', 'OCCUPATION_TYPE_Laborers',
        'OCCUPATION_TYPE_Medicine staff',
        'OCCUPATION_TYPE_Private service staff', 'OCCUPATION_TYPE_Tech Staff',
        'NAME_FAMILY_STATUS_Married', 'NAME_FAMILY_STATUS_Single',
        'AMT_INCOME_TOTAL', 'INCOME_CREDIT_PERC', 'DAYS_EMPLOYED_PERC',
        'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3'
    ]]

    result = prediction(X)

    if result == 1:
        if int(X1['TARGET']) == 1:
            pred = 'Rejected (True Positive)'
        else:
            pred = 'Approved (False Positive)'
    else:
        if int(X1['TARGET']) == 1:
            pred = 'Rejected (False Negative)'
        else:
            pred = 'Approved (True Negative)'

    st.success('Your loan is {}'.format(pred))


def page2():

    st.sidebar.markdown("# Interprétabilité")
    #st.write ('--- Interprétation')

    st.title("Interprétabilité du modèle")

    #st.write ('--- session_state.client page 2')
    id_input = st.session_state.client

    st.write('Pour le client  ', id_input,
             ' poids des variables dans le modèle XGBOOST')

    # informations du client
    st.header("Informations du client")
    examples_file = 'application_API.csv'
    application, liste_id = chargement_data(examples_file)
    application.drop(['Unnamed: 0'], axis=1, inplace=True)
    X_infos_client = application[application['SK_ID_CURR'] == id_input]
    st.write(X_infos_client)

    # scatter plot

    st.header("OCCUPATION_TYPE / EXT_SOURCE_3 / target")
    fig = px.box(application,
                 x="OCCUPATION_TYPE",
                 y="EXT_SOURCE_3",
                 color="TARGET",
                 notched=True)
    st.plotly_chart(fig)

    st.header("OCCUPATION_TYPE / EXT_SOURCE_2 / target")
    fig = px.box(application,
                 x="OCCUPATION_TYPE",
                 y="EXT_SOURCE_2",
                 color="TARGET",
                 notched=True)
    st.plotly_chart(fig)

    # SHAP
    X1 = dataframe[dataframe['SK_ID_CURR'] == id_input]
    X = X1[[
        'CODE_GENDER', 'AGE', 'CNT_CHILDREN', 'DEF_30_CNT_SOCIAL_CIRCLE',
        'NAME_EDUCATION_TYPE_High education',
        'NAME_EDUCATION_TYPE_Low education',
        'NAME_EDUCATION_TYPE_Medium education',
        'ORGANIZATION_TYPE_Construction', 'ORGANIZATION_TYPE_Electricity',
        'ORGANIZATION_TYPE_Government/Industry', 'ORGANIZATION_TYPE_Medicine',
        'ORGANIZATION_TYPE_Other/Construction/Agriculture',
        'ORGANIZATION_TYPE_School', 'ORGANIZATION_TYPE_Services',
        'ORGANIZATION_TYPE_Trade/Business',
        'OCCUPATION_TYPE_Accountants/HR staff/Managers',
        'OCCUPATION_TYPE_Core/Sales staff', 'OCCUPATION_TYPE_Laborers',
        'OCCUPATION_TYPE_Medicine staff',
        'OCCUPATION_TYPE_Private service staff', 'OCCUPATION_TYPE_Tech Staff',
        'NAME_FAMILY_STATUS_Married', 'NAME_FAMILY_STATUS_Single',
        'AMT_INCOME_TOTAL', 'INCOME_CREDIT_PERC', 'DAYS_EMPLOYED_PERC',
        'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3'
    ]]

    # Variables globales
    st.header('Variables globales du modèle XGBOOST :')
    feat_importances = pd.Series(model.feature_importances_,
                                 index=X.columns).sort_values(ascending=False)
    impPlot(feat_importances, 'XGBOOST Classifier')

    # Variables locales
    st.header('Variables locales du modèle XGBOOST :')
    # compute SHAP values
    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)

    #st_shap(shap.plots.waterfall(shap_values[0]), height=300)
    #st_shap(shap.plots.beeswarm(shap_values), height=300)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    st_shap(shap.summary_plot(shap_values, X, plot_type="bar"))
    #st_shap(shap.summary_plot(shap_values, X))

    st_shap(shap.force_plot(explainer.expected_value, shap_values, X),
            height=200,
            width=1000)


def page3():

    st.sidebar.markdown("# Transparence")
    #st.write ('--- Transparence')

    id_input = st.session_state.client
    st.header("Informations du client")
    st.write("Transparence des informations du client  ", id_input)

    # Pour les informations du client
    examples_file = 'application_API.csv'
    application, liste_id = chargement_data(examples_file)
    application = application[~((application['EXT_SOURCE_1'].isnull()))]
    application.drop(['Unnamed: 0'], axis=1, inplace=True)
    X_infos_client = application[application['SK_ID_CURR'] == id_input]
    st.write(X_infos_client)

    # réalimenter X2 avec les variables saisies
    # Saisie des informations Client dans X2 pour prédiction nouvelle

    X2 = dataframe[dataframe['SK_ID_CURR'] == id_input]

    #AMT_INCOME_TOTAL = st.slider("AMT_INCOME_TOTAL", 1, 500000, 220000)
    #X2['AMT_INCOME_TOTAL'] =  AMT_INCOME_TOTAL

    EXT_SOURCE_1 = st.slider("EXT_SOURCE_1", 0.1, 1.0, 0.1)
    X2['EXT_SOURCE_1'] = EXT_SOURCE_1

    EXT_SOURCE_2 = st.slider("EXT_SOURCE_2", 0.1, 1.0, 0.1)
    X2['EXT_SOURCE_2'] = EXT_SOURCE_2

    EXT_SOURCE_3 = st.slider("EXT_SOURCE_3", 0.1, 1.0, 0.1)
    X2['EXT_SOURCE_3'] = EXT_SOURCE_3

    #NAME_EDUCATION_TYPE = st.selectbox("NAME_EDUCATION_TYPE",options=['Low education','Medium education','High education'])
    #NAME_EDUCATION_TYPE_Low_education , NAME_EDUCATION_TYPE_Medium_education , NAME_EDUCATION_TYPE_High_education = 0,0,0
    #if NAME_EDUCATION_TYPE == 'Low education':
    #     #NAME_EDUCATION_TYPE_Low_education = 1
    #     X2['NAME_EDUCATION_TYPE_Low education'] = 1
    #elif NAME_EDUCATION_TYPE == 'Medium education':
    #     #NAME_EDUCATION_TYPE_Medium_education = 1
    #     X2['NAME_EDUCATION_TYPE_Medium education'] = 1
    #else:
    #     #NAME_EDUCATION_TYPE_High_education = 1
    #     X2['NAME_EDUCATION_TYPE_High education']   = 1

    ORGANIZATION_TYPE = st.selectbox(
        "ORGANIZATION_TYPE", options=['Medicine', 'School', 'Services'])
    #ORGANIZATION_TYPE_Construction, ORGANIZATION_TYPE_Electricity, ORGANIZATION_TYPE_Government_Industry = 0,0,0
    ORGANIZATION_TYPE_Medicine, ORGANIZATION_TYPE_School, ORGANIZATION_TYPE_Services, = 0, 0, 0
    #ORGANIZATION_TYPE_Other_Construction_Agriculture, ORGANIZATION_TYPE_Trade_Business = 0,0
    if ORGANIZATION_TYPE == 'Construction':
        ORGANIZATION_TYPE_Construction = 1
        X2['ORGANIZATION_TYPE_Construction'] = 1
    elif ORGANIZATION_TYPE == 'Electricity':
        ORGANIZATION_TYPE_Electricity = 1
        X2['ORGANIZATION_TYPE_Electricity'] = 1
    elif ORGANIZATION_TYPE == 'Government/Industry':
        ORGANIZATION_TYPE_Government_Industry = 1
        X2['ORGANIZATION_TYPE_Government/Industry'] = 1
    elif ORGANIZATION_TYPE == 'Medicine':
        ORGANIZATION_TYPE_Medicine = 1
        X2['ORGANIZATION_TYPE_Medicine'] = 1
    elif ORGANIZATION_TYPE == 'Other/Construction/Agriculture':
        ORGANIZATION_TYPE_Other_Construction_Agriculture = 1
        X2['ORGANIZATION_TYPE_Other/Construction/Agriculture'] = 1
    elif ORGANIZATION_TYPE == 'School':
        ORGANIZATION_TYPE_School = 1
        X2['ORGANIZATION_TYPE_School'] = 1
    elif ORGANIZATION_TYPE == 'Services':
        ORGANIZATION_TYPE_Services = 1
        X2['ORGANIZATION_TYPE_Services'] = 1
    elif ORGANIZATION_TYPE == 'Trade/Business':
        ORGANIZATION_TYPE_Trade_Business = 1
        X2['ORGANIZATION_TYPE_Trade/Business'] = 1

    OCCUPATION_TYPE = st.selectbox("OCCUPATION_TYPE",
                                   options=[
                                       'Accountants_HR_staff_Managers',
                                       'Private_service_staff',
                                       'Medicine staff'
                                   ])

    OCCUPATION_TYPE_Accountants_HR_staff_Managers, OCCUPATION_TYPE_Private_service_staff, OCCUPATION_TYPE_Medicine_staff = 0, 0, 0
    #OCCUPATION_TYPE_Core_Sales_staff, OCCUPATION_TYPE_Laborers = 0,0,0
    #OCCUPATION_TYPE_Medicine_staff, OCCUPATION_TYPE_Private_service_staff, OCCUPATION_TYPE_Tech_Staff = 0,0,0
    if OCCUPATION_TYPE == 'Accountants/HR staff/Managers':
        OCCUPATION_TYPE_Accountants_HR_staff_Managers = 1
        X2['OCCUPATION_TYPE_Accountants/HR staff/Managers'] = 1
    elif OCCUPATION_TYPE == 'Core/Sales staff':
        OCCUPATION_TYPE_Core_Sales_staff = 1
        X2['OCCUPATION_TYPE_Core/Sales staff'] = 1
    elif OCCUPATION_TYPE == 'Laborers':
        OCCUPATION_TYPE_Laborers = 1
        X2['OCCUPATION_TYPE_Laborers'] = 1
    elif OCCUPATION_TYPE == 'Medicine staff':
        OCCUPATION_TYPE_Medicine_staff = 1
        X2['OCCUPATION_TYPE_Medicine staff'] = 1
    elif OCCUPATION_TYPE == 'Private service staff':
        OCCUPATION_TYPE_Private_service_staff = 1
        X2['OCCUPATION_TYPE_Private service staff'] = 1
    elif OCCUPATION_TYPE == 'Tech Staff':
        OCCUPATION_TYPE_Tech_Staff = 1
        X2['OCCUPATION_TYPE_Tech Staff'] = 1

    X3 = X2[[
        'CODE_GENDER', 'AGE', 'CNT_CHILDREN', 'DEF_30_CNT_SOCIAL_CIRCLE',
        'NAME_EDUCATION_TYPE_High education',
        'NAME_EDUCATION_TYPE_Low education',
        'NAME_EDUCATION_TYPE_Medium education',
        'ORGANIZATION_TYPE_Construction', 'ORGANIZATION_TYPE_Electricity',
        'ORGANIZATION_TYPE_Government/Industry', 'ORGANIZATION_TYPE_Medicine',
        'ORGANIZATION_TYPE_Other/Construction/Agriculture',
        'ORGANIZATION_TYPE_School', 'ORGANIZATION_TYPE_Services',
        'ORGANIZATION_TYPE_Trade/Business',
        'OCCUPATION_TYPE_Accountants/HR staff/Managers',
        'OCCUPATION_TYPE_Core/Sales staff', 'OCCUPATION_TYPE_Laborers',
        'OCCUPATION_TYPE_Medicine staff',
        'OCCUPATION_TYPE_Private service staff', 'OCCUPATION_TYPE_Tech Staff',
        'NAME_FAMILY_STATUS_Married', 'NAME_FAMILY_STATUS_Single',
        'AMT_INCOME_TOTAL', 'INCOME_CREDIT_PERC', 'DAYS_EMPLOYED_PERC',
        'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3'
    ]]

    transparence = prediction(X3)
    #st.write('---debug prediction ', transparence)

    if transparence == 1:
        pred = 'Rejected'
    else:
        pred = 'Approved'

    st.success('Your loan is {}'.format(pred))

    predict_probability = model.predict_proba(X3)
    st.write('Probabilité d"appartenance aux classes : ', predict_probability)

    st.subheader(
        'Le client {} a une probabilité de non remboursement de {}%'.format(
            id_input, round(predict_probability[0][1] * 100, 2)))


my_dict = {
    "Calcul du risque": main_page,
    "Interprétabilité": page2,
    "Transparence": page3,
}

keys = list(my_dict.keys())

selected_page = st.sidebar.selectbox("Select a page", keys)
my_dict[selected_page]()
