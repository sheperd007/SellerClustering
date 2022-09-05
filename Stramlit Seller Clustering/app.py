# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 18:38:31 2021

@author: Hamid.Jahani
"""

import streamlit as st
from multiapp import MultiApp
from apps import home,Seller_Churn_Clustering, Seller_Clustering_App,Every_Thing_Clustering # import your app modules here


st.set_page_config(layout="wide")
#CHECK THIS LATER #################IMPORTANT
st.set_option('deprecation.showPyplotGlobalUse', False)

app = MultiApp()

st.markdown("""
# Seller Clustering App
This is a navigation page for Cluster sellers of Digikala according to some features and Options
""")

# Add all your application here
app.add_app("Home", home.app)
app.add_app("Seller Churn Clustering", Seller_Churn_Clustering.app)
app.add_app("Seller Clustering", Seller_Clustering_App.app)
app.add_app("Every thing Clustering", Every_Thing_Clustering.app)
# The main app
app.run()