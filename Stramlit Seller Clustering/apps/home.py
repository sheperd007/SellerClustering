# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 18:35:26 2021

@author: Hamid.Jahani
"""

  
import streamlit as st

def app():
    st.title('Home')

    st.write('This is the `home page` of this multi-page app.')
    st.markdown("""
## Seller Clustering
In this page, we intend to use the information of sellers from the time they entered system to solve the problem of clustering them and based on these clusters make appropriate decisions in other problems and areas.

the manual Page for `Seller Clustering` : [Seller Clustering]""")

    st.markdown("""
## Seller Churn Clustering
We are going to select some specific sellers and cluster them.
This means that according to the churn model of the sellers, we first select a model execution time and two upper and lower boundaries for the probability of being churned, and then we cluster the sellers that have these features.

the manual Page for `Seller Clustering` : [Seller Clustering]""")

