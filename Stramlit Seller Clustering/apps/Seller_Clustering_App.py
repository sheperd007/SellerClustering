# -*- coding: utf-8 -*-
"""
Created on Sat Jun 26 12:09:54 2021

@author: Hamid.Jahani
"""

######################
# Import libraries
######################

import pandas as pd
import seaborn as sns
import streamlit as st
import base64
import altair as alt
from PIL import Image
from apps.Clustering_functions import *


######################
# Page Title
######################
def app():
    #image = Image.open('D:\Projects\Streamlit\Seller Clustering\\apps\Clustering.jpg')
    
    #st.image(image, use_column_width=True)
    
    st.write("""
    # Seller Clustering
    
    We try to cluster DigiKala sellers according to the available features for Them and respond to the needs of the Commercial team and the business intelligence team.
    
    ***
    """)

    ######################
    #Functions
    ######################
    
    def filedownload(df):
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
        href = f'<a href="data:file/csv;base64,{b64}" download="playerstats.csv">Download CSV File</a>'
        return href
    @st.cache
    def outlier_Detector(Data,Eps,Min_samples,col_remove):
        outlier_detection = DBSCAN(min_samples = Min_samples, eps = Eps)
        clusters_DB = outlier_detection.fit_predict(Data.drop(col_remove,axis=1))
        #list(clusters_DB).count(-1)
        Out=pd.DataFrame({'cluster':clusters_DB}).set_index(Data.drop(col_remove,axis=1).index)
        return Out[Out.cluster==-1].index
    
    @st.cache
    def Clustering_Kmeans_Test_K(Data,n_clusters,col_remove):
        cost=[]
        for i in range(1,n_clusters):
            kmean= KMeans(i)
            kmean.fit(Data.drop(col_remove,axis=1).dropna())
            cost.append(kmean.inertia_)     
        return cost
    @st.cache
    def Clustering_KMedoids_Test_K(Data,n_clusters,col_remove):
        cost=[]
        for i in range(1,n_clusters):
            kmediod= KMedoids(metric="euclidean", n_clusters=i,  
                                init='k-medoids++')
            kmediod.fit(Data.drop(col_remove,axis=1).dropna())
            cost.append(kmediod.inertia_) 
        return cost
    ######################
    # Input Text Box
    ######################
    
    @st.cache
    def load(Level="Category Level 1"):
        if Level=="Category Level 1":
            Data=get_data_set('seller_clustering')
        if Level== "Category_Group (Vertical)" :
            Data=get_data_set('seller_clustering_vertical')
        #Data=Data.drop(["Main_Cluster","sub1"],axis=1)
        Data=Data.set_index('PK',drop=True)
        return Data
    
    st.sidebar.header("Options")
    granularity_level=st.sidebar.selectbox("granularity level", ["Category Level 1","Category_Group (Vertical)"])
    
    Data=load(Level=granularity_level)
    
    st.header('Structure Of Data')
    st.write(Data.head())
    st.header('Enter {}'.format(granularity_level))
    
    if granularity_level=="Category Level 1":
        sequence_input = 6979
    if granularity_level== "Category_Group (Vertical)":
        sequence_input = 40
        
    #sequence = st.sidebar.text_area("Sequence input", sequence_input, height=250)
    sequence = int(st.text_area("Group ID", sequence_input))
    
    st.write("""
    ***
    """)
    
    
    #----------------------------------------------------------------------------
    #Preprocess Data
    if granularity_level=="Category Level 1":
        col_remove=['removable columns']
        
    if granularity_level== "Category_Group (Vertical)":
         col_remove=['removable columns']
    log_col=['columns we want to logharithm']
    @st.cache
    def Process(Data,col_remove,sequence,log_col,Gran_level):
        return Preprocess_category_Data(Data,col_remove,sequence,log_col,Level=Gran_level)
    
    X=Process(Data,col_remove,sequence,log_col,Gran_level=granularity_level)  
    Categoriy_Data=X['Data']
    
    features=list(Categoriy_Data.columns)
    for d in col_remove:
        if d in features: features.remove(d)
    
    
    selected_features = st.sidebar.multiselect('Selected Features for Clustering', list(features), list(features))
    
    for i in ['columns we want to logharithm']:
        if i not in selected_features: log_col.remove(i)
    
    Categoriy_Data=Categoriy_Data[selected_features+col_remove]
    
    #if st.button('Data'):
     #   st.header('Sellected Data')
     #   st.write(Data[Data.CategoryLevel1==sequence][:][selected_features+col_remove].head())
    
    if st.button('Scaled Data'):
        st.header('Normalized Data')
        st.write(Categoriy_Data.head())
    
        st.write("""
             ***
             """)
    
    
    st.header('Outlier Detection')
    Outlier = st.selectbox('Do you need outlier analysis:', ["Yes","No"])
    if Outlier=="Yes":
        Eps = 3
        Eps_val = int(st.text_area("Please Insert Appropriate Eps for DBScan", Eps))
    
        Min_samples = 50
        Min_samples_val = int(st.text_area("Please Insert Appropriate Min_samples for DBScan", Min_samples))
    
        pk_of_potential_outliers=outlier_Detector(Categoriy_Data.dropna(), Eps =Eps_val,Min_samples = Min_samples_val,col_remove=col_remove)
        potential_outliers=Categoriy_Data.loc[pk_of_potential_outliers]
        #Unscaling for Outliers
        for i in potential_outliers.columns.drop(col_remove):
            potential_outliers[i]=(potential_outliers[i]* X["Std_for_unscaling"][i][0])+X["Mean_for_unscaling"][i][0]
        for i in log_col:
            potential_outliers[i][potential_outliers[i]>0]=np.exp(potential_outliers[i][potential_outliers[i]>0])
        if "NMV_3" in list(potential_outliers.columns):
            potential_outliers=potential_outliers.loc[potential_outliers[potential_outliers.NMV_3!=0].index]
        st.subheader('Download SellerIDs that are Outliers')
        st.markdown(filedownload(potential_outliers["SellerID"]), unsafe_allow_html=True)
    
        st.subheader("descriptive statistics about Outliers")
        st.write(potential_outliers.drop(col_remove,axis=1).describe())
    
        st.write("There are **{}** Outlier for This Category With the settings of the previous section  ".format(potential_outliers.shape[0]))
        Outlier = st.selectbox('probability distribution function',list(potential_outliers.drop(col_remove,axis=1).columns))
        sns.distplot(potential_outliers[Outlier])   
        st.pyplot(clear_figure=True)
        
        Outlier_Cluster = st.selectbox('Do you need outlier Clustering:', ["No","Yes"],key="A")
        if Outlier_Cluster=="Yes":
            st.subheader("Outlier Clustering")
            cost=Clustering_KMedoids_Test_K(Categoriy_Data.loc[potential_outliers.index],20,col_remove)
            fig, ax = plt.subplots()
            ax.plot(cost, 'bx-')
            st.pyplot(fig,clear_figure=True)
    
            Number_of_Cluster_for_outliers = st.selectbox('Number Of Cluster:', list(range(1,60)))
    
            Random_s =10
            Random_State = int(st.text_area("Please Insert Appropriate Seed for initializing Centroids", Random_s,key="AA"))
            clusters=Clustering_KMedoids(Categoriy_Data.loc[potential_outliers.index],60
                                 ,col_remove,X['Mean_for_unscaling'],X['Std_for_unscaling']
                                 ,log_col,test_K=False,fixed_size=Number_of_Cluster_for_outliers,Random_State=Random_State)
    
    
            centroid=clusters[1]
            clusters=clusters[0]
            st.subheader('Download SellerIDs')
            st.markdown(filedownload(clusters[["SellerID","cluster"]]), unsafe_allow_html=True)
            st.write(centroid)
    
            st.markdown(filedownload(centroid), unsafe_allow_html=True)
        Categoriy_Data=Categoriy_Data.loc[list(set(Categoriy_Data.index) - set(potential_outliers.index))]
    else:
        Categoriy_Data=X['Data']
        Categoriy_Data=Categoriy_Data[selected_features+col_remove]
        
    st.header("Clustering Data")
    Cluster_1 = st.selectbox('Please select your desired algorithm.', ["Kmeans","Kmediod"])
    
    Test_Number_of_Clusters = st.selectbox('Show Loss Function For Choosing Number of Clusters.', ["No","Yes"],key="B")
    if Cluster_1=="Kmediod":
        #st.write(Categoriy_Data)
        if Test_Number_of_Clusters=="Yes":
            cost=Clustering_KMedoids_Test_K(Categoriy_Data,20,col_remove)
            fig, ax = plt.subplots()
            ax.plot(cost, 'bx-')
            st.pyplot(fig,clear_figure=True)
    
        Number_of_Cluster_Main = st.selectbox('Number Of Main Cluster:', list(range(1,50)))
    
        Random_state = 10
        Random_State_KMedoids = int(st.text_area("Please Insert Appropriate Seed for initializing Centroids", Random_state,key="BB"))
        clusters=Clustering_KMedoids(Categoriy_Data,60
                                 ,col_remove,X['Mean_for_unscaling'],X['Std_for_unscaling']
                                 ,log_col,test_K=False,fixed_size=Number_of_Cluster_Main,Random_State=Random_State_KMedoids)
        centroid=clusters[1]
        clusters=clusters[0]
        st.subheader('Download SellerIDs')
        st.markdown(filedownload(clusters[["SellerID","cluster"]]), unsafe_allow_html=True)
        st.write(centroid)
    
        st.markdown(filedownload(centroid), unsafe_allow_html=True)
    
        Main_Cluster_Density = st.selectbox('Select appropriate feature for Drawing Boxplot',selected_features,key="C")
    
        grid= sns.FacetGrid(clusters , col='cluster')
        grid.map(sns.boxplot, Main_Cluster_Density,showfliers = False)
        st.pyplot(clear_figure=True)
    if Cluster_1=="Kmeans":
        #st.write(Categoriy_Data)
        if Test_Number_of_Clusters=="Yes":
            cost=Clustering_Kmeans_Test_K(Categoriy_Data,20,col_remove)
            fig, ax = plt.subplots()
            ax.plot(cost, 'bx-')
            st.pyplot(fig,clear_figure=True)
    
        Number_of_Cluster_Main = st.selectbox('Number Of Main Cluster:', list(range(1,50)))
    
        Random_state_2 = 10
        Random_State_Kmeans = int(st.text_area("Please Insert Appropriate Seed for initializing Centroids", Random_state_2,key="CC"))
        clusters=Clustering_Kmeans(Categoriy_Data,60
                                 ,col_remove,X['Mean_for_unscaling'],X['Std_for_unscaling']
                                 ,log_col,test_K=False,fixed_size=Number_of_Cluster_Main,Random_State=Random_State_Kmeans)
        centroid=clusters[1]
        clusters=clusters[0]
        st.subheader('Download SellerIDs')
        st.markdown(filedownload(clusters[["SellerID","cluster"]]), unsafe_allow_html=True)
        st.write(centroid)
    
        st.markdown(filedownload(centroid), unsafe_allow_html=True)
        Main_Cluster_Density = st.selectbox('Select appropriate feature for Drawing Boxplot',selected_features,key="D")
        grid= sns.FacetGrid(clusters, col='cluster')
        grid.map(sns.boxplot, Main_Cluster_Density,showfliers = False)
        st.pyplot(clear_figure=True)
        
    st.write("""
    ***""")
    st.header("Building SubCluster") 
    st.write("""
In this section, we intend to re-cluster the clusters of the previous section, which are our main clusters. In other words, we intend to extract a number of sub-clusters for some of the clusters of the previous step. How this part works is that the user first selects the cluster he wants to break and re-cluster it, and then performs the same clustering as the previous part.
   
 ***
    """)
    
    Cluster_2 = st.selectbox('Please select your desired algorithm.', ["Kmeans","Kmediod"],key="E")
    
    Test_Number_of_Clusters_sub = st.selectbox('Show Loss Function For Choosing Number of Clusters.', ["No","Yes"],key="F")
    if Cluster_2=="Kmeans":
        main_number = 0
    
    #sequence = st.bar.text_area("Sequence input", sequence_input, height=250)
        main_number_Clsuter = int(st.text_area("Relevant Cluster number", main_number))
        number_of_Sub_Cluster = st.selectbox('Number Of Sub Cluster:', list(range(1,50)))
        sub_Cluster=Categoriy_Data.loc[clusters[clusters.cluster==main_number_Clsuter].index]
        if Test_Number_of_Clusters_sub=="Yes":
            cost=Clustering_Kmeans_Test_K(sub_Cluster,20,col_remove)
            fig, ax = plt.subplots()
            ax.plot(cost, 'bx-')
            st.pyplot(fig,clear_figure=True)
        clusters_sub=Clustering_Kmeans(sub_Cluster,60,col_remove,X['Mean_for_unscaling'],X['Std_for_unscaling'],log_col,test_K=False,fixed_size=number_of_Sub_Cluster)
        centroid_sub=clusters_sub[1]
        clusters_sub=clusters_sub[0]
        st.subheader('Download SellerIDs')
        st.markdown(filedownload(clusters_sub[["SellerID","cluster"]]), unsafe_allow_html=True)
        st.write(centroid_sub)
        
        st.markdown(filedownload(centroid_sub), unsafe_allow_html=True)
        Main_Cluster_Density = st.selectbox('Select appropriate feature for Drawing Boxplot',selected_features,key="G")
        grid= sns.FacetGrid(clusters_sub, col='cluster')
        grid.map(sns.boxplot, Main_Cluster_Density,showfliers = False)
        st.pyplot(clear_figure=True)
    if Cluster_2=="Kmediod":
        main_number = 0
    
    #sequence = st.bar.text_area("Sequence input", sequence_input, height=250)
        main_number_Clsuter = int(st.text_area("Relevant Cluster number", main_number))
        number_of_Sub_Cluster = st.selectbox('Number Of Sub Cluster:', list(range(1,50)))
    
        sub_Cluster=Categoriy_Data.loc[clusters[clusters.cluster==main_number_Clsuter].index]
        if Test_Number_of_Clusters_sub=="Yes":
            cost=Clustering_KMedoids_Test_K(sub_Cluster,20,col_remove)
            fig, ax = plt.subplots()
            ax.plot(cost, 'bx-')
            st.pyplot(fig,clear_figure=True)
        clusters_sub=Clustering_KMedoids(sub_Cluster,60,col_remove,X['Mean_for_unscaling'],X['Std_for_unscaling'],log_col,test_K=False,fixed_size=number_of_Sub_Cluster)
        centroid_sub=clusters_sub[1]
        clusters_sub=clusters_sub[0]
        st.subheader('Download SellerIDs')
        st.markdown(filedownload(clusters_sub[["SellerID","cluster"]]), unsafe_allow_html=True)
        st.write(centroid_sub)
        
        st.markdown(filedownload(centroid_sub), unsafe_allow_html=True)
        Main_Cluster_Density = st.selectbox('Select appropriate feature for Drawing Boxplot',list(clusters_sub.drop(col_remove,axis=1).columns),key="H")
        grid= sns.FacetGrid(clusters_sub, col='cluster')
        grid.map(sns.boxplot, Main_Cluster_Density,showfliers = False)
        st.pyplot(clear_figure=True)        