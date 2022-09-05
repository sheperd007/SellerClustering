# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 11:11:43 2021

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

def Preprocess_category_Data(Data,col_remove,box=False):
        '''
        
    
        Parameters
        ----------
        Data : DataFrame
             Clustering Data
        col_remove : List of Strings (Columns Name from Data)
             Columns that must be Remove
        number : Category Level 1 
             Category level 1 that must be separated from the data and analyzed
        log_col :  List of Strings (Columns Name from Data)
             Columns that must be logarithmized before the preprocessing operation (usually columns with a currency)
        box: boolean
            Can a box plot of all variables be drawn or not?
        
        about Structure that is in Functions:
            First we examine whether a column has only one value or not.
            If it has only one value then it should be deleted because then the clustering algorithm used will encounter an error.
            In the next step, we take the logarithm from the variables specified in log_col.
            In the next step, the average of the variables is subtracted from them and then divided by the standard deviation until the variables are standardized.
            
        Raises
        ------
        Exception
             if there is Duplicate Seller in Category level 1 then function will rise a Exception
        Returns
        -------
        dict
             [Preprocessed , Mean of Scaling columns for Unscaling,Std of Scaling columns for Unscaling]
        '''
        l=list()
        for i in Data.columns.drop(col_remove):
            if len(np.unique(Data[i].dropna()))==1:
                l.append(i)
        Data=Data.drop(l,axis=1)
        # Scaling
        col=Data.columns.drop(col_remove)
        mean=pd.DataFrame(Data[col].mean()).transpose()
        std=pd.DataFrame(Data[col].std()).transpose()
        for i in col:
            if Data[i].std()!=0:
                Data[i]=(Data[i]- Data[i].mean())/ Data[i].std()
        if(box==True):
            plt.figure(dpi=1200)
            Data.drop(col_remove,axis=1).boxplot()
            plt.xticks(rotation='vertical')
            plt.show()
        return {'Data':Data,'Mean_for_unscaling':mean,'Std_for_unscaling':std}



######################
# Page Title
######################
def app():
#Rewrite PreProcess
    

    #image = Image.open('D:\Projects\Streamlit\Seller Clustering\\apps\Clustering.jpg')
    
    #st.image(image, use_column_width=True)
    
    st.write("""
    # Every thing Clustering
    
    We try to cluster Any Kind of Data.
    
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
    st.sidebar.header("Options")
    with st.form('Uploads'):
        st.write("""Data Need To be in Excel sheet and Less than 100K Rows """)
        uploaded_file = st.file_uploader(f'Choose a file for Clusteirng:', type=['xlsx', 'csv'], key='data')
        upload_submitted = st.form_submit_button('Submit')
        if uploaded_file is None:
            st.write('Waiting...')
            conn=conncet_to_server()
            query = """
                loading a piece of deafult data to show"""
            Data = pd.read_sql_query(query, conn)
            conn.close()
            pass

        elif uploaded_file is not None:
            st.write('Successfully recieved.')
            Data = pd.read_excel(uploaded_file)

    if(Data.shape[0]<=100000):    
        st.header('Structure Of Data')
        st.write(Data.head())
        
        
        #----------------------------------------------------------------------------
        #Preprocess Data
        Key=st.sidebar.selectbox('Primary Key of Data', list(Data.columns))
        col_remove = [column for column in Data.columns if Data[column].dtype == 'object']
        col_remove.append(Key)
        col_remove = st.sidebar.multiselect('Columns identifying data', list(Data.columns),list(col_remove))
        
    
        @st.cache
        def Process(Data,col_remove):
            return Preprocess_category_Data(Data,col_remove)
        
        X=Process(Data,col_remove)  
        Categoriy_Data=X['Data']
        
        features=list(Categoriy_Data.columns)
        for d in col_remove:
            if d in features: features.remove(d)
        
    
        
        
        selected_features = st.sidebar.multiselect('Selected Features for Clustering', list(features), list(features))
        
        
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
            st.subheader('Download SellerIDs that are Outliers')
            st.markdown(filedownload(potential_outliers["%s" %(Key)]), unsafe_allow_html=True)
        
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
                                     ,log_col=None,test_K=False,fixed_size=Number_of_Cluster_for_outliers,Random_State=Random_State)
        
        
                centroid=clusters[1]
                clusters=clusters[0]
                st.subheader('Download %s' %(Key))
                st.markdown(filedownload(clusters[["%s" %(Key) ,"cluster"]]), unsafe_allow_html=True)
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
                                     ,log_col=None,test_K=False,fixed_size=Number_of_Cluster_Main,Random_State=Random_State_KMedoids,Key=Key)
            centroid=clusters[1]
            clusters=clusters[0]
            st.subheader('Download %s' %(Key))
            st.markdown(filedownload(clusters[["%s" %(Key),"cluster"]]), unsafe_allow_html=True)
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
                                     ,log_col=None,test_K=False,fixed_size=Number_of_Cluster_Main,Random_State=Random_State_Kmeans,Key=Key)
            centroid=clusters[1]
            clusters=clusters[0]
            st.subheader('Download %s' %(Key))
            st.markdown(filedownload(clusters[["%s" %(Key),"cluster"]]), unsafe_allow_html=True)
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
            clusters_sub=Clustering_Kmeans(sub_Cluster,60,col_remove,X['Mean_for_unscaling'],X['Std_for_unscaling'],log_col=None,test_K=False,fixed_size=number_of_Sub_Cluster,Key=Key)
            centroid_sub=clusters_sub[1]
            clusters_sub=clusters_sub[0]
            st.subheader('Download %s' %(Key))
            st.markdown(filedownload(clusters_sub[["%s" %(Key),"cluster"]]), unsafe_allow_html=True)
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
            clusters_sub=Clustering_KMedoids(sub_Cluster,60,col_remove,X['Mean_for_unscaling'],X['Std_for_unscaling'],log_col=None,test_K=False,fixed_size=number_of_Sub_Cluster,Key=Key)
            centroid_sub=clusters_sub[1]
            clusters_sub=clusters_sub[0]
            st.subheader('Download SellerIDs')
            st.markdown(filedownload(clusters_sub[["%s" %(Key),"cluster"]]), unsafe_allow_html=True)
            st.write(centroid_sub)
            
            st.markdown(filedownload(centroid_sub), unsafe_allow_html=True)
            Main_Cluster_Density = st.selectbox('Select appropriate feature for Drawing Boxplot',list(clusters_sub.drop(col_remove,axis=1).columns),key="H")
            grid= sns.FacetGrid(clusters_sub, col='cluster')
            grid.map(sns.boxplot, Main_Cluster_Density,showfliers = False)
            st.pyplot(clear_figure=True)        
    else:
        del(Data)
        st.write("""Data Need To be in Excel sheet and Less than 100K Rows """)