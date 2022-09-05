#!/usr/bin/env python
# coding: utf-8

# In[1]:



import warnings

warnings.filterwarnings('ignore')

"""لود کردن توابع موردنیاز برای انجام تحلیل"""
from Clustering_functions import *

# Reading from Server
# conncet_to_server()
# فراخوانی داده ها
Data = get_data_set('seller_clustering')
# Data=Data.drop(["Main_Cluster","sub1"],axis=1)
#مشخص کردن Index اصلی مجموعه داده، توجه کنید که تحلیل و بسیاری از توابع به این متغیر PK وابسته هستند.
Data = Data.set_index('PK', drop=True)
#Data.CategoryLevel1=Data.CategoryLevel1.astype("category")
#Data.Bucket=Data.Bucket.astype("category")
#Data.Category_Label=Data.Category_Label.astype("category")


# In[8]:
# col_remove : Columns represent the identity of the seller
# log_col : Columns That Must Take Log from Them
# Center : Gathering Centers Of Cluster for importing Them inside Datawarehouse
#LABELS : Ghathering Clusters LABELS
Center = list()
"""ساختن یک pandas dataframe برای ذخیره سازی labelهای تولید شده """
LABELS = pd.DataFrame(index=Data.index, columns=["Main_Cluster", "sub1"])

col_remove = ['put useless column in here']
log_col = ['NMV_3', 'GrossMargin', 'NMV_3_SBS', 'GrossMargin_SBS']
""" حال در ادامه کدی نوشته می‌شود که برروی تمام کتگوری ها جدا جدا اجرا شود و سه خوشه برروی هرکدام می‌سازد سپس هرخوشه را اگر بزرگتر از 1500 نفر عضو داشته باشد می‌شکند و سه خوشه دیگر ایجاد می‌کند .
البته توجه کنید این ساختار موقت است و تعداد خوشه مناسب برای هر کتگوری به همراه دیگر پارامترها باید از طریق نسخه داینامیک مدل خوشه‌بندی استخراج شود،
سپس مقادیر مناسب باید به این کد نسخه اتوماتیک تزریق شود تا نسخه بهینه به صورت کاملا اتوماتیک تولید شود. """
for Cat in np.unique(Data[["CategoryLevel1"]]):
    #Procesing Data before fitting model
    X = Preprocess_category_Data(Data, col_remove, Cat, log_col)
    Categoriy_Data = X['Data']
    
    
    # finding Outliers with DBSCAN and Returning PK of it for seperating them from data
    pk_of_potential_outliers = outlier_Detector(Categoriy_Data, Eps=3, Min_samples=50, col_remove=col_remove)
    potential_outliers = Categoriy_Data.loc[pk_of_potential_outliers]
    
    
    for i in pk_of_potential_outliers:
        #Labeling Outlier as -1 (Cluster Number)
        LABELS.loc[i]["Main_Cluster"] = -1

    # Unscaling Outliers
    for i in potential_outliers.columns.drop(col_remove):
        potential_outliers[i] = (potential_outliers[i] * X["Std_for_unscaling"][i][0]) + X["Mean_for_unscaling"][i][0]
    for i in log_col:
        potential_outliers[i][potential_outliers[i] > 0] = np.exp(potential_outliers[i][potential_outliers[i] > 0])

    #Some outliers Are Absolute Dead without any GMV and NMV . we back them to data
    potential_outliers = potential_outliers.loc[potential_outliers[potential_outliers.NMV_3 != 0].index]
    #Checking Outlier Data Shapes , if data is Empty (potential_outliers.shape[0] = 0) then we will encounter with error
    #Building SubClusters for Cluster -1
    if potential_outliers.shape[0] != 0:
        clusters = Clustering_KMedoids(Categoriy_Data.loc[potential_outliers.index], 60, col_remove,
                                   X['Mean_for_unscaling'], X['Std_for_unscaling'], log_col, test_K=False, fixed_size=3)
        #Centroid of Clusters
        centroid = clusters[1]
        #Data with it's Coresponding Labels
        clusters = clusters[0]
        centroid.rename(columns={'cluster': 'sub1'}, inplace=True)
        
        
        #prepearing centers for Adding Subcluster to them
        cen = pd.DataFrame({"CategoryLevel1": [Cat] * 3, "Main_Cluster": [-1] * 3}).join(centroid)
        Center.append(cen)
        for i in clusters.index:
            #adding Labels to LABELS DataFrame
            LABELS.loc[i]["sub{}".format(str(1))] = clusters.loc[i]["cluster"]
            
     #Removing Outliers from Primary Category_Data       
    Categoriy_Data = Categoriy_Data.loc[list(set(Categoriy_Data.index) - set(potential_outliers.index))]
    
    
    #Let's go for Remaining Data (Cluster Remaining Data after Outlier Detection)
    if Categoriy_Data.shape[0] != 0:
        cluster_kmediod = Clustering_KMedoids(Categoriy_Data, 40, col_remove, X['Mean_for_unscaling'],
                                          X['Std_for_unscaling'], log_col, test_K=False, fixed_size=3)
        centroid_kmediod = cluster_kmediod[1]
        centroid_kmediod_copy = centroid_kmediod.rename(columns={'cluster': 'Main_Cluster'})
        cen = pd.DataFrame({"CategoryLevel1": [Cat] * 3}).join(centroid_kmediod_copy)
        Center.append(cen)
        cluster_kmediod = cluster_kmediod[0]
        
        
        #Adding Labels to LABEL DataFrame After Clustreing
        for i in cluster_kmediod.index:
            LABELS.loc[i]["Main_Cluster"] = cluster_kmediod.loc[i]["cluster"]
            
        #Building Sub Clusters for every Clusters that extracted from Previous Code  
        for Sub in range(3):
            
            
            #if a Clusters have 1500 member or More it will break to some Sub_clusters. detail of parameter should
            if int(centroid_kmediod[centroid_kmediod.cluster == Sub].Count) > 1500:
                # ATTENTION : Number Of Cluster is Not Fix
                #Separate data from one Sepecific cluster
                sub_Cluster = Categoriy_Data.loc[cluster_kmediod[cluster_kmediod.cluster == Sub].index]
                #Bulding Sub_clusters
                clusters_sub = Clustering_Kmeans(sub_Cluster, 60, col_remove, X['Mean_for_unscaling'],
                                             X['Std_for_unscaling'], log_col, test_K=False, fixed_size=3)
                centroid_sub = clusters_sub[1]
                clusters_sub = clusters_sub[0]
                centroid_sub.rename(columns={'cluster': 'sub1'}, inplace=True)
                cen = pd.DataFrame({"CategoryLevel1": [Cat] * 3, "Main_Cluster": [Sub] * 3}).join(centroid_sub)
                Center.append(cen)
                for i in clusters_sub.index:
                    LABELS.loc[i]["sub{}".format(str(1))] = clusters_sub.loc[i]["cluster"]

# In[19]:


LABELS = LABELS.reset_index()

# In[15]:


Center = pd.concat(Center)

LABELS.Main_Cluster = LABELS.Main_Cluster.astype(float)
LABELS.sub1 = LABELS.sub1.astype(float)

l=["Final features"]
for i in l:
    Center[l]=Center[l].round()
# In[17]:
LABELS["run_date"]=pd.Series(np.repeat(pd.date_range('today', periods=1, freq='D').normalize(),np.max(LABELS.index)+1))
Center["run_date"]=pd.Series(np.repeat(pd.date_range('today', periods=1, freq='D').normalize(),np.max(Center.index)+1))

df_to_db(LABELS, server="", database="", Schema="",
         table="", strategy_if_exist='replace')
df_to_db(Center, server="", database="", Schema="",
         table="", strategy_if_exist='replace')
