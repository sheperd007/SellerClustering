# -*- coding: utf-8 -*-
"""
Spyder Editor

In this file, I am creating the appropriate functions to clear clustering data,
apply clustering, and then restore the data to its previous state.
All these functions are called in an ipynb file and the operation is performed.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyodbc
import sqlalchemy
from six.moves import urllib
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids


def conncet_to_server(Server='', Database=''):
    """


    Parameters
    ----------
    Server : TYPE, String
        Server That Data is Located . The default is ''.
    Database : TYPE, String
        Data Base That Data is Located. The default is ''.

    Returns a Connection to that Specific Data Base for Reading Data From That
    -------
    None.

    """
    # ---------connect to server----------
    conn = pyodbc.connect('Driver={SQL Server};'
                          'Server=' + Server +
                          ';Database=' + Database +
                          ';Trusted_Connection=yes;'
                          'uid=;'
                          'pwd=')
    #     cursor = conn.cursor()
    return conn


# --------------------------------------------------------------------------
def get_data_set(table_name):
    """


    Parameters
    ----------
    table_name : TYPE
        The table on which the data is located..

    Returns Data That Needed for Analysis
    -------
    dataset : Pandas Data Frame
        Data.

    Note that some Category should not be included in the data being read
    They are Gifts and some Other Cats.
    """
    conn = conncet_to_server(Database='')
    query = """
            select * from  
            'database'.'schema'.""" + table_name
    dataset = pd.read_sql_query(query, conn)
    conn.close()
    return dataset


def insert_data_set(table_name, Data):
    """


    Parameters
    ----------
    table_name : TYPE
        The table on which the data is located..

    Returns Data That Needed for Analysis
    -------
    dataset : Pandas Data Frame
        Data.

    Note that some Category should not be included in the data being read
    They are Gifts and some Other Cats.
    """
    conn = conncet_to_server()
    cursor = conn.cursor()
    for index, row in Data.iterrows():
        cursor.execute('''UPDATE 'database'.'schema'.''' + table_name + '''set Main_Cluster={}, sub1={}
                where PK={}'''.format(row[0], row[1], index))
    conn.commit()
    conn.close()


def Preprocess_category_Data(Data, col_remove, number, log_col, box=False):
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
    Categoriy_Data = Data[Data.CategoryLevel1 == number][:]
    if len((np.unique(Categoriy_Data.SellerID))) != len((Categoriy_Data.SellerID)):
        raise Exception('there is Duplicate SellerID in this Category')
    l = list()
    for i in Categoriy_Data.columns.drop(col_remove):
        if len(np.unique(Categoriy_Data[i].dropna())) == 1:
            l.append(i)
    Categoriy_Data = Categoriy_Data.drop(l, axis=1)
    for d in l:
        if d in log_col: log_col.remove(d)
    for i in log_col:
        Categoriy_Data[i][Categoriy_Data[i] > 0] = np.log(Categoriy_Data[i][Categoriy_Data[i] > 0])
    # Scaling
    col = Categoriy_Data.columns.drop(col_remove)
    mean = pd.DataFrame(Categoriy_Data[col].mean()).transpose()
    std = pd.DataFrame(Categoriy_Data[col].std()).transpose()
    for i in col:
        if Categoriy_Data[i].std() != 0:
            Categoriy_Data[i] = (Categoriy_Data[i] - Categoriy_Data[i].mean()) / Categoriy_Data[i].std()
    if (box == True):
        plt.figure(dpi=1200)
        Categoriy_Data.drop(col_remove, axis=1).boxplot()
        plt.xticks(rotation='vertical')
        plt.show()
    return {'Data': Categoriy_Data, 'Mean_for_unscaling': mean, 'Std_for_unscaling': std}


def outlier_Detector(Data, Eps, Min_samples, col_remove):
    """


    Parameters
    ----------
    Data : pandas Data Frame
        it is Category Level 1 Data that must it's Outlier be found
    Eps : a Hyper patameter That Must be Tuned in
        Two important parameters are required for DBSCAN: epsilon (“eps”) and minimum points (“MinPts”).
        The parameter eps defines the radius of neighborhood around a point x. It's called called the ϵ-neighborhood of x.
        The parameter MinPts is the minimum number of neighbors within “eps” radius.
    Min_samples : Same as Eps
        DESCRIPTION.
    col_remove : List of Strings (Columns Name from Data)
         Columns that must be Remove


    Returns
    -------
    TYPE
        Index of Dataframe To extract lines that are Outlier..

    """
    outlier_detection = DBSCAN(min_samples=Min_samples, eps=Eps)
    clusters_DB = outlier_detection.fit_predict(Data.drop(col_remove, axis=1).dropna())
    # list(clusters_DB).count(-1)
    Out = pd.DataFrame({'cluster': clusters_DB}).set_index(Data.drop(col_remove, axis=1).dropna().index)
    return Out[Out.cluster == -1].index


def Clustering_Kmeans(Data, n_clusters, col_remove, Mean_for_unscaling, Std_for_unscaling, log_col, test_K=True,
                      fixed_size=None, Random_State=None):
    """


    Parameters
    ----------
    Data : pandas Data Frame
        Data That Must be Used for Clustering.
    n_clusters : Int
        Maximum number of clusters to plot the model loss function for Choosing Number Of Cluster for Final Run
    col_remove : Same as other Functions
        DESCRIPTION.
    Mean_for_unscaling : list
        The average required to unscale the data and obtain the data with initial scale.
    Std_for_unscaling : list
        The Standard Deviation to unscale the data and obtain the data with initial scale.
    log_col : list of Columns Name
        same as other Functions.
    test_K : Boolean, optional
        Do You want to plot loss Curve in order to Specify K value. The default is True.
    fixed_size : int, optional
        Do You Know the Value of K? Pass it here. The default is None.
    Random_State= Float, optional
        Random Seed for initializing Clusteroid of Clusters
    -------
    this Function has Some "if" For Testing that Values Are in Correct Shape. Do not take them seriously!
    -------
    Returns
    -------
    list
        Centers of Cluster and Unscale Clusterd Data with it's Cluster Number.

    """
    cost = []
    if (test_K == True):
        for i in range(1, n_clusters):
            kmean = KMeans(i)
            kmean.fit(Data.drop(col_remove, axis=1))
            cost.append(kmean.inertia_)
        plt.figure(dpi=1200)
        plt.plot(cost, 'bx-')
        plt.show()
    l = True  # for testing that number_of_cluster is int or not
    while (l):
        if fixed_size is None:
            number_of_cluster = int(input("Please Insert Number of Clusters :"))
        else:
            number_of_cluster = fixed_size
        if type(number_of_cluster) == int:
            l = False
        else:
            print("Number of Cluster Must be an Integer")
    kmean = KMeans(number_of_cluster, random_state=Random_State)
    kmean.fit(Data.drop(col_remove, axis=1).dropna())
    labels = kmean.labels_
    # Concatination Labels to Data
    clusters = pd.concat([Data.loc[Data.drop(col_remove, axis=1).dropna().index],
                          pd.DataFrame({'cluster': labels}).set_index(Data.drop(col_remove, axis=1).dropna().index)],
                         axis=1)
    # UnScaling
    for i in Data.columns.drop(col_remove):
        clusters[i] = (clusters[i] * Std_for_unscaling[i][0]) + Mean_for_unscaling[i][0]
    for i in log_col:
        clusters[i][clusters[i] > 0] = np.exp(clusters[i][clusters[i] > 0])
    centroid = pd.DataFrame(kmean.cluster_centers_, columns=Data.columns.drop(col_remove))
    for i in centroid.columns:
        centroid[i] = (centroid[i] * Std_for_unscaling[i][0]) + Mean_for_unscaling[i][0]
    for i in log_col:
        centroid[i][centroid[i] > 0] = np.exp(centroid[i][centroid[i] > 0])
    cen = clusters.groupby('cluster').agg({'SellerID': 'count'}).reset_index().join(centroid)
    cen = cen.rename(columns={"SellerID": "Count"})
    return [clusters, cen]


def Clustering_KMedoids(Data, n_clusters, col_remove, Mean_for_unscaling, Std_for_unscaling, log_col, test_K=True,
                        fixed_size=None, Random_State=None):
    """
    Same as Clustering_Kmeans Functions

    """
    cost = []
    if (test_K == True):
        for i in range(1, n_clusters):
            kmediod = KMedoids(metric="euclidean", n_clusters=i,
                               init='random', random_state=Random_State)
            kmediod.fit(Data.drop(col_remove, axis=1))
            cost.append(kmediod.inertia_)
        plt.figure(dpi=1200)
        plt.plot(cost, 'bx-')
        plt.show()
    l = True  # for testing that number_of_cluster is int or not
    while (l):
        if fixed_size is None:
            number_of_cluster = int(input("Please Insert Number of Clusters :"))
        else:
            number_of_cluster = fixed_size
        if type(number_of_cluster) == int:
            l = False
        else:
            print("Number of Cluster Must be an Integer")
    kmediod = KMedoids(metric="euclidean", n_clusters=number_of_cluster,
                       init='k-medoids++')
    kmediod.fit(Data.drop(col_remove, axis=1).dropna())
    labels = kmediod.labels_
    # Concatination Labels to Data
    clusters = pd.concat([Data.loc[Data.drop(col_remove, axis=1).dropna().index],
                          pd.DataFrame({'cluster': labels}).set_index(Data.drop(col_remove, axis=1).dropna().index)],
                         axis=1)
    # UnScaling
    for i in Data.columns.drop(col_remove):
        clusters[i] = (clusters[i] * Std_for_unscaling[i][0]) + Mean_for_unscaling[i][0]
    for i in log_col:
        clusters[i][clusters[i] > 0] = np.exp(clusters[i][clusters[i] > 0])
    centroid = pd.DataFrame(kmediod.cluster_centers_, columns=Data.columns.drop(col_remove))
    for i in centroid.columns:
        centroid[i] = (centroid[i] * Std_for_unscaling[i][0]) + Mean_for_unscaling[i][0]
    for i in log_col:
        centroid[i][centroid[i] > 0] = np.exp(centroid[i][centroid[i] > 0])
    cen = clusters.groupby('cluster').agg({'SellerID': 'count'}).reset_index().join(centroid)
    cen = cen.rename(columns={"SellerID": "Count"})
    return [clusters, cen]


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def df_to_db(df, server, database, Schema, table, strategy_if_exist='fail', chunksize=5):
    """
    Create a table or use available table to write pandas dataframe in database.
    CAUTION: if didn't work reset your session

    df: Pandas DataFrame
    server:server name
    database: database name
    table: table name
    strategy_if_exist: by default return fail if table exist you cand use 'replace' and 'append' too
    chunksize: it's possible but Not available at the moment
    method: it's possible but Not available at the moment (None is  row by row and 'multi')

    """

    db_params = urllib.parse.quote_plus(
        r'DRIVER={SQL Server};SERVER=' + server + ';DATABASE=' + database + ';Trusted_Connection=yes')
    engine = sqlalchemy.create_engine("mssql+pyodbc:///?odbc_connect={}".format(db_params))

    df.to_sql(table, engine, schema=Schema, index=False, if_exists=strategy_if_exist, chunksize=chunksize,
              method='multi')


