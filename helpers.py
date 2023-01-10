#!/usr/bin/env python
# coding: utf-8

# In[3]:


from urllib.request import urlopen
from io import BytesIO
from zipfile import ZipFile
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA , KernelPCA
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import nn
import torch
import torch.optim as optim
import networkx as nx


# In[4]:


def interactions_as_embeddings(df):
    """
    Transforms user-product interactions in a DataFrame into product embeddings.
    
    Args:
        df: a Pandas DataFrame containing user-product interactions, with columns "sku" and "user_id".
        
    Returns:
        A DataFrame containing product embeddings, with columns "sku" and "embedding".
    """
    table = pd.pivot_table(df, values='type', index=["sku"],columns=['user_id'])
    table = table.fillna(0)
    sparsity = sum((table == 0).astype(int).sum())/table.size
    table["embedding"]=table.apply(lambda x: x.values,axis=1)
    product_embeddings = pd.DataFrame(table.to_records())[["sku","embedding"]]
    return product_embeddings


# In[5]:


def map_indices_to_skus(index_series,index_dict):
    """
    Maps a series of indices to a series of SKUs using a provided mapping dictionary.
    
    Parameters:
        index_series (list): A series of indices to map to SKUs.
        index_dict (dict): A dictionary that maps indices to SKUs.
        
    Returns:
        List: A series of SKUs corresponding to the provided indices.
    """
    sku_series = []
    for i in index_series:
        sku_series.append(index_dict[i])
    return sku_series


# In[6]:


def generate_recommendations(indices,embeddings,index_dict):
    """
    Generates product recommendations based on nearest neighbors of a given set of embeddings.
    
    Parameters:
        indices (pandas.DataFrame): A DataFrame of nearest neighbor indices for the embeddings.
        embeddings (pandas.DataFrame): A DataFrame of the embeddings.
        index_dict (dict): A dictionary mapping indices to product SKUs.
        
    Returns:
        pandas.DataFrame: A DataFrame of the recommended product SKUs.
    """
    recommendations = pd.DataFrame(indices).reset_index()
    recommendation_df = embeddings.reset_index().merge(recommendations,how="left",on = "index")
    recommendation_df=recommendation_df[["sku",1,2,3,4,5,6,7,8,9,10]]
    recommendations_skus = recommendation_df.drop("sku",axis=1).apply(lambda x: map_indices_to_skus(x,index_dict[0]),axis=0)
    final_recommendatons = embeddings.reset_index().merge(recommendations_skus.reset_index(),how="left",on = "index")[["sku",1,2,3,4,5,6,7,8,9,10]]
    return final_recommendatons


# In[ ]:


def transform_embeddings_with_kernel_pca(embeddings,n_components = 25,kernel = 'sigmoid'):
    """
    Transforms a series of embeddings using Kernel PCA.

    Parameters:
        embeddings (pandas.DataFrame): A DataFrame containing the embeddings to transform.
        n_components (int): The number of components to keep after transformation.
        kernel (str): The kernel to use for the transformation.

    Returns:
        pandas.DataFrame: A DataFrame containing the transformed embeddings.
    """
    transformer = KernelPCA(n_components = n_components, kernel = kernel)
    X_transformed = transformer.fit_transform(embeddings.embedding.tolist())
    df1 = pd.DataFrame()
    df1["embedding"]=X_transformed.tolist()
    df1.reset_index(inplace=True)
    df1 = df1.apply(np.array)
    embeddings = embeddings.reset_index().merge(df1,how="left",on = "index")[["sku","embedding_y"]]
    pca_embeddings = embeddings.rename(columns = {"embedding_y":"embedding"})
    return pca_embeddings


# In[ ]:


def generate_ann_embeddings(model,dataset,sku):
    """
    Returns the embedding for a product in a trained model.
    
    Parameters:
        model (EmbeddingLearningModel): The trained model.
        dataset (InteractionDataset): The dataset the model was trained on.
        sku (str): The SKU of the product to get the embedding for.
        
    Returns:
        numpy.ndarray: The embedding for the product.
    """

    sku = dataset.vocab[dataset.vocab["sku"]==sku].index[0]
    embeddings = model.sku_embedding_layer(torch.tensor(sku)).cpu().detach().numpy()
    return embeddings


# In[ ]:


def get_ann_embeddings(model,df,dataset):
    """
    Generates embeddings for products using an ANN model.
    
    Parameters:
        model (EmbeddingLearningModel): The ANN model.
        df (pandas.DataFrame): A DataFrame of product data.
        dataset (InreractionDataset): The dataset the model was trained on.
        
    Returns:
        pandas.DataFrame: A DataFrame of product SKUs and their corresponding embeddings.
    """

    product_sku = pd.DataFrame(list(set(df.sku))).rename(columns={0:"sku"})
    product_sku["embedding"] = product_sku.sku.apply(lambda x:generate_ann_embeddings(model,dataset,x))
    return product_sku


# In[ ]:


def create_graph(df):
    """
    Creates a graph from a DataFrame of user-product interactions.
    
    Parameters:
        df (pandas.DataFrame): A DataFrame of user-product interactions, with columns "user_id", "sku", and "type".
        
    Returns:
        networkx.Graph: A graph of the user-product interactions.
    """

    return nx.convert_matrix.from_pandas_edgelist(df,'user_id','sku',edge_attr="type", create_using=nx.Graph())


# In[ ]:


def graph_relations(source,G_users):
    """
    Generates a list of recommended products based on the relations of a given user in a user-product interaction graph.
    
    Parameters:
        source (str): The product SKU to get recommendations for.
        G_users (networkx.Graph): The user-product interaction graph.
        
    Returns:
        list: A list of recommended product SKUs.
    """
    commons_dict = {}
    commons_dict = {}
    for e in G_users.neighbors(source):
        for e2 in G_users.edges(e):
            if e2==source:
                continue
            product = e2[1]
            weight = G_users[e2[0]][e2[1]]["type"]
            if product not in commons_dict.keys():
                commons_dict[product] = weight
            else:
                commons_dict[product] += weight
    return sorted(commons_dict, key=commons_dict.get, reverse=True)[:10]

