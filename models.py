#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA , KernelPCA
from torch import nn
import torch
from helpers import graph_relations,create_graph


# In[ ]:


def find_nearest_neighbors(embeddings):
    """
    Finds the nearest neighbors for a series of embeddings using k-nearest neighbors algorithm.
    
    Parameters:
        embeddings (pandas.DataFrame): A DataFrame containing the embeddings to find neighbors for.
        
    Returns:
        tuple: A tuple containing two NumPy arrays, the first containing distances to the nearest neighbors and the second containing the indices of the nearest neighbors.
    """
    index_dict = embeddings.reset_index()[["index","sku"]].set_index('index').T.to_dict("records")
    nbrs = NearestNeighbors(n_neighbors=11, algorithm='ball_tree').fit(embeddings.embedding.tolist())
    distances, indices = nbrs.kneighbors(embeddings.embedding.tolist())
    return indices, index_dict


# In[ ]:


def graph_recommendations(df):
    """
    Generates product recommendations based on a user-product interaction graph.
    
    Parameters:
        df (pandas.DataFrame): A DataFrame of user-product interactions.
        
    Returns:
        pandas.DataFrame: A DataFrame of product SKUs and their corresponding recommendations.
    """

    G_users = create_graph(df)
    product_sku = pd.DataFrame(list(set(df.sku))).rename(columns={0:"sku"})
    product_sku["embeddings"]=product_sku.sku.apply(lambda x: graph_relations(x,G_users))
    return product_sku


# In[ ]:


class embedding_learning_model(nn.Module):
    """
    A PyTorch model for learning embeddings from user-product pairs.
    """

    def __init__(self,num_user,num_product):
        """
        Initializes the model with the number of users and products.
        
        Parameters:
            num_user (int): The number of users in the dataset.
            num_product (int): The number of products in the dataset.
        """
        super(embedding_learning_model, self).__init__()
        self.user_embedding_layer = nn.Embedding(num_user, 128)
        self.sku_embedding_layer = nn.Embedding(num_product, 128)
        self.linear_relu_stack = nn.Sequential(
                nn.Linear(256, 64),
                nn.ReLU(),
                nn.Linear(64, 3)
            )
    def forward(self,x):
        """
        Performs a forward pass through the model.
        
        Parameters:
            x (tuple): A tuple containing the user ID and product SKU to process.
            
        Returns:
            torch.Tensor: The output class probablities of the model.
        """
        user_id,product_sku = x
        user_embedding = self.user_embedding_layer(user_id)
        sku_embedding = self.sku_embedding_layer(product_sku)
        concat = torch.cat([user_embedding, sku_embedding], dim=0).reshape(1,256)
        x = self.linear_relu_stack(concat)
        return x



# In[ ]:





# In[ ]:





# In[ ]:




