#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from urllib.request import urlopen
from io import BytesIO
from zipfile import ZipFile
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import nn
import torch


# In[ ]:


def download_and_unzip(url, extract_to='.'):
    """
    Downloads a ZIP file from the specified URL and extracts its contents to a specified location.
    
    Args:
        url: string, the URL of the ZIP file to download.
        extract_to: string, the path to the directory where the contents of the ZIP file should be extracted.
                    Defaults to the current working directory.
                    
    Returns:
        None
    """
    http_response = urlopen(url)
    zipfile = ZipFile(BytesIO(http_response.read()))
    zipfile.extractall(path=extract_to)


# In[1]:


def prepare_dataset(df):
    '''
    Function for convert given dataset to our objective: Given a pandas DataFrame, 
    returns converted pandas DataFrame.
    Args:
        df: a Pandas DataFrame containing the dataset.
    
    Returns:
        The modified DataFrame.
    '''
    subtitude = {0.5:0,
             1.0:0,
             2.0:0,
             3.0:1,
             3.5:1,
             5.0:2,
             4.0:2,
             4.5:2,
             2.5:1,
             1.5:0
             }
    df = df.rename({"userId":"user_id","movieId":"sku","rating":"type"},axis=1)
    df = df.drop("timestamp",axis=1)
    df["type"] = df["type"].map(subtitude)
    df.sku = "MYO-" + df.sku.astype(str)
    return df


# In[ ]:


class InteractionDataset(Dataset):
    """
    A dataset for user-product interactions, containing user IDs, product SKUs, and labels indicating the type of interaction.
    """

    def __init__(self,df):
        """
        Initializes the dataset with a DataFrame of interaction data.
        
        Parameters:
            df (pandas.DataFrame): The DataFrame containing the interaction data.
        """
        self.df = df
        self.vocab = pd.DataFrame(list(set(self.df.sku))).reset_index().rename(columns={0:"sku"})

    def __len__(self):
        """
        Returns the number of interaction in the dataset.
        """
        return self.df.shape[0]

    def __getitem__(self,idx):
        """
        Returns the user ID, product SKU, and label for a interaction at a given index.
        
        Parameters:
            idx (int): The index of the interaction to return.
            
        Returns:
            tuple: A tuple containing the user ID, product SKU, and label for the interaction.
        """
        user_id = self.df.iloc[idx].user_id
        product_sku =  self.df.iloc[idx].sku
        product_sku = self.vocab[self.vocab["sku"]==product_sku].index[0]
        label =  self.df.iloc[idx]["type"]
        return (user_id,product_sku),label


# In[2]:


def get_ann_loader(df,batch_size):
    """
    Creates a dataset and data loader for training an ANN model.
    
    Parameters:
        df (pandas.DataFrame): A DataFrame of user-product interactions.
        batch_size (int): The batch size for the data loader.
        
    Returns:
        tuple: A tuple containing the dataset and data loader.
    """

    dataset = InteractionDataset(df)
    train_dataloader = DataLoader(dataset, batch_size, shuffle=True)
    return dataset, train_dataloader


# In[ ]:




