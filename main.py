# Imports

from utils import download_and_unzip,prepare_dataset,InteractionDataset,get_ann_loader
from argparse import ArgumentParser
import logging
from helpers import interactions_as_embeddings,generate_recommendations,transform_embeddings_with_kernel_pca,get_ann_embeddings
from models import find_nearest_neighbors,graph_recommendations,embedding_learning_model
import pandas as pd
from torch import nn
import torch
import torch.optim as optim

# Create and configure logger
logging.basicConfig(level = logging.INFO)

# Creating an object
log = logging.getLogger()

# Create Args
parser = ArgumentParser()
parser.add_argument("--data", default="sample", type=str)
parser.add_argument("--download_data", default="False", type=str)
parser.add_argument("--model", default="ann", type=str)
parser.add_argument("--ann_batch_size", default=1, type=int)
parser.add_argument("--epoch", default=5, type=int)
parser.add_argument("--learning_rate", default=0.001, type=float)

args, unknown = parser.parse_known_args()

if __name__=='__main__' :
    
    # Download and loading data
    if args.download_data == "True":
        log.info("Downloading data has started")

        download_and_unzip("https://files.grouplens.org/datasets/movielens/ml-20m.zip")
    data = pd.read_csv("ml-20m/ratings.csv")
    if args.data != "full":
        df = prepare_dataset(data).iloc[:10000]
    else:
        df = prepare_dataset(data)

    # train and run Neirest Neighbor model with occurrence matrix embeddings
    if args.model == "nn" or args.model == "pca_nn":
        embeddings = interactions_as_embeddings(df)
        
        # train and run Neirest Neighbor model after reduce dimensions of embeddings with PCA
        if args.model == "pca_nn":
            log.info("Neirest Neighbor with pca procedure has started")
            embeddings=transform_embeddings_with_kernel_pca(embeddings)

        # Neirest Neighbor predictions
        log.info("Neirest Neighbor procedure has started")

        indices, index_dict = find_nearest_neighbors(embeddings)
        recommendations = generate_recommendations(indices,embeddings,index_dict)

    # Run Graph model and predict
    elif args.model == "graph":
        log.info("Graph procedure has started")

        recommendations = graph_recommendations(df)

    # train and run Artificial Neural Network model 
    else:
        log.info("ANN procedure has started")

        dataset,train_loader = get_ann_loader(df,args.ann_batch_size)
        unique_user = df.user_id.nunique()
        unique_product = df.sku.nunique()

        model = embedding_learning_model(unique_user+1,unique_product+1) 
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
        log.info("ANN training has started")

        for epoch in range(args.epoch):
            for idx,(x,y) in enumerate(train_loader):
                y_hat = model(x)
                loss = criterion(y_hat,y)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

    # Predictions for Artificial Neural Network model 
        log.info("ANN prediction procedure has started")

        embeddings = get_ann_embeddings(model,df,dataset)
        indices, index_dict = find_nearest_neighbors(embeddings)
        recommendations = generate_recommendations(indices,embeddings,index_dict)
        
    recommendations.to_csv(f"recommendations_with_{args.model}.csv")
    log.info("Recommendations saved")


