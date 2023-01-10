# item_based_recommendation
Item Based Recommender Systems Case Study for Data Scientist Interview

This repository contains a work for a product recommendation system that uses various machine learning models to generate personalized recommendations for users. The models include Nearest Neighbor, Graph and Artificial Neural Network. It also provides the option to use Principal Component Analysis (PCA) to reduce the dimensionality of the data before using the Nearest Neighbor model.Main idea behind project is, using various techniques to generate product2product recommendations.

This work used 3 different methods for product2product recommendations  

1-Using Co occurance matrix as product embeddings and using Nearest Neighbor algorithm to do recommendations.   
2-Using Graph algorithm to recommend similar products using graph relationship  
3-Using embedding based Neural Network to learn user-product embeddings and using Nearest Neighbor to recommend similar products  

## Getting Started

To get the code running on your local machine, you'll need to have Python and the necessary libraries installed. This code is compatible with Python 3.x.

The following libraries are required:

-pandas.  
-numpy.   
-sklearn.   
-torch.   

You can install these libraries using pip by running the command pip install -r requirements.txt.

## Data

The code uses the **MovieLens 20M** dataset which contains ratings for movies from users. By default, the code uses the full dataset but you can also use a subset of the data by setting args.data to "subset". The code will automatically download and extract the dataset if the args.download_data is set to "True".

## Running the Code

The code can be run by executing the script main.py.Also i highly encourage you to play with sandbox.ipnb.You can find all the function and classes straightforward.There are several command line arguments that you can use to specify the model to use, the size of the batch when using the Artificial Neural Network model, the number of epochs and the learning rate when training the model.

```
python main.py --model "pca_nn"

```

For Graph model,

```
python main.py --model "graph"

```

For Artificial Neural Network model


```
python main.py --model "ann" --ann_batch_size 1 --epoch 5 --learning_rate 0.001

```

## Recommendations

The code generates recommendations for each user in the dataset and saves them to a CSV file named recommendations_with_{args.model}.csv. This file will be saved to the root directory of the code and will contain the recommended movies for each user.

## Additional Note

The code was simplified and log functionality is removed, also modification of some hyperparameter and variables was done . I would recommend to check the code in that manner as well.

## To Do's
- Using Approximate Nearest Neighbor algorithms like Annoy or Faiss in order to recommend products in higher dimensions



