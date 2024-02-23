import json
import sys
import os
from functools import partial

import pandas as pd
import numpy as np
import tensorflow as tf
import gzip

tf.get_logger().setLevel('ERROR')  # only show error messages

from recommenders.utils.timer import Timer
from recommenders.models.deeprec.models.graphrec.lightgcn import LightGCN
from recommenders.models.deeprec.DataModel.ImplicitCF import ImplicitCF
from recommenders.datasets import movielens
from recommenders.datasets.python_splitters import python_stratified_split
from recommenders.evaluation.python_evaluation import ndcg_at_k, precision_at_k, recall_at_k, map_at_k
from recommenders.utils.constants import SEED as DEFAULT_SEED
from recommenders.models.deeprec.deeprec_utils import prepare_hparams

print(f"System version: {sys.version}")
print(f"Pandas version: {pd.__version__}")
print(f"Tensorflow version: {tf.__version__}")

# top k items to recommend
TOP_K = 10

# Select MovieLens data size: 100k, 1m, 10m, or 20m
MOVIELENS_DATA_SIZE = '100k'

# Model parameters
EPOCHS = 50
BATCH_SIZE = 100000  # original 1024

SEED = DEFAULT_SEED  # Set None for non-deterministic results

yaml_file = "C:/Users/Lukas/Desktop/studienarbeit/lightgcn.yaml"
user_file = "C:/Users/Lukas/Desktop/studienarbeit/user_embeddings.csv"
item_file = "C:/Users/Lukas/Desktop/studienarbeit/item_embeddings.csv"
amazon_auto_file = "C:/Users/Lukas/Desktop/studienarbeit/qa_Automotive.json.gz"
google_local_reviews_small_alabama = "C:/Users/Lukas/Desktop/studienarbeit/review-Alaska_10.json.gz"


if __name__ == '__main__':
    df_default = movielens.load_pandas_df(size=MOVIELENS_DATA_SIZE)

    df_default.head()


    def parse_amazon(path):
        g = gzip.open(path, 'rb')
        for l in g:
            yield eval(l)


    def getDF(path):
        i = 0
        df = {}
        for d in parse_amazon(path):
            df[i] = d
            i += 1
        return pd.DataFrame.from_dict(df, orient='index')


    def parse_google(path):
        g = gzip.open(path, 'r')
        for l in g:
            yield json.loads(l)

    def get_df_google(path):
        i = 0
        df = {}
        for d in parse_google(path):
            df[i] = d
            i += 1
        return pd.DataFrame.from_dict(df, orient='index')

    df_amazon = getDF(amazon_auto_file)

    df_google = get_df_google(google_local_reviews_small_alabama)
    df_google.rename(columns={"user_id": "userID", "gmap_id": "itemID"}, inplace=True)

    # Split data set into two parts train
    # all users are in both train and test datasets
    train_g, test_g = python_stratified_split(df_google, ratio=0.75, col_user="userID", col_item="itemID")
    # train_d, test_d = python_stratified_split(df_default, ratio=0.75)
    # Training data with at least columns (col_user, col_item, col_rating).
    data_g = ImplicitCF(train=train_g, test=test_g, seed=SEED, col_user="userID", col_item="itemID")
    # data_d = ImplicitCF(train=train_d, test=test_d, seed=SEED)
    # df_amazon.to_latex("ergebnisse.tex", c)
    hparams = prepare_hparams(yaml_file,
                              n_layers=3,
                              batch_size=BATCH_SIZE,
                              epochs=EPOCHS,
                              learning_rate=0.005,
                              eval_epoch=5,
                              top_k=TOP_K,
                              )
    model = LightGCN(hparams, data_g, seed=SEED)

    with Timer() as train_time:
        model.fit()

    print("Took {} seconds for training.".format(train_time.interval))

    topk_scores = model.recommend_k_items(test_g, top_k=TOP_K, remove_seen=True)

    topk_scores.head()

    eval_map = map_at_k(test_g, topk_scores, k=TOP_K)
    eval_ndcg = ndcg_at_k(test_g, topk_scores, k=TOP_K)
    eval_precision = precision_at_k(test_g, topk_scores, k=TOP_K)
    eval_recall = recall_at_k(test_g, topk_scores, k=TOP_K)

    print("MAP:\t%f" % eval_map,
          "NDCG:\t%f" % eval_ndcg,
          "Precision@K:\t%f" % eval_precision,
          "Recall@K:\t%f" % eval_recall, sep='\n')

    # # Record results for tests - ignore this cell
    # store_metadata("map", eval_map)
    # store_metadata("ndcg", eval_ndcg)
    # store_metadata("precision", eval_precision)
    # store_metadata("recall", eval_recall)

    model.infer_embedding(user_file, item_file)
