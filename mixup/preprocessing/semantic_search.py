# Code for to extract exemplars using semantic search
# File: semantic_search.py
# Author: Atharva Kulkarni

# ------------------------------------------ Extract similar exemplars ------------------------------------------

# nohup python3 -u semantic_search.py --data train --type similar > /home/atharvak/LCS2-Hate-Speech-Detection-Diffusion/out-files/exemplar/exemplar_retrieval_rank_semantic_train.out &

# nohup python3 -u semantic_search.py --data val --type similar > /home/atharvak/LCS2-Hate-Speech-Detection-Diffusion/out-files/exemplar/exemplar_retrieval_semantic_search_val.out &

# nohup python3 -u semantic_search.py --data test --type similar > /home/atharvak/LCS2-Hate-Speech-Detection-Diffusion//out-files/exemplar/exemplar_retrieval_semantic_search_test.out &


# ------------------------------------------ Extract dissimilar exemplars ------------------------------------------

# nohup python3 -u semantic_search.py --data train --type dissimilar > /home/atharvak/LCS2-Hate-Speech-Detection-Diffusion/out-files/exemplar/exemplar_retrieval_rank_semantic_train_dissimilar.out &

# nohup python3 -u semantic_search.py --data val --type dissimilar > /home/atharvak/LCS2-Hate-Speech-Detection-Diffusion/out-files/exemplar/exemplar_retrieval_semantic_search_val_dissimilar.out &

# nohup python3 -u semantic_search.py --data test --type dissimilar > /home/atharvak/LCS2-Hate-Speech-Detection-Diffusion//out-files/exemplar/exemplar_retrieval_semantic_search_test_dissimilar.out &

import argparse
import pickle
import random
import pandas as pd
import numpy as np
import os
import gc
from tqdm import tqdm
from datetime import datetime
import torch
import warnings
from sentence_transformers import SentenceTransformer, util

warnings.filterwarnings("ignore")

# -------------------------------------------------------------- CONFIG -------------------------------------------------------------- #

INPUT_PATH = '/home/atharvak/LCS2-Hate-Speech-Detection-Diffusion/dataset/csv/'
COLUMN = 'clean_text'
LABEL = 'label'



TWEET_MAX_LEN = 300
BATCH_SIZE = 32
TOP = 10



if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print("Using GPU")
else:
    DEVICE = torch.device("cpu")
    print("Using CPU")

    
    
def set_random_seed(seed: int):
    """
    Helper function to seed experiment for reproducibility.
    If -1 is provided as seed, experiment uses random seed from 0~9999
    Args:
        seed (int): integer to be used as seed, use -1 to randomly seed experiment
    """
    print("Seed: {}".format(seed))

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
SEED = 42
set_random_seed(SEED)    
    
if __name__ == "__main__":
    
    start = datetime.now()
    print("\n\nDocument retrieval started at ", start, "\n\n")
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default='train')
    parser.add_argument("--type", default='similar')
    args = parser.parse_args()
    
    print("\n\nLoading model...\n\n")
    MODEL = SentenceTransformer('bert-base-multilingual-cased')
    MODEL.max_seq_length = 300
    MODEL.to(DEVICE)
    
    
    
    if str(args.data) != 'train':
        print("\nRetreiving exemplars for : ", args.data)
        
        base_corpus = pd.read_csv(INPUT_PATH+'train_final.csv')[COLUMN].values.tolist()
        print("\nbase corpus length: ", len(base_corpus))
        
        corpus_embeddings = MODEL.encode(base_corpus, convert_to_tensor=True)
        print("\ncorpus_embeddings shape: ", corpus_embeddings.shape)
        
        path = INPUT_PATH + str(args.data) + '_final.csv'
        query_df = pd.read_csv(path)
        query_corpus = query_df[COLUMN].values.tolist()
        print("\nquery_corpus length: ", len(query_corpus))
        
        query_embeddings = MODEL.encode(query_corpus, convert_to_tensor=True)
        print("\nquery_embeddings shape: ", query_embeddings.shape)
        
        exemplar_data = util.semantic_search(query_embeddings, corpus_embeddings, top_k=int(TOP)+1)
        print("\nexemplar_data size: ", len(exemplar_data))
        exemplar_dict = dict()
        for exemplars, tweet_id in list(zip(exemplar_data, query_df['id'].values.tolist())):
            exemplar_dict[int(tweet_id)] = [base_corpus[exemplar['corpus_id']] for exemplar in exemplars[1:]]
        print("\nexemplar_dict size: ", len(exemplar_dict))
      
    
    
    
    else:
        print("Retreiving exemplars for : ", args.data)
        
        path = INPUT_PATH + str(args.data) + '_final.csv'
        df = pd.read_csv(path)
        print("\ndf size: ", df.shape)
        
        # ---------------------------------------------- Get exemplars for label 0 (hate) ----------------------------------------------
        
        corpus_0 = df[df['label'] == 0][COLUMN].values.tolist()
        print("\ncorpus_0 length: ", len(corpus_0))
        
        corpus_0_embeddings = MODEL.encode(corpus_0, convert_to_tensor=True)
        print("\ncorpus_0_embeddings shape: ", corpus_0_embeddings.shape)
        
        exemplar_data_0 = util.semantic_search(corpus_0_embeddings, corpus_0_embeddings, top_k=int(TOP)+1)
        print("\nexemplar_data_0: ", len(exemplar_data_0))
        exemplar_dict = dict()
        for exemplars, tweet_id in list(zip(exemplar_data_0, df[df['label'] == 0]['id'].values.tolist())):
            exemplar_dict[int(tweet_id)] = [corpus_0[exemplar['corpus_id']] for exemplar in exemplars[1:]]
        print("\nexemplar_dict size: ", len(exemplar_dict))
        
        # ---------------------------------------------- Get exemplars for label 1 (offensive) ----------------------------------------------
        corpus_1 = df[df['label'] == 1][COLUMN].values.tolist()
        print("\ncorpus_1 length: ", len(corpus_1))
        
        corpus_1_embeddings = MODEL.encode(corpus_1, convert_to_tensor=True)
        print("\ncorpus_1_embeddings shape: ", corpus_1_embeddings.shape)
        
        exemplar_data_1 = util.semantic_search(corpus_1_embeddings, corpus_1_embeddings, top_k=int(TOP)+1)
        print("\nexemplar_data_1: ", len(exemplar_data_1))
        for exemplars, tweet_id in list(zip(exemplar_data_1, df[df['label'] == 1]['id'].values.tolist())):
            exemplar_dict[int(tweet_id)] = [corpus_1[exemplar['corpus_id']] for exemplar in exemplars[1:]]
        print("\nexemplar_dict size: ", len(exemplar_dict))
       
        # --------------------------------------------- Get exemplars for label 2 (provocative) ---------------------------------------------
        corpus_2 = df[df['label'] == 2][COLUMN].values.tolist()
        print("\ncorpus_2 length: ", len(corpus_2))
        
        corpus_2_embeddings = MODEL.encode(corpus_2, convert_to_tensor=True)
        print("\ncorpus_2_embeddings shape: ", corpus_2_embeddings.shape)
        
        exemplar_data_2 = util.semantic_search(corpus_2_embeddings, corpus_2_embeddings, top_k=int(TOP)+1)
        print("\nexemplar_data_2: ", len(exemplar_data_2))
        for exemplars, tweet_id in list(zip(exemplar_data_2, df[df['label'] == 2]['id'].values.tolist())):
            exemplar_dict[int(tweet_id)] = [corpus_2[exemplar['corpus_id']] for exemplar in exemplars[1:]]
        print("\nexemplar_dict size: ", len(exemplar_dict))
       
        # ---------------------------------------------- Get exemplars for label 3 (control) ----------------------------------------------
        
        corpus_3 = df[df['label'] == 3][COLUMN].values.tolist()
        print("\ncorpus_3 length: ", len(corpus_3))
        
        corpus_3_embeddings = MODEL.encode(corpus_3, convert_to_tensor=True)
        print("\ncorpus_3_embeddings shape: ", corpus_3_embeddings.shape)
        
        exemplar_data_3 = util.semantic_search(corpus_3_embeddings, corpus_3_embeddings, top_k=int(TOP)+1)
        print("\nexemplar_data_3: ", len(exemplar_data_3))
        for exemplars, tweet_id in list(zip(exemplar_data_3, df[df['label'] == 3]['id'].values.tolist())):
            exemplar_dict[int(tweet_id)] = [corpus_3[exemplar['corpus_id']] for exemplar in exemplars[1:]]
        print("\nexemplar_dict size: ", len(exemplar_dict))
        
    
    
    
    pickle_path = '/home/atharvak/LCS2-Hate-Speech-Detection-Diffusion/dataset/exemplar/' + args.data + '_data_exemplar_' + str(TOP) + '_semantic_search.pickle'    
    with open(pickle_path, 'wb') as file:
        pickle.dump(exemplar_dict, file, protocol=pickle.HIGHEST_PROTOCOL)
    file.close()
    print("\n\nexemplar_dict saved")
    
    end = datetime.now()
    print("\n\nDocument retrieval ended at ", end)
    print("\n\nTotal time taken ", end-start)