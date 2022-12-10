# Author: Atharva Kulkarni
# File to train entropy-based mixup Roberta

'''

nohup python3 -u roberta_mixup_fine_tune.py \
--task_name sarcasm \
--roberta_version roberta-base \
--include_none \
--mixup_type category \
--mixup_use_label \
--mixup_use_entropy \
--device 0 > ../out_files/sarcasm/modeling_roberta_mixup_use_none_category_use_label_use_entropy_sarcasm_flag_elu_jsd_weighted_sampling.out &

nohup python3 -u roberta_mixup_fine_tune.py \
--task_name sarcasm \
--roberta_version roberta-base \
--include_none \
--mixup_type category \
--mixup_use_label \
--device 1 > ../out_files/sarcasm/modeling_roberta_non_easy_ambi.out &

python3 roberta_mixup_fine_tune.py \
--task_name imdb \
--roberta_version roberta-base \
--mixup_type random \
--device 0

'''

import os
import fire
import argparse

import numpy as np
import pandas as pd
import random
import gc

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import (
    Dataset, 
    DataLoader,
    WeightedRandomSampler
)

from sklearn.metrics import (
    f1_score, 
    recall_score, 
    precision_score, 
    accuracy_score
)

import torch.nn.functional as F
from tqdm import tqdm, trange

from transformers import (
    RobertaTokenizer,
    set_seed
)
from modeling_mixup_roberta import RobertaMixerForSequenceClassification
from config import *
from mixup_data_utils import *



class MixupDataset(Dataset):
    
    def __init__(
        self, 
        data: pd.DataFrame,
        dataset_type: str,
        tokenizer
    ):
        self.data = data
        self.dataset_type = dataset_type
        
        self.tokenizer = tokenizer
        self.tokenized_data = tokenizer.batch_encode_plus(
            self.data[INPUT_COLUMN].tolist(),
            max_length=MAX_LEN,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_token_type_ids=True,            
            return_tensors='pt'
        )
        if self.dataset_type =='mixup':
            self.tokenized_data_2 = tokenizer.batch_encode_plus(
                self.data[f"{INPUT_COLUMN}_2"].tolist(),
                max_length=MAX_LEN,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                add_special_tokens=True,
                return_token_type_ids=True,            
                return_tensors='pt'
            )

        
                
    def __len__(
        self
    ):
        return len(self.data)
    
    
    
    def __getitem__(
        self,
        index: int
    ):
        data = {
            'input_ids': self.tokenized_data['input_ids'][index].flatten(),
            'attention_mask': self.tokenized_data['attention_mask'][index].flatten(),
            'labels': torch.tensor(self.data.iloc[index][OUTPUT_COLUMN], dtype=torch.long),
        }
        
        if self.dataset_type == 'eval':
            return data

        if self.dataset_type =='mixup':
            data['input_ids_2'] = self.tokenized_data_2['input_ids'][index].flatten()
            data['attention_mask_2'] = self.tokenized_data_2['attention_mask'][index].flatten()
            data['labels_2'] = torch.tensor(self.data.iloc[index][f"{OUTPUT_COLUMN}_2"], dtype=torch.long)

        return data



# --------------------------------------------------- Train Utils ---------------------------------------------------

def get_optimizer(model):
    # return optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    decay = []
    no_decay = []
    skip_params = ['LayerNorm', 'bias']
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        elif len(param.shape) == 1 or name in skip_params:
            no_decay.append(param)
        else:
            decay.append(param)

    return optim.AdamW([
        {'params': no_decay, 'lr': LEARNING_RATE, 'weight_decay': 0.0},
        {'params': decay, 'lr': LEARNING_RATE, 'weight_decay': WEIGHT_DECAY}
    ])



def get_class_weights(df, column_label):
    column_label_dict = {k: i for i, k in enumerate(df[column_label].unique().tolist())}
    print("column_label_dict: ", column_label_dict)
    df['integer'] = df[column_label].apply(lambda x: column_label_dict[x])
    target_list = torch.tensor(df.integer)
    class_count = np.bincount(df.integer)
    class_weights = 1./torch.tensor(class_count, dtype=torch.float)
    class_weights_all = class_weights[target_list]
    
    return class_weights_all



def train(model, tokenizer, device, train_data, eval_data, arguments):

    # Saving path
    entropy = ""
    if arguments.mixup_use_entropy:
        entropy = 'use_entropy' 
    label = ""   
    if arguments.mixup_use_label:
        label = 'use_label'
    none = ""
    if arguments.include_none:
        non = "use_none"

    SAVE_PATH = f'../output/roberta_ckpts_mixup_{arguments.include_none}_{arguments.mixup_type}_{label}_{entropy}_{arguments.task_name}/'
    FNAME = f'roberta_mixup_{arguments.include_none}_{arguments.mixup_type}_{label}_{entropy}_{arguments.task_name}'
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)
    OUTPUT_DIR = f'../model_checkpoints/roberta_ckpts_mixup_{arguments.include_none}_{arguments.mixup_type}_{label}_{entropy}_{arguments.task_name}'
    print(f"\n\nSAVE_PATH: {SAVE_PATH}")
    print(f"\OUTPUT_DIR: {OUTPUT_DIR}")

    # Process train dataset
    train_data_processed = prepare_dataset_original(train_data, include_none=arguments.include_none)
    
    CLASS_WEIGHTS = get_class_weights(train_data_processed, column_label='category')
    print("\n\nCLASS_WEIGHTS size: ", len(CLASS_WEIGHTS))
    weighted_sampler = WeightedRandomSampler(
            weights=CLASS_WEIGHTS,
            num_samples=len(CLASS_WEIGHTS),
            replacement=True
        )  

    train_dataset = MixupDataset(data=train_data_processed, dataset_type='train', tokenizer=tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=weighted_sampler, shuffle=False)
    print(f"\n\nNormal train data size: {len(train_data_processed)}")
    print(f"Normal train_loader size: {len(train_loader)}\n\n")

    # Process eval dataset
    eval_dataset = MixupDataset(data=eval_data, dataset_type='eval', tokenizer=tokenizer)
    eval_loader = DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=False)
    print(f"\n\nEval data size: {len(eval_data)}")
    print(f"eval_loader size: {len(eval_loader)}")

    # Get optimizer
    optimizer = get_optimizer(model)
    grad_acc_step = 1

    losses = []
    val_losses = []
    train_iterator = trange(int(NUM_EPOCHS), desc='Epoch')
    for epoch in train_iterator:
        
        model.train()

        if epoch + 1 == MIXUP_START:
            del train_data_processed
            del train_dataset
            del train_loader
            # del epoch_iterator
            gc.collect()
            torch.cuda.empty_cache()

            info_df = pd.read_csv(f'../dy_log/{arguments.task_name}/roberta-base/training_dynamics/final_4.csv')
            info_df = info_df.rename(columns={'guid': 'idx', 'sm': 'softmax', 'en': 'entropy'})

            if arguments.mixup_type == 'random':
                print("\n\nRandom mixup processing...\n")
                train_data_processed = prepare_dataset_random_mixup(train_data, info_data=info_df, include_none=arguments.include_none, use_label=arguments.mixup_use_label)
            elif arguments.mixup_type == 'category':
                print("\n\nCategory-based mixup processing...\n")
                train_data_processed = prepare_dataset_category_mixup(train_data, info_data=info_df, include_none=arguments.include_none, use_label=arguments.mixup_use_label, use_entropy=arguments.mixup_use_entropy)

            MIXUP_CLASS_WEIGHTS = get_class_weights(train_data_processed, column_label='mixup_type')
            print("\n\nCLASS_WEIGHTS size: ", len(MIXUP_CLASS_WEIGHTS))
            weighted_sampler = WeightedRandomSampler(
                    weights=MIXUP_CLASS_WEIGHTS,
                    num_samples=len(MIXUP_CLASS_WEIGHTS),
                    replacement=True
                )  

            train_dataset = MixupDataset(data=train_data_processed, dataset_type='mixup', tokenizer=tokenizer)

            grad_acc_step = 2
            train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE//grad_acc_step, sampler=weighted_sampler, shuffle=False)
            print(f"\n\nMixup train data size: {len(train_data_processed)}")
            print("Mixup label distribution: ")
            get_count(train_data_processed, 'label', 'category')  
            print(f"Mixup train_loader size: {len(train_loader)}\n\n")

        tr_loss = 0
        step = None
        epoch_iterator = tqdm(train_loader, desc='Training')
        for step, batch in enumerate(epoch_iterator):
        
            inputs = {k:v.to(device) for k, v in batch.items()}
            labels = inputs['labels']
            
            optimizer.zero_grad()

            outputs = model(**inputs)
            out = outputs['logits'].double().to(device)
            loss = outputs['loss']

            loss.backward()

            if ((step + 1) % grad_acc_step == 0) or (step + 1 == len(train_loader)):
                optimizer.step()
                optimizer.zero_grad()

            tr_loss += loss.item()
        losses.append(tr_loss/(step+1))
        print('\ntrain loss: {}'.format(tr_loss/(step+1)))

        # evaluate model
        if eval_loader is not None:
            print('\n\nEvaluating model')
            probs, val_loss = eval(model, eval_loader, device, with_labels=True)

            if epoch == 0 or val_loss  < min(val_losses):
                print('\n\nSaving model and tokenizer..')
                # model.save_pretrained(OUTPUT_DIR)
                # tokenizer.save_pretrained(OUTPUT_DIR)

                # print("\n\nSaving eval results...")
                # np.save(os.path.join(SAVE_PATH, f'{FNAME}_probs'), probs)
                # msp = np.max(probs, axis=1)
                # if FNAME is not None:
                #     np.save(os.path.join(SAVE_PATH, f'{FNAME}_msp'), msp)

            val_losses.append(val_loss)
    # save model and tokenizer



def eval(model, eval_loader, device, with_labels=True):
    probs = None
    gold_labels = None


    SAVE_PATH = f'../results/{arguments.task_name}/'
            f'roberta_mixup_{arguments.include_none}_{arguments.mixup_type}_{label}_{entropy}_{arguments.task_name}/'
    FNAME = f'roberta_mixup_{arguments.include_none}_{arguments.mixup_type}_{label}_{entropy}_{arguments.task_name}'
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)
    OUTPUT_DIR = f'../model_checkpoints/roberta_ckpts_mixup_{arguments.include_none}_{arguments.mixup_type}_{label}_{entropy}_{arguments.task_name}'
    print(f"\n\nSAVE_PATH: {SAVE_PATH}")
    print(f"\OUTPUT_DIR: {OUTPUT_DIR}")

    eval_loss = 0
    step = None
    eval_iterator = tqdm(eval_loader, desc='Evaluating')
    for step, batch in enumerate(eval_iterator):
        model.eval()

        with torch.no_grad():
            
            inputs = {k:v.to(device) for k, v in batch.items()}
            labels = inputs['labels']
 
            outputs = model(**inputs)
            out = outputs['logits'].double().to(device)
            out = F.softmax(out, dim=1)
            loss = outputs['loss']

            if probs is None:
                probs = out.detach().cpu().numpy()
                if with_labels:
                    gold_labels = labels.detach().cpu().numpy()
            else:
                probs = np.append(probs, out.detach().cpu().numpy(), axis=0)
                if with_labels:
                    gold_labels = np.append(gold_labels, labels.detach().cpu().numpy(), axis=0)

            if with_labels:
                eval_loss += loss.item()
    
    if with_labels:
        eval_loss /= (step+1)
        print('\neval loss: {}'.format(eval_loss))

        # compute accuracy
        preds = np.argmax(probs, axis=1)
        accuracy = np.sum(preds == gold_labels)/len(preds)

        print('eval accuracy: {}'.format(accuracy))
        # Compute other scores
        print('eval precision: {}'.format(round(precision_score(gold_labels, preds, average='macro'), 5)))
        print('eval recall: {}'.format(round(recall_score(gold_labels, preds, average='macro'), 5)))
        print('eval f1_score: {}'.format(round(f1_score(gold_labels, preds, average='macro'), 5)))

        preds = preds.tolist()
        gold_labels = gold_labels.tolist()
        results = pd.DataFrame(list(zip(gold_labels, preds)), columns=['gold', 'preds'])
        results.to_csv(f"{}.csv", index=False)

    return probs, eval_loss



# --------------------------------------------------- Train Utils ---------------------------------------------------

def main():

    print("\n\nThis process has the PID: ", os.getpid())
    
    # create argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', help='Task to fine-tune RoBERTa on', default='sst2')
    parser.add_argument('--roberta_version', type=str, default='roberta-base', help='Version of RoBERTa to use')
    parser.add_argument('--device', type=int, default=0, help='which GPU to use')
    parser.add_argument('--include_none', action='store_true', help='Use none category data during training')
    parser.add_argument('--mixup_type', type=str, default='random', help='How to sample data', choices=('random', 'category'))
    parser.add_argument('--mixup_use_entropy', action='store_true', help='Do use entropy based matching for mixup')
    parser.add_argument('--mixup_use_label', action='store_true', help='Do mixup of same labels')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for initialization')
    parser.add_argument('--file_format', type=str, default='.tsv', help='Data file format for tasks not available for download at HuggingFace Datasets')
    args = parser.parse_args()

    # set device
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    print(f"\n\nDevice: {device}")
    
    # set seed
    set_seed(args.seed)

    # load RoBERTa tokenizer and model
    print('\n\nLoading RoBERTa tokenizer and model')
    tokenizer = RobertaTokenizer.from_pretrained(args.roberta_version, cache_dir='cache/huggingface/transformers')
    model = RobertaMixerForSequenceClassification.from_pretrained(args.roberta_version, cache_dir='cache/huggingface/transformers', num_labels=2).to(device)
    model.resize_token_embeddings(len(tokenizer))

    # process dataset
    print('\n\nReading dataset\n')

    # Process train dataset
    train_file = f'../datasets/{args.task_name}/{args.task_name}_categorized.csv'
    train_df = pd.read_csv(train_file)
    
    # Process eval dataset
    val_file = f'../datasets/{args.task_name}/test.csv'
    eval_df = pd.read_csv(val_file)    

    # fine-tune model 
    if train_df is not None:
        print('\nFine-tuning model')
        train(model, tokenizer, device, train_df, eval_df, args)


if __name__ == '__main__':
    main()
    print("\n\n--------DONE--------")