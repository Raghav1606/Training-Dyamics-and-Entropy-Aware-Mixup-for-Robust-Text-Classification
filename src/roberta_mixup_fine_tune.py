'''

nohup python3 -u roberta_mixup_fine_tune.py \
--task_name imdb \
--roberta_version roberta-base \
--sampling_type random \
--device 0 > ../out_files/modeling_mixup_roberta_imdb_random.out &

python3 roberta_mixup_fine_tune.py \
--task_name imdb \
--roberta_version roberta-base \
--sampling_type random \
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
    DataLoader
)

import torch.nn.functional as F
from tqdm import tqdm, trange

from transformers import (
    RobertaTokenizer,
    set_seed
)
from modeling_mixup_roberta import RobertaMixerForSequenceClassification
from config import *

# --------------------------------------------------- MixupDataset Class ---------------------------------------------------

def prepare_dataset(data, mixup=False, sampling_type='random'):
    
    # Remove none and hard examples
    data = data[(data['category'] != 'none') & (data['category'] != 'hard')].reset_index(drop=True)
    
    if not mixup:
        if sampling_type == 'sequential':
            sorting_dict = {
                'easy': 0,
                'ambiguous': 1
            }
            data = data.iloc[data.category.map(sorting_dict).argsort()].reset_index(drop=True)
        return data

    # Same class mixup    
    data_easy = data[data['category'] == 'easy'].reset_index(drop=True)
    temp_easy = data_easy[['idx', 'label', 'category']].copy().rename(columns={"idx": "idx_2", "label": "label_2", "category": "category_2"}).sample(frac=1).reset_index(drop=True)
    data_easy = pd.concat([data_easy, temp_easy], axis=1)

    data_ambiguous = data[data['category'] == 'ambiguous'].reset_index(drop=True)
    temp_ambiguous = data_ambiguous[['idx', 'label', 'category']].copy().rename(columns={"idx": "idx_2", "label": "label_2", "category": "category_2"}).sample(frac=1).reset_index(drop=True)
    data_ambiguous = pd.concat([data_ambiguous, temp_ambiguous], axis=1)
        
    same_data = pd.concat([data_easy, data_ambiguous]).sample(frac=1).reset_index(drop=True)
    same_data['mixup_type'] = 'same'
    
    # Different class mixup
    data_easy = data[data['category'] == 'easy'].reset_index(drop=True)
    data_ambiguous = data[data['category'] == 'ambiguous'].reset_index(drop=True)

    easy_tuple = list(zip(data_easy['idx'].tolist(), data_easy['label'].tolist(), data_easy['category'].tolist()))
    ambiguous_tuple = list(zip(data_ambiguous['idx'].tolist(), data_ambiguous['label'].tolist(), data_ambiguous['category'].tolist()))

    ambiguous4easy = random.choices(ambiguous_tuple, weights=np.ones(len(ambiguous_tuple)), k=len(data_easy))
    easy4ambiguous = random.choices(easy_tuple, weights=np.ones(len(easy_tuple)), k=len(data_ambiguous))

    ambiguous4easy = pd.DataFrame(ambiguous4easy, columns=['idx_2', 'label_2', 'category_2'])
    data_easy = pd.concat([data_easy, ambiguous4easy], axis=1)

    easy4ambiguous = pd.DataFrame(easy4ambiguous, columns=['idx_2', 'label_2', 'category_2'])
    data_ambiguous = pd.concat([data_ambiguous, easy4ambiguous], axis=1)

    different_data = pd.concat([data_easy, data_ambiguous]).sample(frac=1).reset_index(drop=True)
    different_data['mixup_type'] = 'different'

    data = pd.concat([same_data, different_data]).sample(frac=1).reset_index(drop=True)
    
    if sampling_type == 'sequential':
        sorting_dict = {
            'non_mixup_easy': 0,
            'non_mixup_ambiguous': 1,
            'same_easy': 2,
            'different_easy': 3,
            'same_ambiguous': 4,
            'different_ambiguous': 5
        }
        data['data_type'] = data['mixup_type'] + '_' + data['category']
        data = data.iloc[data.data_type.map(sorting_dict).argsort()].reset_index(drop=True)
    
    return data  



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
            idx2 = self.data.iloc[index]['idx_2']

            if idx2 is not None:
                index2 = int(self.data[self.data['idx'] == idx2].index[0])
                data['input_ids_2'] = self.tokenized_data['input_ids'][index2].flatten()
                data['attention_mask_2'] = self.tokenized_data['attention_mask'][index2].flatten()
                data['labels_2'] = torch.tensor(self.data.iloc[index2][OUTPUT_COLUMN], dtype=torch.long)

        return data



# --------------------------------------------------- Train Utils ---------------------------------------------------

def train(model, tokenizer, optimizer, device, train_data, eval_data, sampling_type, shuffle, num_epochs, output_dir, save_path, fname):
    
    train_data_processed = prepare_dataset(train_data, mixup=False, sampling_type=sampling_type)
    train_dataset = MixupDataset(data=train_data_processed, dataset_type='train', tokenizer=tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=shuffle)
    print(f"\n\nNormal train data size: {len(train_data_processed)}")
    print(f"Normal train_loader size: {len(train_loader)}\n\n")

    eval_dataset = MixupDataset(data=eval_data, dataset_type='eval', tokenizer=tokenizer)
    eval_loader = DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=False)
    print(f"\n\nEval data size: {len(eval_data)}")
    print(f"eval_loader size: {len(eval_loader)}")

    losses = []
    val_losses = []
    train_iterator = trange(int(num_epochs), desc='Epoch')
    for epoch in train_iterator:
        
        model.train()
        if epoch + 1 == MIXUP_START:
            del train_data_processed
            del train_dataset
            del train_loader
            # del epoch_iterator
            gc.collect()
            torch.cuda.empty_cache()

            train_data_processed = prepare_dataset(train_data, mixup=True, sampling_type=sampling_type)
            train_dataset = MixupDataset(data=train_data_processed, dataset_type='mixup', tokenizer=tokenizer)
            train_loader = DataLoader(train_dataset, batch_size=4, shuffle=shuffle)
            print(f"\n\nMixup train data size: {len(train_data_processed)}")
            print(f"Mixup train_loader size: {len(train_loader)}\n\n")

        tr_loss = 0
        step = None
        epoch_iterator = tqdm(train_loader, desc='Training')
        for step, batch in enumerate(epoch_iterator):
            
            optimizer.zero_grad()

            inputs = {k:v.to(device) for k, v in batch.items()}
            labels = inputs['labels']
            
            outputs = model(**inputs)
            out = outputs['logits'].double().to(device)
            loss = outputs['loss']

            loss.backward()
            optimizer.step()

            tr_loss += loss.item()
        losses.append(tr_loss/(step+1))
        print('\ntrain loss: {}'.format(tr_loss/(step+1)))

        # evaluate model
        if eval_loader is not None:
            print('\n\nEvaluating model')
            probs, val_loss = eval(model, eval_loader, device, with_labels=True)

            if epoch == 0 or val_loss  < min(val_losses):
                print('\n\nSaving model and tokenizer..')
                model.save_pretrained(output_dir)
                tokenizer.save_pretrained(output_dir)

                print("\n\nSaving eval results...")
                np.save(os.path.join(save_path, f'{fname}_probs'), probs)
                msp = np.max(probs, axis=1)
                if fname is not None:
                    np.save(os.path.join(save_path, f'{fname}_msp'), msp)

            val_losses.append(val_loss)
    # save model and tokenizer



def eval(model, eval_loader, device, with_labels=True):
    probs = None
    gold_labels = None

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

    return probs, eval_loss



# --------------------------------------------------- Train Utils ---------------------------------------------------

def main():

    print("\n\nThis process has the PID: ", os.getpid())
    
    # create argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', help='Task to fine-tune RoBERTa on', default='sst2')
    parser.add_argument('--roberta_version', type=str, default='roberta-base', help='Version of RoBERTa to use')
    parser.add_argument('--device', type=int, default=0, help='which GPU to use')
    parser.add_argument('--sampling_type', type=str, default='random', help='How to sample data')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for initialization')
    parser.add_argument('--file_format', type=str, default='.tsv', help='Data file format for tasks not available for download at HuggingFace Datasets')
    parser.add_argument('--n', type=int, default=None, help='Number of examples to process (for debugging)')
    args = parser.parse_args()

    # set device
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    print(f"\n\nDevice: {device}")

    num_labels = 2

    # Saving path
    SAVE_PATH = f'../output/roberta_ckpts_mixup_{args.sampling_type}_{args.task_name}/'
    FNAME = f'roberta_mixup_{args.sampling_type}_{args.task_name}'
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)
    OUTPUT_DIR = f'../model_checkpoints/roberta_ckpts_mixup_{args.sampling_type}_{args.task_name}'

    print(f"\n\nSAVE_PATH: {SAVE_PATH}")
    print(f"\OUTPUT_DIR: {OUTPUT_DIR}")

    # Shuffling decision
    if args.sampling_type == 'sequential' or args.sampling_type == 'suby':
        SHUFFLE = False
    else:
        SHUFFLE = True
    
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

    # instantiate optimizer
    decay = []
    no_decay = []
    skip_params = ['LayerNorm', 'bias']
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        elif len(param.shape) == 1 or name in skip_params:
            print(name)
            no_decay.append(param)
        else:
            decay.append(param)

    optimizer = optim.AdamW([
        {'params': no_decay, 'lr': LEARNING_RATE, 'weight_decay': 0.0},
        {'params': decay, 'lr': LEARNING_RATE, 'weight_decay': WEIGHT_DECAY}
    ])

    # fine-tune model 
    if train_df is not None:
        print('\nFine-tuning model')
        train(model, tokenizer, optimizer, device, train_df, eval_df, args.sampling_type, SHUFFLE, NUM_EPOCHS, OUTPUT_DIR, SAVE_PATH, FNAME)


if __name__ == '__main__':
    main()
    print("\n\n--------DONE--------")