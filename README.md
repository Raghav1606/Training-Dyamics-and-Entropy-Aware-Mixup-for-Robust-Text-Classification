## Training Dyamics and Entropy-Aware Mixup for Robust Text Classification

![image](https://user-images.githubusercontent.com/44756809/206886033-67aa0727-e0a5-45cf-85e4-33073c059bce.png)


Paper Reference: 

- https://arxiv.org/pdf/2009.10795.pdf
- Official: https://aclanthology.org/2021.emnlp-main.835/

## Environment
Please use the .yml file to set up the new environment before running the experiments. We used python version 3.10.


## Files
-  `run_glue.py` - to collect data logits for segregating it into easy, hard and ambiguous data instances.
- `data_selection.py` - to segregate data into easy, hard and ambiguous instances`
- `roberta_fine_tune.py` is used to finetune the Roberta models.

## How to run

For example, we want to record the training dynamics of iSarcasm dataset, we do the following steps:

...
python src/run_glue.py --train_file datasets/sarcasm/train.csv --validation_file datasets/sarcasm/test.csv --model_name_or_path roberta-base 
...

The following infomation will be recorded during training:

After training, we can find the log files in ./dy_log/{TASK_NAME}/{MODEL}/training_dynamics directory like:

dynamics_epoch_0.jsonl
dynamics_epoch_1.jsonl
dynamics_epoch_2.jsonl
...
each file contains records like:

{"guid": 50325, "logits_epoch_0": [2.943110942840576, -2.2836594581604004], "gold": 0, "device": "cuda:0"}
{"guid": 42123, "logits_epoch_0": [-2.7155513763427734, 3.249767541885376], "gold": 1, "device": "cuda:0"}
{"guid": 42936, "logits_epoch_0": [-1.1907235383987427, 2.1173453330993652], "gold": 1, "device": "cuda:0"}
...

### Data Selection
After recording the training dynamics, we can re-train the model by selecting a subset (e.g. use only the ambiguous samples for training). For example, for iSarcasm task and roberta-base model, just run:
...
python data_selection.py --task_name sarcasm --model_name roberta_base --burn_out 4
...
then you can get a json file at dy_log/sarcasm/roberta_base/three_regions_data_indices.json

### Data Categorization

Using these idices, we ran the notebooks/data_prep.py file for saving the data into {DATASET_NAME}_easy /  {DATASET_NAME}_ambi etc. This can be found in Datasets folder. 

We ran in-domain classification using roberta_fine_tune.py

...
python -u roberta_fine_tune.py  --batch_size 8 --fname roberta_sarcasm_easy  --train_file datasets/sarcasm/sarcasm_easy.csv --val_file datasets/sarcasm/test.csv --output_dir roberta_ckpts_sarcasm_easy/ --task_name sarcasm >outfile/roberta_sarcasm_easy
...

### Mixup



Abstract = Interpolation-based data augmentation techniques such as ‘Mixup’ have proven helpful in data-scarce regimes. However, traditional mixup techniques choose samples randomly to generate synthetic data via linear interpolation. In this work, we propose an intelligent mixup technique that pairs data samples based on their difficulty and entropy for generating diverse and robust synthetic examples. To begin with, we identify samples that are easy- to-learn, mislabeled, and ambiguous using the model’s training dynamics. Then we prefer a mixup-based data augmentation technique that generates effective synthetic samples by intelligently combining easy and difficult samples based on their entropy in a label-aware fashion. These examples contribute toward improving model robustness and generalization capabilities. Moreover, by discarding the mislabeled examples and training the model with synthetically generated examples of the same size as the original dataset, our method yields about 2% improvement on the challenging iSarcasm dataset. Thus, requiring no extra data, our technique proves to be potent in the low-data setting. We further analyze our results along with ablation experiments. Lastly, we provide qualitative and quantitive analysis to assess our model’s performance.
