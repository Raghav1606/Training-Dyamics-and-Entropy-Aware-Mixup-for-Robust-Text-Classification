## Training Dyamics and Entropy-Aware Mixup for Robust Text Classification

Script with the steps to run can be found in `run_details.pdf` in this reporsitory.

Paper Reference: 

- Arxiv: https://arxiv.org/abs/2109.06827
- Official: https://aclanthology.org/2021.emnlp-main.835/

## Environment
Please use the .yml file to set up the new environment before running the experiments. We used python version 3.10.

## Files
- `roberta_fine_tune.py` is used to finetune the Roberta models.
- `msp_eval.py` are used to find the MSPs of a dataset pair's examples using the finetuned model.

## How to run
These steps show how to train calibration models on the SST2 dataset, and evaluated against IMDB.

A differet dataset pair can be used by updating the approriate `dataset_name` or `id_data`/`ood_data` values as shown below:


### Training the Calibration Model (RoBERTa)
1. Using HF Datasets -
   ```
   id_data="sst2"
   nohup python -u roberta_fine_tune.py  --batch_size 16 --fname roberta_sst2 --output_dir roberta_ckpts_sst2/ --task_name sst2 > roberta_sst2
   ```


### Finding Maximum Softmax Probability (MSP)
1. Using HF Datasets -
   ```
   id_data="sst2"
   ood_data="imdb"
   python msp_eval.py --model_path roberta_ckpts/roberta-$id_data --dataset_name $ood_data --fname ${id_data}_$ood_data
   ```

### Evaluating AUROC
1. Compute AUROC of MSP -
    ```
   import utils
   id_data = 'sst2'
   ood_data = 'imdb'
   id_msp = utils.read_model_out(f'output/roberta/roberta_{id_data}_msp.npy')
   ood_msp = utils.read_model_out(f'output/msp/{id_data}_{ood_data}_msp.npy')
   score = utils.compute_auroc(-id_msp, -ood_msp)
   
    ```


Abstract = Interpolation-based data augmentation techniques such as ‘Mixup’ have proven helpful in data-scarce regimes. However, traditional mixup techniques choose samples randomly to generate synthetic data via linear interpolation. In this work, we propose an intelligent mixup technique that pairs data samples based on their difficulty and entropy for generating diverse and robust synthetic examples. To begin with, we identify samples that are easy- to-learn, mislabeled, and ambiguous using the model’s training dynamics. Then we prefer a mixup-based data augmentation technique that generates effective synthetic samples by intelligently combining easy and difficult samples based on their entropy in a label-aware fashion. These examples contribute toward improving model robustness and generalization capabilities. Moreover, by discarding the mislabeled examples and training the model with synthetically generated examples of the same size as the original dataset, our method yields about 2% improvement on the challenging iSarcasm dataset. Thus, requiring no extra data, our technique proves to be potent in the low-data setting. We further analyze our results along with ablation experiments. Lastly, we provide qualitative and quantitive analysis to assess our model’s performance.
