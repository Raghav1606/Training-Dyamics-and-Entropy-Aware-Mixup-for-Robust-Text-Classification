# Author: Atharva Kulkarni
# Config file to train entropy-based mixup Roberta

FLAG = False
MIXUP_START = 3
LAMBDA = 0.5

INPUT_COLUMN = 'text'
DATA_COLUMN = 'category'
OUTPUT_COLUMN = 'label'

NUM_EPOCHS = 10
MAX_LEN = None
BATCH_SIZE = 8
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 0.001

print("INPUT_COLUMN: ", INPUT_COLUMN)
print("DATA_COLUMN: ", DATA_COLUMN)
print("OUTPUT_COLUMN: ", OUTPUT_COLUMN)
print("NUM_EPOCHS: ", NUM_EPOCHS)
print("MAX_LEN: ", MAX_LEN)
print("BATCH_SIZE: ", BATCH_SIZE)
print("LAMBDA: ", LAMBDA)
print("FLAG: ", FLAG)
print("MIXUP_START: ", MIXUP_START)