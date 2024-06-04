import os
from sklearn.model_selection import train_test_split
import numpy as np
import json

############# CONFIG #############

DATA_PATH = 'data/processed_dataset' # data source
SAVE_PATH = 'split_3sets.json' # save path to store splitting info
PRETRAIN_SIZE = 0.8
VALIDATION_SIZE = 0.1
TEST_SIZE = 0.1

############# END CONFIG #############

assert (PRETRAIN_SIZE + VALIDATION_SIZE + TEST_SIZE) == 1

path_indices = np.arange(len(os.listdir(DATA_PATH)))
pretraining_indices, valtest_indices = train_test_split(path_indices, test_size=(1 - PRETRAIN_SIZE), random_state=121)

full_paths = np.array(os.listdir(DATA_PATH))

# SPLIT
# Split pretrain set
valtest_paths = full_paths[valtest_indices]
valtest_indices = np.arange(len(valtest_paths))
test_size = TEST_SIZE / (VALIDATION_SIZE + TEST_SIZE)
val_indices, test_indices = train_test_split(valtest_indices, test_size=test_size, random_state=121)

# Split validation and test set
pretraining_paths = full_paths[pretraining_indices]
val_paths = valtest_paths[val_indices]
test_paths = valtest_paths[test_indices]

# CHECK LEAKAGE
# val in pretrain?
assert any(np.array([k in pretraining_paths for k in val_paths])) is False

# test in pretrain?
assert any(np.array([k in pretraining_paths for k in test_paths])) is False

# val in test?
assert any(np.array([k in test_paths for k in val_paths])) is False

# SAVE
json.dump({'pretrain_paths': list(pretraining_paths), 'val_paths': list(val_paths), 'test_paths': list(test_paths)}, open(SAVE_PATH, 'w'))