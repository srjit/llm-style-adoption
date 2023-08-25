import warnings

import config
import utils
import network

import numpy as np
import random
import torch
import suggestions

from datasets import GPT2Dataset
from constants import (
    CPU,
    RUN, TYPE,
    TEST, PRODUCTION
)

torch.manual_seed(42)

device = torch.device(CPU)
seed_val = 42

random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)

warnings.filterwarnings('ignore')

cfg = config.read()
run_type = cfg.get(RUN, TYPE)

notes = utils.get_sample_notes(run_type)
#notes = suggestions.get_notes()

import ipdb
ipdb.set_trace()

tokenizer = utils.get_tokenizer()
dataset = GPT2Dataset(notes, tokenizer, max_length=768)

# Split into training and validation sets
configuration, model = utils.get_configuration_and_model()
model.resize_token_embeddings(len(tokenizer))
if run_type == TEST:
    network.retrain(model, dataset, tokenizer)
elif run_type == PRODUCTION:
    network.retrain(model, dataset, tokenizer, False)

utils.save(model, tokenizer)

version = utils.get_latest_version_of_saved_model()
print(version)

    
