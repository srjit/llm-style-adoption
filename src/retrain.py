
import warnings
warnings.filterwarnings('ignore')

import os
import config
import utils

import pandas as pd
import numpy as np
import random
import torch
from torch.utils.data import random_split, RandomSampler, SequentialSampler
torch.manual_seed(42)

from transformers import GPT2LMHeadModel,  GPT2LMHeadModel
from transformers import AdamW, get_linear_schedule_with_warmup

from datasets import GPT2Dataset

device = torch.device("cpu")
seed_val = 42

random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)


cfg = config.read()
run_type = cfg.get("run", "type")

notes = utils.get_sample_notes(run_type)

tokenizer = utils.get_tokenizer()
dataset = GPT2Dataset(notes, tokenizer, max_length=768)

# Split into training and validation sets

configuration, model = utils.get_configuration_and_model()
model.resize_token_embeddings(len(tokenizer))
if run_type == "test":
    utils.retrain(model, dataset)
else:
    utils.retrain(model, dataset, False)

import ipdb
ipdb.set_trace()
