
import warnings
warnings.filterwarnings('ignore')

import os
import config
import utils

import pandas as pd
import seaborn as sns
import numpy as np
import random

import matplotlib.pyplot as plt

import torch
from torch.utils.data import random_split, RandomSampler, SequentialSampler
torch.manual_seed(42)

from transformers import GPT2LMHeadModel,  GPT2Tokenizer, GPT2Config, GPT2LMHeadModel
from transformers import AdamW, get_linear_schedule_with_warmup

from datasets import GPT2Dataset


notes_directory = "notes/"


cfg = config.read()
run_type = cfg.get("run", "type")

notes = utils.get_sample_notes(run_type)
tokenizer = utils.get_tokenizer()
dataset = GPT2Dataset(notes, tokenizer, max_length=768)

import ipdb
ipdb.set_trace()
