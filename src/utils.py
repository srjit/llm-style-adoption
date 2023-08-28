import os
import logging
import pandas as pd
import matplotlib.pyplot as plt
import viz

import random
import torch
from datetime import datetime
import config

from transformers import GPT2LMHeadModel, \
    GPT2Tokenizer, GPT2Config


from constants import (
    DATA, SAMPLE_DATA_FOLDER, SAMPLE_DATA,
    TEST,
    MODEL, TYPE, GPT2, EPOCHS,
    TRAIN_FRACTION, LEARNING_RATE, WARMUP_STEPS, EPSILON,
    SAMPLE_EVERY, BATCH_SIZE,
    MODEL_SAVE_FOLDER
)

cfg = config.read()


def get_latest_version_of_saved_model():

    model_save_folder = cfg.get(MODEL, MODEL_SAVE_FOLDER)

    if os.path.exists(model_save_folder):
        existing_versions = os.listdir(model_save_folder)

        if ".DS_Store" in existing_versions:
            existing_versions.remove(".DS_Store")

        if len(existing_versions) == 0:
            return 0

        return max([int(x) for x in existing_versions])
        
    return 0
    

def get_sample_notes(run_type):

    notes_directory = cfg.get(DATA, SAMPLE_DATA_FOLDER)
    sonnet_f = cfg.get(DATA, SAMPLE_DATA)
    df = None
    if run_type == TEST:
        with open(sonnet_f) as f:
            txt = f.read()
        sonnets = txt.split("\n\n")[:100]
        # for i, sonnet in enumerate(sonnets):
        #     with open(notes_directory + "/" + str(i) + ".txt", "w") as f:
                # f.write(sonnet)

        print(f"Number of notes for training: {len(sonnets)}")
        df = pd.DataFrame(sonnets)[2:]
        df.columns = ["note"]
        df.dropna(inplace=True)
    else:
        pass
    
    return df.note


def get_tokenizer():

    model_type = cfg.get(MODEL, TYPE)
    tokenizer = None
    if model_type == GPT2:

        # Load the GPT tokenizer.
        tokenizer = GPT2Tokenizer.from_pretrained(GPT2,
                                                  bos_token='<|startoftext|>',
                                                  eos_token='<|endoftext|>',
                                                  pad_token='<|pad|>')
    else:
        pass

    return tokenizer


def get_configuration_and_default_model():

    model_type = cfg.get(MODEL, TYPE)
    configuration, model = None, None

    if model_type == GPT2:

        configuration = GPT2Config.from_pretrained(GPT2,
                                                   output_hidden_states=False)
        model = GPT2LMHeadModel.from_pretrained(GPT2,
                                                config=configuration)

    return configuration, model



def save(model, tokenizer_):

    version = get_latest_version_of_saved_model() + 1
    output_dir = os.path.join(cfg.get(MODEL, MODEL_SAVE_FOLDER), str(version))

    # Create output directory if needed
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("Saving model to %s" % output_dir)

    # Save a trained model, configuration and tokenizer
    # using `save_pretrained()`.
    # They can then be reloaded using `from_pretrained()`
    model_to_save = model.module if hasattr(model, 'module') else model
    model_to_save.save_pretrained(output_dir)
    tokenizer_.save_pretrained(output_dir)

