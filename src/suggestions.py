import os
import pandas as pd
from transformers import pipeline
from transformers import OpenAIGPTConfig, GPT2LMHeadModel
from datasets import GPT2Dataset

import network
import config
import utils
from constants import (
    MODEL, MODEL_SAVE_FOLDER,
    SUGGESTIONS, MAX_LENGTH, NUM_RETURN_SEQUENCES,
    DATA, NOTES_FOLDER, USED_NOTES_FOLDER,
    MAC_META_FILE
)

cfg = config.read()
max_length = int(cfg.get(SUGGESTIONS, MAX_LENGTH))
num_return_seqs = int(cfg.get(SUGGESTIONS, NUM_RETURN_SEQUENCES))
generator = None


def get_generator():
    
    tokenizer = utils.get_tokenizer()
    models_dir = cfg.get(MODEL, MODEL_SAVE_FOLDER)
    version = utils.get_latest_version_of_saved_model()
    model_path = os.path.join(models_dir, str(version))

    print("Using retrained model version: ", str(version))
    generator = pipeline(task='text-generation',
                         model=model_path,
                         tokenizer=tokenizer,
                         framework='pt')
    return generator


def get(text):

    global generator
    
    if generator is None:
        generator = get_generator()

    return generator(text,
                     max_length=max_length,
                     num_return_sequences=num_return_seqs)


def get_previous_model_and_tokenizer():

    tokenizer = utils.get_tokenizer()
    models_dir = cfg.get(MODEL, MODEL_SAVE_FOLDER)
    version = utils.get_latest_version_of_saved_model()

    if version > 0:
        model_path = os.path.join(models_dir, str(version))
        model = GPT2LMHeadModel.from_pretrained(model_path)
    else:
       _, model = utils.get_configuration_and_model()
    return model, tokenizer
    

def get_notes():
    
    texts = []
    notes_folder = cfg.get(DATA, NOTES_FOLDER)

    if os.path.exists(notes_folder):
    
        notes = [os.path.join(notes_folder, txt) for txt
                 in os.listdir(notes_folder)]

        if MAC_META_FILE in notes:
            notes.remove(MAC_META_FILE)

        for note in notes:
            with open(note, "r") as f:
                text = f.read()
                notes = text.split("\n \n")
            texts = texts + notes
    
    return texts


def retrain():

    notes = get_notes()

    model, tokenizer = get_previous_model_and_tokenizer()
    dataset = GPT2Dataset(notes, tokenizer, max_length=768)

    import ipdb
    ipdb.set_trace()
    
    network.retrain(model, dataset, tokenizer, validate=True)

    utils.save(model, tokenizer)
    version = utils.get_latest_version_of_saved_model()
    print(f"Retrained and saved new model with {version}")
