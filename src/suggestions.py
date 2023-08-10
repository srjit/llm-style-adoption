import os
from transformers import pipeline

import config
import utils
from constants import (
    MODEL, MODEL_SAVE_FOLDER,
    SUGGESTIONS, MAX_LENGTH, NUM_RETURN_SEQUENCES
)

cfg = config.read()
max_length = cfg.get(SUGGESTIONS, MAX_LENGTH)
num_return_seqs = cfg.get(SUGGESTIONS, NUM_RETURN_SEQUENCES)
generator = None


def get_generator():
    
    tokenizer = utils.get_tokenizer()
    
    models_dir = cfg.get(MODEL, MODEL_SAVE_FOLDER)
    version = utils.get_latest_version_of_saved_model()
    model_path = os.path.join(models_dir, version)

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
    
