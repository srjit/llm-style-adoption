import pandas as pd

import config

from transformers import GPT2LMHeadModel, \
    GPT2Tokenizer, GPT2Config, GPT2LMHeadModel
from transformers import AdamW, get_linear_schedule_with_warmup

cfg = config.read()


def get_sample_notes(run_type):

    notes_directory = cfg.get("data", "sample_data_folder")
    sonnet_f = cfg.get("data", "sonnet_f")

    df = None
    if run_type == "test":
        with open(sonnet_f) as f:
            txt = f.read()

        sonnets = txt.split("\n\n")
        for i, sonnet in enumerate(sonnets):
            with open(notes_directory + "/" + str(i) + ".txt", "w") as f:
                f.write(sonnet)

        df = pd.DataFrame(sonnets)[2:]
        df.columns = ["note"]
        df.dropna(inplace=True)

    else:
        pass
    
    return df


def get_tokenizer():

    tokenizer_type = cfg.get("model", "tokenizer")
    tokenizer = None

    if tokenizer_type == "gpt2":

        # Load the GPT tokenizer.
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2',
                                                  bos_token='<|startoftext|>',
                                                  eos_token='<|endoftext|>',
                                                  pad_token='<|pad|>')
    else:
        pass

    return tokenizer
