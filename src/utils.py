import time
import os
import pandas as pd
import matplotlib.pyplot as plt
import viz

import random
import torch
from torch.utils.data import DataLoader, \
    random_split, RandomSampler, SequentialSampler

from datetime import datetime, timedelta
import config

from transformers import GPT2LMHeadModel, \
    GPT2Tokenizer, GPT2Config
from transformers import AdamW, get_linear_schedule_with_warmup

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
    existing_versions = os.listdir(model_save_folder)

    if ".DS_Store" in existing_versions:
        existing_versions.remove(".DS_Store")

    if len(existing_versions) == 0:
        return 0
    return max([int(x) for x in existing_versions])
    

def get_sample_notes(run_type):

    notes_directory = cfg.get(DATA, SAMPLE_DATA_FOLDER)
    sonnet_f = cfg.get(DATA, SAMPLE_DATA)

    df = None
    if run_type == TEST:
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


def get_configuration_and_model():

    model_type = cfg.get(MODEL, TYPE)
    configuration, model = None, None

    if model_type == GPT2:

        configuration = GPT2Config.from_pretrained(GPT2,
                                                   output_hidden_states=False)
        model = GPT2LMHeadModel.from_pretrained(GPT2,
                                                config=configuration)

    return configuration, model


def format_time(elapsed):
    return str(timedelta(seconds=int(round((elapsed)))))


def save(model, tokenizer_):

    version = get_latest_version_of_saved_model() + 1
    output_dir = os.path.join(cfg.get(MODEL, MODEL_SAVE_FOLDER), version)

    # Create output directory if needed
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("Saving model to %s" % output_dir)

    # Save a trained model, configuration and tokenizer using `save_pretrained()`.
    # They can then be reloaded using `from_pretrained()`
    model_to_save = model.module if hasattr(model, 'module') else model
    model_to_save.save_pretrained(output_dir)
    tokenizer_.save_pretrained(output_dir)


def plot_training_status(stats):

    epochs = list(range(len(stats)))
    
    # Plot the learning curve.
    f, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(8, 5))
    viz.plot(epochs,
             stats['Training Loss'],
             ax=ax1,
             ls="--",
             format_y=False,
             make_x_string=False,
             color="gray",
             label="Training")
    viz.plot(epochs,
             stats['Valid. Loss'], ax=ax1,
             xlabel="Epoch",
             ylabel="Loss",
             format_y=False,
             make_x_string=False,
             ls='--',
             color="k",
             label="Validation")

    # Label the plot.
    plt.legend()

    graphs_root_folder = "../plots"
    fname = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    savepath = os.path.join(graphs_root_folder,  fname + ".png")
    plt.savefig(savepath)
    

def retrain(model, dataset, tokenizer_, validate=True, device='cpu'):

    epochs = int(cfg.get(MODEL, EPOCHS))
    learning_rate = float(cfg.get(MODEL, LEARNING_RATE))
    warmup_steps = float(cfg.get(MODEL, WARMUP_STEPS))
    epsilon = float(cfg.get(MODEL, EPSILON))
    sample_every = int(cfg.get(MODEL, SAMPLE_EVERY))
    batch_size = int(cfg.get(MODEL, BATCH_SIZE))

    optimizer = AdamW(model.parameters(),
                      lr=learning_rate,
                      eps=epsilon)
    
    total_t0 = time.time()
    training_stats = []
    model = model.to(device)

    train_dataloader = None
    
    if validate:

        train_size = int(0.9 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset,
                                                  [train_size, val_size])

        train_dataloader = DataLoader(
            train_dataset,  # The training samples.
            sampler=RandomSampler(train_dataset), # Select batches randomly
            batch_size=batch_size # Trains with this batch size.
        )

        validation_dataloader = DataLoader(
            val_dataset, # The validation samples.
            sampler=SequentialSampler(val_dataset), # Pull out batches sequentially.
            batch_size = batch_size # Evaluate with this batch size.
        )

        total_steps = len(train_dataloader) * epochs
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=warmup_steps,
                                                    num_training_steps=total_steps)

    for epoch_i in range(0, epochs):

        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')

        t0 = time.time()
        total_train_loss = 0

        model.train()
        for step, batch in enumerate(train_dataloader):

            b_input_ids = batch[0].to(device)
            b_labels = batch[0].to(device)
            b_masks = batch[1].to(device)

            model.zero_grad()        
            outputs = model(b_input_ids,
                            labels=b_labels,
                            attention_mask=b_masks,
                            token_type_ids=None)

            loss = outputs[0]
            batch_loss = loss.item()
            total_train_loss += batch_loss

            if step % sample_every == 0 and not step == 0:

                elapsed = format_time(time.time() - t0)
                print('  Batch {:>5,}  of  {:>5,}. Loss: {:>5,}.  \
                Elapsed: {:}.'.format(step,
                                      len(train_dataloader),
                                      batch_loss,
                                      elapsed))

                model.eval()
                sample_outputs = model.generate(
                    bos_token_id=random.randint(1, 30000),
                    do_sample=True,
                    top_k=50,
                    max_length=200,
                    top_p=0.95,
                    num_return_sequences=1
                )
                for i, sample_output in enumerate(sample_outputs):
                    print("{}: {}".format(i,
                                          tokenizer_.decode(sample_output,
                                                            skip_special_tokens=True)))
            
                model.train()

            loss.backward()
            optimizer.step()
            scheduler.step()

        # Calculate the average loss over all of the batches.
        avg_train_loss = total_train_loss / len(train_dataloader)       
        # Measure how long this epoch took.
        training_time = format_time(time.time() - t0)

        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epoch took: {:}".format(training_time))
            
        # ========================================
        #               Validation
        # ========================================

        print("")
        print("Running Validation...")

        t0 = time.time()
        model.eval()
        total_eval_loss = 0

        # Evaluate data for one epoch
        for batch in validation_dataloader:
        
            b_input_ids = batch[0].to(device)
            b_labels = batch[0].to(device)
            b_masks = batch[1].to(device)
            with torch.no_grad():       

                outputs = model(b_input_ids,
#                            token_type_ids=None,
                                attention_mask=b_masks,
                                labels=b_labels)
          
                loss = outputs[0] 
            batch_loss = loss.item()
            total_eval_loss += batch_loss

        avg_val_loss = total_eval_loss / len(validation_dataloader)
        validation_time = format_time(time.time() - t0)

        print("  Validation Loss: {0:.2f}".format(avg_val_loss))
        print("  Validation took: {:}".format(validation_time))

        # Record all statistics from this epoch.
        training_stats.append(
            {
                'epoch': epoch_i + 1,
                'Training Loss': avg_train_loss,
                'Valid. Loss': avg_val_loss,
                'Training Time': training_time,
                'Validation Time': validation_time
            }
        )

    print("")
    print("Training complete!")
    print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))

    df_stats = pd.DataFrame(data=training_stats)
    df_stats = df_stats.set_index('epoch')
    plot_training_status(df_stats)

    return model
