import time
import pandas as pd

import torch
from torch.utils.data import DataLoader, \
    random_split, RandomSampler, SequentialSampler

import datetime
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
    
    return df.note


def get_tokenizer():

    model_type = cfg.get("model", "type")
    tokenizer = None

    if model_type == "gpt2":

        # Load the GPT tokenizer.
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2',
                                                  bos_token='<|startoftext|>',
                                                  eos_token='<|endoftext|>',
                                                  pad_token='<|pad|>')
    else:
        pass

    return tokenizer


def get_configuration_and_model():

    model_type = cfg.get("model", "type")
    configuration, model = None, None

    if model_type == "gpt2":

        configuration = GPT2Config.from_pretrained('gpt2', output_hidden_states=False)
        model = GPT2LMHeadModel.from_pretrained("gpt2", config=configuration)

    return configuration, model



def format_time(elapsed):
    return str(datetime.timedelta(seconds=int(round((elapsed)))))
    

def retrain(model, dataset, validate=True, device='cpu'):

    epochs = 5
    learning_rate = 5e-4
    warmup_steps = 1e2
    epsilon = 1e-8
    sample_every = 100
    batch_size = 2

    optimizer = AdamW(model.parameters(),
                  lr = learning_rate,
                  eps = epsilon
                )
    
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
            sampler = RandomSampler(train_dataset), # Select batches randomly
            batch_size = batch_size # Trains with this batch size.
        )

        validation_dataloader = DataLoader(
            val_dataset, # The validation samples.
            sampler = SequentialSampler(val_dataset), # Pull out batches sequentially.
            batch_size = batch_size # Evaluate with this batch size.
        )

        total_steps = len(train_dataloader) * epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = warmup_steps, 
                                            num_training_steps = total_steps)

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
            outputs = model( b_input_ids,
                             labels=b_labels, 
                             attention_mask = b_masks,
                             token_type_ids=None
                            )

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
                    bos_token_id=random.randint(1,30000),
                    do_sample=True,   
                    top_k=50, 
                    max_length = 200,
                    top_p=0.95, 
                    num_return_sequences=1
                )
                for i, sample_output in enumerate(sample_outputs):
                    print("{}: {}".format(i,
                                          tokenizer.decode(sample_output,
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
        nb_eval_steps = 0

        # Evaluate data for one epoch
        for batch in validation_dataloader:
        
            b_input_ids = batch[0].to(device)
            b_labels = batch[0].to(device)
            b_masks = batch[1].to(device)
            with torch.no_grad():        

                outputs  = model(b_input_ids, 
#                            token_type_ids=None, 
                                 attention_mask = b_masks,
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
            
