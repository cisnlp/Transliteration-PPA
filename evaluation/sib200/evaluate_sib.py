import argparse
import os
import random
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score

from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from transformers import (
    AdamW,
    get_linear_schedule_with_warmup,
    XLMRobertaConfig,
    XLMRobertaTokenizer,
    XLMRobertaForSequenceClassification,
    RobertaConfig,
    RobertaTokenizer,
    RobertaForSequenceClassification
)


MODEL_CLASSES = {
    "xlmr"   : (XLMRobertaConfig, XLMRobertaForSequenceClassification, XLMRobertaTokenizer),
    "roberta": (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer)
}


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


def read_data(file):
    print('Preparing dataset...')
    df = pd.read_csv(file, delimiter='\t', on_bad_lines='skip', engine='python')
    print('Number of sentences: {:,}\n'.format(df.shape[0]))
    df['category'].replace({
        'science/technology': int(0), 'travel': int(1), 'politics': int(2),
        'sports'         : int(3), 'health': int(4), 'entertainment': int(5), 'geography': int(6)
    }, inplace=True)

    sentences = df.text.values
    labels = df.category.values

    return sentences, labels


def encode(tokenizer, sentences, labels, max_length):
    print('Loading model tokenizer...')
    input_ids = []
    attention_masks = []
    for sent in sentences:
        sent = str(sent)
        encoded_dict = tokenizer.encode_plus(
            sent,  # Sentence to encode.
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            truncation=True,
            max_length=max_length,  # Pad & truncate all sentences.
            pad_to_max_length=True,
            return_attention_mask=True,  # Construct attn. masks.
            return_tensors='pt',  # Return pytorch tensors.
        )
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)

    labels = torch.tensor(labels)

    return input_ids, attention_masks, labels


def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def calc_f1_score(preds, labels):
    pre = np.argmax(preds, axis=1)
    f1 = f1_score(labels, pre, average='macro')
    return f1


def train(args, model, optimizer, scheduler, train_dataloader, validation_dataloader):
    training_stats = []
    best_f1 = -1.0
    epochs = args.epochs
    device = args.device
    accum_iter = args.accum_iter
    output_dir = args.output_dir

    for epoch_i in range(0, epochs):
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')

        total_train_loss = 0
        total_train_f1 = 0

        # training
        model.train()
        for step, batch in enumerate(train_dataloader):
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            model.zero_grad()

            with torch.set_grad_enabled(True):
                (loss, logits) = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels, return_dict=False)

                logits = logits.detach().cpu().numpy()
                b_label_ids = b_labels.to('cpu').numpy()

                total_train_loss += loss.item()
                total_train_f1 += calc_f1_score(logits, b_label_ids)
                loss = loss / accum_iter
                loss.backward()
                if ((step + 1) % accum_iter == 0) or (step + 1 == len(train_dataloader)):
                    optimizer.step()
                    scheduler.step()

        avg_train_loss = total_train_loss / len(train_dataloader)
        avg_train_f1 = total_train_f1 / len(train_dataloader)
        print('train_f1: ', avg_train_f1)

        # evaluation
        model.eval()

        label_ids = []
        predictions = []

        total_eval_loss = 0
        for batch in validation_dataloader:
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            with torch.no_grad():
                (loss, logits_current) = model(b_input_ids,
                                               attention_mask=b_input_mask,
                                               labels=b_labels,
                                               return_dict=False)
            total_eval_loss += loss.item()

            predictions += np.argmax(logits_current.detach().cpu().numpy(), axis=1).reshape(-1).tolist()
            label_ids += b_labels.to('cpu').numpy().reshape(-1).tolist()

        # f1 score and accuracy of the whole validation set
        total_eval_f1_score = f1_score(label_ids, predictions, average='macro')
        total_eval_accuracy = accuracy_score(label_ids, predictions)

        avg_val_accuracy = total_eval_accuracy
        avg_val_f1 = total_eval_f1_score
        avg_val_loss = total_eval_loss / len(validation_dataloader)
        print('avg_val_accuracy:', avg_val_accuracy, 'avg_val_f1: ', avg_val_f1, 'avg_val_loss', avg_val_loss)
        if avg_val_f1 > best_f1:
            best_f1 = avg_val_f1
            torch.save(model, f"{output_dir}/pytorch_model.bin")
            print('Better validation f1!')
        else:
            pass
        print('val_f1: ', avg_val_f1)

        training_stats.append(
            {
                'epoch'     : epoch_i + 1,
                'Train Loss': avg_train_loss,
                'Val Loss'  : avg_val_loss,
                'Val Accu.' : avg_val_accuracy,
                'val F1 '   : avg_val_f1,
            }
        )

        print("")
        print("Training complete!")
    pd.set_option('display.precision', 2)
    df_stats = pd.DataFrame(data=training_stats)
    df_stats = df_stats.set_index('epoch')
    print(df_stats)


def test(args, model, test_dataloader):
    device = args.device
    model.eval()

    predictions, true_labels = [], []

    # Predict
    result = {}
    for batch in test_dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch

        with torch.no_grad():
            outputs = model(b_input_ids, attention_mask=b_input_mask)

        logits = outputs[0]

        predictions += np.argmax(logits.detach().cpu().numpy(), axis=1).reshape(-1).tolist()
        true_labels += b_labels.to('cpu').numpy().reshape(-1).tolist()

    result['accuracy'] = accuracy_score(true_labels, predictions)
    result['f1'] = f1_score(true_labels, predictions, average='macro')

    return result


def run(args):
    num_labels = args.num_labels
    args.model_type = args.model_type.lower()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    max_seq_length = args.max_length

    if not args.init_checkpoint:
        config = config_class.from_pretrained(args.model_name_or_path, num_labels=num_labels)
        config.output_attentions = False
        config.output_hidden_states = False
        tokenizer = tokenizer_class.from_pretrained(
            args.model_name_or_path, do_lower_case=args.do_lower_case)
        model = model_class.from_pretrained(
            args.model_name_or_path, config=config)
    else:
        tokenizer = tokenizer_class.from_pretrained(args.init_checkpoint, do_lower_case=args.do_lower_case)
        config = config_class.from_pretrained(args.init_checkpoint)
        config.num_labels = num_labels
        config.output_attentions = False
        config.output_hidden_states = False
        model = model_class.from_pretrained(args.init_checkpoint, config=config)
    print(model)
    model.to(args.device)

    set_seed(args)

    if args.do_train:
        train_file = f"{args.data_dir}/{args.source_language}/train.tsv"
        train_sentences, train_label = read_data(train_file)
        train_input_ids, train_attention_masks, train_labels = encode(
            tokenizer, train_sentences, train_label, max_seq_length)
        train_dataset = TensorDataset(train_input_ids, train_attention_masks, train_labels)

        dev_file = f"{args.data_dir}/{args.source_language}/dev.tsv"
        dev_verse, dev_label = read_data(dev_file)
        dev_input_ids, dev_attention_masks, dev_labels = encode(tokenizer, dev_verse, dev_label, max_seq_length)
        val_dataset = TensorDataset(dev_input_ids, dev_attention_masks, dev_labels)

        train_dataloader = DataLoader(
            train_dataset,  # The training samples.
            sampler=RandomSampler(train_dataset),  # Select batches randomly
            batch_size=args.train_batch_size,  # Trains with this batch size.
        )

        validation_dataloader = DataLoader(
            val_dataset,  # The validation samples.
            sampler=SequentialSampler(val_dataset),  # Pull out batches sequentially.
            batch_size=args.train_batch_size  # Evaluate with this batch size.
        )

        learning_rate = 1e-5
        optimizer = AdamW(model.parameters(),
                          lr=learning_rate,
                          eps=1e-8
                          )

        total_steps = len(train_dataloader) * args.epochs
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=0,
                                                    num_training_steps=total_steps)
        train(args, model, optimizer, scheduler, train_dataloader, validation_dataloader)

    best_checkpoint = f"{args.output_dir}/pytorch_model.bin"
    model = torch.load(best_checkpoint)
    model.to(args.device)
    output_test_results_file = os.path.join(args.output_dir, "test_results.txt")

    # performing testing for all languages
    with open(output_test_results_file, "w") as result_writer:
        for lang in args.predict_langs:
            print('--------')
            print(f"Testing on {lang} ...")
            test_file = f"{args.data_dir}/{lang}/test.tsv"
            test_sentences, test_label = read_data(test_file)
            test_input_ids, test_attention_masks, test_labels = encode(
                tokenizer, test_sentences, test_label, max_seq_length)
            test_dataset = TensorDataset(test_input_ids, test_attention_masks, test_labels)

            test_dataloader = DataLoader(
                test_dataset,
                sampler=SequentialSampler(test_dataset),
                batch_size=args.test_batch_size,
            )

            result = test(args, model, test_dataloader)

            # Save results
            result_writer.write("=====================\nlanguage={}\n".format(lang))
            for key in sorted(result.keys()):
                result_writer.write("{} = {}\n".format(key, str(result[key])))
                print("{} = {}\n".format(key, str(result[key])))


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type")
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--init_checkpoint", default=None, type=str)

    ## training parameters
    parser.add_argument("--train_batch_size", default=32, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--test_batch_size", default=32, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--accum_iter", default=1, type=int)
    parser.add_argument("--num_labels", default=7, type=int)
    parser.add_argument("--do_lower_case", action="store_true",
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed for initialization")
    # Run for multiple seeds
    parser.add_argument("--nr_of_seeds", type=int, default=1,
                        help="Number of evaluation to run")

    parser.add_argument("--data_dir", type=str, default=None)
    # Specify the different source languages to evaluate
    parser.add_argument('--source_language', type=str, default='eng_Latn')
    # Specify the boundary of the checkpoints to evaluate
    parser.add_argument("--max_checkpoint_num", type=int, default=10000)
    parser.add_argument('--min_checkpoint_num', type=int, default=0)
    parser.add_argument('--max_length', type=int, default=160)

    args = parser.parse_args()

    args.predict_langs = os.listdir(args.data_dir)

    # Evaluate all checkpoints
    model_dir = args.init_checkpoint
    main_output_dir = args.output_dir + f"/{model_dir.split('/')[-1]}"

    checkpoints = [f for f in os.listdir(model_dir) if 'checkpoint' in f]
    checkpoints = sorted(checkpoints, key=lambda num: int(num.split('-')[1]))
    checkpoints = [checkpoint for checkpoint in checkpoints
                   if args.min_checkpoint_num <= int(checkpoint.split('-')[1]) <= args.max_checkpoint_num]
    # checkpoints.insert(0, "checkpoint-0")
    print(checkpoints)

    run_combos = [(checkpoint, lang, seed_nr)
                  for checkpoint in checkpoints
                  for lang in args.source_language.split(',')
                  for seed_nr in range(args.nr_of_seeds)]
    default_seed = args.seed
    for checkpoint, lang, seed_nr in run_combos:
        print(f"Running {checkpoint} on {lang} with seed {seed_nr}")
        seed = default_seed + seed_nr

        args.output_dir = main_output_dir + f"/{checkpoint}_{lang}_{seed}"
        args.source_language = lang
        args.seed = seed

        # update the init_checkpoint
        if 'checkpoint-0' in checkpoint:
            args.init_checkpoint = 'cis-lmu/glot500-base'
        else:
            args.init_checkpoint = model_dir + f"/{checkpoint}"

        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
            args.do_train = True
        else:
            continue

        args.log_file = args.output_dir + '/train.log'

        run(args)


if __name__ == "__main__":
    main()
