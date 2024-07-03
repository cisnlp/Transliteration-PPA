# coding=utf-8
# Copyright The XTREME Benchmark Authors.
# Copyright (c) 2018, NVIDIA CORPORATION.    All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Evaluate pretrained or fine-tuned models on retrieval tasks."""
import argparse
import logging
import os

import numpy as np
import torch
from tqdm import tqdm
from transformers import (
    RobertaConfig, RobertaModel, RobertaTokenizer
)
from transformers import (
    XLMRobertaConfig, XLMRobertaModel, XLMRobertaTokenizer
)

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    "xlmr"   : (XLMRobertaConfig, XLMRobertaModel, XLMRobertaTokenizer),
    "roberta": (RobertaConfig, RobertaModel, RobertaTokenizer)
}

FALSY_STRINGS = {'off', 'false', '0'}
TRUTHY_STRINGS = {'on', 'true', '1'}


def bool_flag(s):
    """
    Parse boolean arguments from the command line.
    """
    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError("Invalid value for a boolean flag!")


def load_embeddings(embed_file):
    logger.info(' loading from {}'.format(embed_file))
    embeds = np.load(embed_file)
    return embeds


def similarity_search(x, y, normalize=False):
    if normalize:
        x = x / np.linalg.norm(x, axis=1, keepdims=True)
        y = y / np.linalg.norm(y, axis=1, keepdims=True)
    distances = np.sqrt(np.sum((x[:, np.newaxis, :] - y[np.newaxis, :, :]) ** 2, axis=2))
    prediction = np.argsort(distances, axis=0)[:10].T
    return prediction


def prepare_batch(sentences, tokenizer, model_type, device="cuda", max_length=512, langid=None,
                  use_local_max_length=True, pool_skip_special_token=False):
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token

    pad_token_id = tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0]

    batch_input_ids = []
    batch_attention_mask = []
    batch_pool_mask = []

    local_max_length = min(max([len(s) for s in sentences]) + 2, max_length)
    if use_local_max_length:
        max_length = local_max_length

    for sent in sentences:

        if len(sent) > max_length - 2:
            sent = sent[: (max_length - 2)]
        input_ids = tokenizer.convert_tokens_to_ids(
            [cls_token] + sent + [sep_token])

        padding_length = max_length - len(input_ids)
        attention_mask = [1] * len(input_ids) + [0] * padding_length
        pool_mask = [0] + [1] * (len(input_ids) - 2) + \
                    [0] * (padding_length + 1)
        input_ids = input_ids + ([pad_token_id] * padding_length)

        batch_input_ids.append(input_ids)
        batch_attention_mask.append(attention_mask)
        batch_pool_mask.append(pool_mask)

    input_ids = torch.LongTensor(batch_input_ids).to(device)
    attention_mask = torch.LongTensor(batch_attention_mask).to(device)

    if pool_skip_special_token:
        pool_mask = torch.LongTensor(batch_pool_mask).to(device)
    else:
        pool_mask = attention_mask

    if model_type == "xlm":
        langs = torch.LongTensor(
            [[langid] * max_length for _ in range(len(sentences))]).to(device)
        return {"input_ids": input_ids, "attention_mask": attention_mask, "langs": langs}, pool_mask
        # update below
    elif model_type == 'bert' or model_type == 'xlmr' or model_type == 'roberta':
        token_type_ids = torch.LongTensor(
            [[0] * max_length for _ in range(len(sentences))]).to(device)
        return {"input_ids": input_ids, "attention_mask": attention_mask, "token_type_ids": token_type_ids}, pool_mask
    elif model_type in ('bert-retrieval', 'xlmr-retrieval'):
        token_type_ids = torch.LongTensor(
            [[0] * max_length for _ in range(len(sentences))]).to(device)
        return {
            "q_input_ids": input_ids, "q_attention_mask": attention_mask, "q_token_type_ids": token_type_ids
        }, pool_mask


def tokenize_text(text_file, tok_file, tokenizer):
    if os.path.exists(tok_file):
        tok_sentences = [line.strip().split(' ') for line in open(tok_file)]
        logger.info(' -- loading from existing tok_file {}'.format(tok_file))
        return tok_sentences

    tok_sentences = []
    sents = [line.strip() for line in open(text_file)]
    with open(tok_file, 'w') as writer:
        for sent in tqdm(sents, desc='tokenize'):
            tok_sent = tokenizer.tokenize(sent)
            tok_sentences.append(tok_sent)
            writer.write(' '.join(tok_sent) + '\n')
    logger.info(' -- save tokenized sentences to {}'.format(tok_file))

    logger.info('============ Example of tokenized sentences ===============')
    for i, tok_sentence in enumerate(tok_sentences[:1]):
        logger.info('S{}: {}'.format(i, ' '.join(tok_sentence)))
    logger.info('==================================')
    return tok_sentences


def load_model(args, lang, output_hidden_states=None):
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    # if the model is not initialized from a stored checkpoint
    if not args.init_checkpoint:
        config = config_class.from_pretrained(args.model_name_or_path)
        if output_hidden_states is not None:
            config.output_hidden_states = output_hidden_states
        langid = config.lang2id.get(
            lang, config.lang2id[lang]) if args.model_type == 'xlm' else 0
        logger.info("langid={}, lang={}".format(langid, lang))
        tokenizer = tokenizer_class.from_pretrained(
            args.model_name_or_path, do_lower_case=args.do_lower_case)
        logger.info("tokenizer.pad_token={}, pad_token_id={}".format(
            tokenizer.pad_token, tokenizer.pad_token_id))
        model = model_class.from_pretrained(
            args.model_name_or_path, config=config)
    else:
        tokenizer = tokenizer_class.from_pretrained(args.init_checkpoint)
        config = config_class.from_pretrained(args.init_checkpoint)
        if output_hidden_states is not None:
            config.output_hidden_states = output_hidden_states
        langid = config.lang2id.get(
            lang, config.lang2id[lang]) if args.model_type == 'xlm' else 0
        logger.info("langid={}, lang={}".format(langid, lang))
        logger.info("tokenizer.pad_token={}, pad_token_id={}".format(
            tokenizer.pad_token, tokenizer.pad_token_id))
        model = model_class.from_pretrained(args.init_checkpoint, config=config)
    model.to(args.device)
    model.eval()
    return config, model, tokenizer, langid


def extract_embeddings(args, model, tokenizer, text_file, tok_file, embed_file, pool_type='mean',
                       langid=0):
    num_embeds = args.num_layers
    all_embed_files = ["{}_{}.npy".format(
        embed_file, i) for i in range(num_embeds)]
    if all(os.path.exists(f) for f in all_embed_files):
        logger.info('loading files from {}'.format(all_embed_files))
        return [load_embeddings(f) for f in all_embed_files]

    model.config.output_hidden_states = True

    sent_toks = tokenize_text(text_file, tok_file, tokenizer)
    max_length = max([len(s) for s in sent_toks])
    logger.info('max length of tokenized text = {}'.format(max_length))

    batch_size = args.batch_size
    num_batch = int(np.ceil(len(sent_toks) * 1.0 / batch_size))
    num_sents = len(sent_toks)

    all_embeds = [np.zeros(shape=(num_sents, args.embed_size),
                           dtype=np.float32) for _ in range(num_embeds)]
    for i in tqdm(range(num_batch), desc='Batch'):
        start_index = i * batch_size
        end_index = min((i + 1) * batch_size, num_sents)
        batch, pool_mask = prepare_batch(sent_toks[start_index: end_index],
                                         tokenizer,
                                         args.model_type,
                                         args.device,
                                         args.max_seq_length,
                                         langid=langid,
                                         pool_skip_special_token=args.pool_skip_special_token)

        with torch.no_grad():
            outputs = model(**batch)

            all_layer_outputs = outputs['hidden_states']

            # get the pool embedding
            if pool_type == 'cls':
                all_batch_embeds = cls_pool_embedding(
                    all_layer_outputs[-args.num_layers:])
            else:
                all_batch_embeds = []
                all_layer_outputs = all_layer_outputs[-args.num_layers:]
                all_batch_embeds.extend(
                    mean_pool_embedding(all_layer_outputs, pool_mask))

        for embeds, batch_embeds in zip(all_embeds, all_batch_embeds):
            embeds[start_index: end_index] = batch_embeds.cpu(
            ).numpy().astype(np.float32)
        del all_layer_outputs, outputs
        torch.cuda.empty_cache()

    if embed_file is not None:
        for file, embeds in zip(all_embed_files, all_embeds):
            logger.info('save embed {} to file {}'.format(embeds.shape, file))
            np.save(file, embeds)
    return all_embeds


def mean_pool_embedding(all_layer_outputs, masks):
    sent_embeds = []
    for embeds in all_layer_outputs:
        embeds = (embeds * masks.unsqueeze(2).float()).sum(dim=1) / \
                 masks.sum(dim=1).view(-1, 1).float()
        sent_embeds.append(embeds)
    return sent_embeds


def cls_pool_embedding(all_layer_outputs):
    sent_embeds = []
    for embeds in all_layer_outputs:
        embeds = embeds[:, 0, :]
        sent_embeds.append(embeds)
    return sent_embeds


def run(args):
    logging.basicConfig(
        handlers=[logging.FileHandler(os.path.join(args.output_dir, args.log_file)), logging.StreamHandler()],
        format='%(asctime)s - %(levelname)s - %(name)s -     %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO)
    logging.info("Input args: %r" % args)

    # Setup CUDA, GPU
    device_name = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    device = torch.device(device_name)
    args.n_gpu = torch.cuda.device_count()
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -     %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logging.info("Torch device: %s" % device_name)

    config, model, tokenizer, langid = load_model(args, args.src_language, output_hidden_states=True)
    logging.info(str(model))

    with open(os.path.join(args.output_dir, f'test_results.txt'), 'w') as result_writer:
        for src_lang in args.predict_langs:
            tgt_lang = args.src_language

            if src_lang == tgt_lang:
                continue

            data_dir = '/'.join(args.data_dir.split('/')[:-1])
            src_text_file = os.path.join(
                data_dir, 'tatoeba.{}-{}.{}'.format(src_lang, 'eng_Latn', src_lang))
            if not os.path.exists(src_text_file):
                continue

            if tgt_lang == 'eng_Latn':
                tgt_text_file = os.path.join(
                    data_dir, 'tatoeba.{}-{}.{}'.format(src_lang, tgt_lang, tgt_lang))
            else:
                tgt_text_file = os.path.join(
                    data_dir, 'tatoeba.{}-{}.{}'.format(tgt_lang, 'eng_Latn', tgt_lang))
            if not os.path.exists(tgt_text_file):
                continue

            src_tok_file = os.path.join(
                args.tokenized_dir, '{}-eng.tok.{}'.format(src_lang, src_lang))
            tgt_tok_file = os.path.join(
                args.tokenized_dir, '{}-eng.tok.{}'.format(src_lang, tgt_lang))

            all_src_embeds = extract_embeddings(
                args, model, tokenizer, src_text_file, src_tok_file, None, pool_type=args.pool_type,
                langid=langid)
            all_tgt_embeds = extract_embeddings(
                args, model, tokenizer, tgt_text_file, tgt_tok_file, None, pool_type=args.pool_type,
                langid=langid)

            for layer_idx in [args.specific_layer]:
                x, y = all_src_embeds[layer_idx], all_tgt_embeds[layer_idx]
                predictions = similarity_search(
                    x, y, normalize=(args.dist == 'cosine'))
                correct_num_top1, correct_num_top5, correct_num_top10 = 0, 0, 0
                num = 0
                for i, pred in enumerate(predictions):
                    if i in pred[:1]:
                        correct_num_top1 += 1
                    if i in pred[:5]:
                        correct_num_top5 += 1
                    if i in pred[:10]:
                        correct_num_top10 += 1
                    num += 1
                # Save results
                result_writer.write(
                    "=====================\nlanguage={}\n".format(src_lang))
                result_writer.write(
                    "Acc1 = {}\n".format(str(correct_num_top1 / num)))
                result_writer.write(
                    "Acc5 = {}\n".format(str(correct_num_top5 / num)))
                result_writer.write(
                    "Acc10 = {}\n".format(str(correct_num_top10 / num)))
                with open(os.path.join(args.output_dir, f'test_{src_lang}_predictions.txt'), 'w') as fout:
                    for p in predictions:
                        fout.write(str(p) + '\n')


def main():
    parser = argparse.ArgumentParser(description='bitext mining')
    parser.add_argument('--encoding', default='utf-8',
                        help='character encoding for input/output')
    parser.add_argument('--embed_size', type=int, default=768,
                        help='Dimensions of output embeddings')
    parser.add_argument('--pool_type', type=str, default='mean',
                        help='pooling over work embeddings')

    # Required parameters
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="The input data dir. Should contain the input files for the task.",
    )
    parser.add_argument(
        "--tokenized_dir",
        default=None,
        type=str,
        required=False,
        help="The input data dir in which the tokenized are stored (optional).",
    )
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type",
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name",
    )
    parser.add_argument(
        "--init_checkpoint",
        default=None,
        type=str,
        help="Path to pre-trained model or shortcut name selected in the list"
    )
    parser.add_argument("--src_language", type=str,
                        default="eng_Latn", help="source language.")
    parser.add_argument("--tgt_language", type=str, help="target language.")
    parser.add_argument("--batch_size", type=int,
                        default=100, help="batch size.")
    parser.add_argument("--tgt_text_file", type=str,
                        default=None, help="tgt_text_file.")
    parser.add_argument("--src_text_file", type=str,
                        default=None, help="src_text_file.")
    parser.add_argument("--tgt_tok_file", type=str,
                        default=None, help="tgt_tok_file")
    parser.add_argument("--src_tok_file", type=str,
                        default=None, help="src_tok_file")
    parser.add_argument("--num_layers", type=int,
                        default=12, help="num layers")
    parser.add_argument("--pool_skip_special_token", action="store_true")
    parser.add_argument("--dist", type=str, default='cosine')

    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory",
    )

    parser.add_argument(
        "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model."
    )
    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
             "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument("--no_cuda", action="store_true",
                        help="Avoid using CUDA when available")
    parser.add_argument("--specific_layer", type=int,
                        default=7, help="use specific layer")

    args = parser.parse_args()

    args.predict_langs = []
    with open('/dss/dsshome1/06/di35ver/evaluation/retrieval/tatoeba_lang_list.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            args.predict_langs.append(line.strip().split('\t')[0])

    # Evaluate all checkpoints
    assert args.init_checkpoint is not None

    checkpoints = [f for f in os.listdir(args.init_checkpoint) if 'checkpoint' in f]
    # checkpoints.insert(0, "checkpoint-0")
    main_model_dir = args.init_checkpoint
    main_out_dir = args.output_dir

    for checkpoint in checkpoints:
        print(f"Current checkpoint: {checkpoint}...")
        args.output_dir = main_out_dir + f"/{main_model_dir.split('/')[-1]}"
        args.output_dir += f"/{checkpoint}_{args.src_language}"

        if 'checkpoint-0' in checkpoint:
            args.init_checkpoint = 'cis-lmu/glot500-base'
        else:
            args.init_checkpoint = main_model_dir + f"/{checkpoint}"

        if not os.path.exists(args.tokenized_dir):
            os.makedirs(args.tokenized_dir)

        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        else:
            continue
        args.log_file = args.output_dir + '/train.log'

        run(args)


if __name__ == "__main__":
    main()
