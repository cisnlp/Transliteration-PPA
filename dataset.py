from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import PreTrainedTokenizerBase


def _tensorize_batch(examples: List[Union[torch.Tensor, np.ndarray]],
                     padding_value, pad_to_multiple_of: Optional[int]) -> torch.Tensor:
    examples = [torch.tensor(ex) if not isinstance(ex, torch.Tensor) else ex for ex in examples]
    if pad_to_multiple_of is None:
        return pad_sequence(examples, batch_first=True, padding_value=padding_value)
    else:
        max_sequence_length = max([ex.shape[0] for ex in examples])
        padded_length = ((max_sequence_length + pad_to_multiple_of - 1) // pad_to_multiple_of) * pad_to_multiple_of
        padded_examples = [torch.nn.functional.pad(ex, (0, padded_length - ex.size(0)), value=padding_value) for ex in
                           examples]
        return torch.stack(padded_examples, dim=0)


@dataclass
class DataCollatorForTransliterationModeling:
    tokenizer: PreTrainedTokenizerBase
    max_seq_length: int
    mlm_probability: float = 0.15
    pad_to_multiple_of: Optional[int] = None
    use_tlm: bool = False
    reset_position_ids: bool = False

    def __call__(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        input_ids_1 = [e['input_ids_1'] for e in examples]
        attention_mask_1 = [e['attention_mask_1'] for e in examples]
        special_tokens_mask_1 = [e['special_tokens_mask_1'] for e in examples]
        token_type_ids_1 = [e['token_type_ids_1'] for e in examples]

        input_ids_2 = [e['input_ids_2'] for e in examples]
        attention_mask_2 = [e['attention_mask_2'] for e in examples]
        special_tokens_mask_2 = [e['special_tokens_mask_2'] for e in examples]
        token_type_ids_2 = [e['token_type_ids_2'] for e in examples]

        input_ids_tlm, attention_mask_tlm, special_tokens_mask_tlm, token_type_ids_tlm = None, None, None, None
        position_ids_tlm = None
        seq_2_start = 0 if self.reset_position_ids else 1
        if self.use_tlm:
            orders = np.random.choice([0, 1], len(examples))
            input_ids_tlm = [
                (input_ids_1[i] + input_ids_2[i][seq_2_start:] if order else
                 input_ids_2[i] + input_ids_1[i][seq_2_start:])[:self.max_seq_length]
                for i, order in enumerate(orders)
            ]
            attention_mask_tlm = [
                (attention_mask_1[i] + attention_mask_2[i][seq_2_start:] if order else
                 attention_mask_2[i] + attention_mask_1[i][seq_2_start:])[:self.max_seq_length]
                for i, order in enumerate(orders)
            ]
            special_tokens_mask_tlm = [
                (special_tokens_mask_1[i] + special_tokens_mask_2[i][seq_2_start:] if order
                 else special_tokens_mask_2[i] + special_tokens_mask_1[i][seq_2_start:])[:self.max_seq_length]
                for i, order in enumerate(orders)
            ]
            token_type_ids_tlm = [
                (token_type_ids_1[i] + token_type_ids_2[i][seq_2_start:] if order else
                 token_type_ids_2[i] + token_type_ids_1[i][seq_2_start:])[:self.max_seq_length]
                for i, order in enumerate(orders)
            ]

            if self.reset_position_ids:
                start_position = self.tokenizer.pad_token_id + 1

                position_ids_1 = [
                    [start_position + i for i in range(len(seq_1))] for seq_1 in input_ids_1
                ]
                position_ids_2 = [
                    [start_position + i for i in range(len(seq_2))] for seq_2 in input_ids_2
                ]
                position_ids_tlm = [
                    (position_ids_1[i] + position_ids_2[i] if order else
                     position_ids_2[i] + position_ids_1[i])[:self.max_seq_length]
                    for i, order in enumerate(orders)
                ]

        input_ids_1 = _tensorize_batch(input_ids_1, padding_value=self.tokenizer.pad_token_id,
                                       pad_to_multiple_of=self.pad_to_multiple_of)
        attention_mask_1 = _tensorize_batch(attention_mask_1, padding_value=0,
                                            pad_to_multiple_of=self.pad_to_multiple_of)
        special_tokens_mask_1 = _tensorize_batch(special_tokens_mask_1, padding_value=1,
                                                 pad_to_multiple_of=self.pad_to_multiple_of)
        token_type_ids_1 = _tensorize_batch(token_type_ids_1, padding_value=0,
                                            pad_to_multiple_of=self.pad_to_multiple_of)

        input_ids_2 = _tensorize_batch(input_ids_2, padding_value=self.tokenizer.pad_token_id,
                                       pad_to_multiple_of=self.pad_to_multiple_of)
        attention_mask_2 = _tensorize_batch(attention_mask_2, padding_value=0,
                                            pad_to_multiple_of=self.pad_to_multiple_of)
        special_tokens_mask_2 = _tensorize_batch(special_tokens_mask_2, padding_value=1,
                                                 pad_to_multiple_of=self.pad_to_multiple_of)
        token_type_ids_2 = _tensorize_batch(token_type_ids_2, padding_value=0,
                                            pad_to_multiple_of=self.pad_to_multiple_of)

        if self.use_tlm:
            input_ids_tlm = _tensorize_batch(input_ids_tlm, padding_value=self.tokenizer.pad_token_id,
                                             pad_to_multiple_of=self.pad_to_multiple_of)
            attention_mask_tlm = _tensorize_batch(attention_mask_tlm, padding_value=0,
                                                  pad_to_multiple_of=self.pad_to_multiple_of)
            special_tokens_mask_tlm = _tensorize_batch(special_tokens_mask_tlm, padding_value=1,
                                                       pad_to_multiple_of=self.pad_to_multiple_of)
            token_type_ids_tlm = _tensorize_batch(token_type_ids_tlm, padding_value=0,
                                                  pad_to_multiple_of=self.pad_to_multiple_of)

            if self.reset_position_ids:
                position_ids_tlm = _tensorize_batch(position_ids_tlm, padding_value=self.tokenizer.pad_token_id,
                                                    pad_to_multiple_of=self.pad_to_multiple_of)

        batch = dict()

        batch["input_ids_1"], batch["labels_1"] = self.torch_mask_tokens(
            input_ids_1, special_tokens_mask=special_tokens_mask_1
        )

        batch["input_ids_2"], batch["labels_2"] = self.torch_mask_tokens(
            input_ids_2, special_tokens_mask=special_tokens_mask_2
        )

        batch['attention_mask_1'] = attention_mask_1
        batch['special_tokens_mask_1'] = special_tokens_mask_1
        batch['token_type_ids_1'] = token_type_ids_1

        batch['attention_mask_2'] = attention_mask_2
        batch['special_tokens_mask_2'] = special_tokens_mask_2
        batch['token_type_ids_2'] = token_type_ids_2

        if self.use_tlm:
            batch['input_ids_tlm'], batch['labels_tlm'] = self.torch_mask_tokens(
                input_ids_tlm, special_tokens_mask=special_tokens_mask_tlm
            )

            batch['attention_mask_tlm'] = attention_mask_tlm
            batch['token_type_ids_tlm'] = token_type_ids_tlm

            if self.reset_position_ids:
                batch['position_ids_tlm'] = position_ids_tlm

        return batch

    def torch_mask_tokens(self, inputs: Any, special_tokens_mask: Optional[Any] = None) -> Tuple[Any, Any]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.mask_token_id

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels


def load_transliteration_dataset_new(transliteration_train_file, tokenizer, max_seq_length,
                                     pad_to_multiple_of_8, model_args, data_args):
    transliteration_data_files = dict()
    transliteration_data_files["train"] = transliteration_train_file
    extension = transliteration_train_file.split(".")[-1]
    if extension == "txt":
        extension = "text"
    transliteration_datasets = load_dataset(
        extension,
        cache_dir=model_args.cache_dir,
        data_files=transliteration_data_files,
        use_auth_token=True if model_args.use_auth_token else None
    )

    text_column_name = "text"
    transliteration_column_name = "transliteration"

    def preprocess_function_tcm(examples):
        new_examples = {'text': [], 'transliteration': []}
        for i in range(len(examples['text'])):
            if (
                examples["text"][i] is None or len(examples["text"][i]) == 0
                or examples["transliteration"][i] is None
                or len(examples["transliteration"][i]) == 0
            ):
                continue
            else:
                new_examples['text'].append(examples["text"][i])
                new_examples['transliteration'].append(examples["transliteration"][i])

        examples = new_examples

        tokenized_text = tokenizer(examples["text"], max_length=max_seq_length, padding=False,
                                   truncation=True, return_special_tokens_mask=True)
        tokenized_transliteration = tokenizer(examples["transliteration"], max_length=max_seq_length,
                                              padding=False,
                                              truncation=True, return_special_tokens_mask=True)
        # concatenate
        model_inputs = dict()
        model_inputs['input_ids_1'] = tokenized_text['input_ids']
        model_inputs['attention_mask_1'] = tokenized_text['attention_mask']
        model_inputs['special_tokens_mask_1'] = tokenized_text['special_tokens_mask']
        model_inputs['token_type_ids_1'] = [[0 for _ in x] for x in tokenized_text['input_ids']]

        model_inputs['input_ids_2'] = tokenized_transliteration['input_ids']
        model_inputs['attention_mask_2'] = tokenized_transliteration['attention_mask']
        model_inputs['special_tokens_mask_2'] = tokenized_transliteration['special_tokens_mask']
        model_inputs['token_type_ids_2'] = [[0 for _ in x] for x in tokenized_transliteration['input_ids']]

        return model_inputs

    tokenized_transliteration_datasets = transliteration_datasets.map(
        preprocess_function_tcm,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        load_from_cache_file=not data_args.overwrite_cache,
        remove_columns=[text_column_name, transliteration_column_name],
        desc="Running tokenizer on paired transliteration dataset line_by_line"
    )

    transliteration_data_collator = DataCollatorForTransliterationModeling(
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        mlm_probability=data_args.mlm_probability,
        pad_to_multiple_of=8 if pad_to_multiple_of_8 else None,
        use_tlm=data_args.use_tlm,
        reset_position_ids=data_args.reset_position_ids
    )

    return tokenized_transliteration_datasets, transliteration_data_collator
