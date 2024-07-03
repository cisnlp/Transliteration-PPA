import torch
import torch.distributed as dist
from transformers import TrainerControl, TrainerState, TrainingArguments
from transformers.trainer import Trainer, TrainerCallback
from transformers.utils import logging
from torch import nn
from typing import Dict, Union, Any, List
from transformers.file_utils import is_apex_available

if is_apex_available():
    from apex import amp

logger = logging.get_logger(__name__)


class AddExtraLosses(TrainerCallback):
    def __init__(self, extra_losses: List[str]):
        self.extra_losses = extra_losses

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        control.extra_losses = {k: torch.tensor(0.0).to(args.device) for k in self.extra_losses}
        return control


class TCMContinuedPreTrainer(Trainer):
    def __init__(self,
                 train_cls: bool = False,
                 contrast_layer: int = 8,
                 temperature: float = 1.0,
                 tcm_loss_weight=1.0,
                 use_contrastive=True,
                 use_lm=True,
                 use_tlm=False,
                 extra_losses: List[str] = None,
                 use_cosine=False,
                 use_smoothing=False,
                 **kwargs):
        logger.debug("Initialising trainer")
        super().__init__(**kwargs)

        if extra_losses is not None:
            self.add_callback(AddExtraLosses(extra_losses))

        self.temperature = temperature
        self.train_cls = train_cls
        self.contrast_layer = contrast_layer
        self.tcm_loss_weight = tcm_loss_weight
        self.use_contrastive = use_contrastive
        self.use_lm = use_lm
        self.use_tlm = use_tlm
        self.use_cosine = use_cosine
        self.use_smoothing = use_smoothing

    def _maybe_log_save_evaluate(self, tr_loss, model, trial, epoch, ignore_keys_for_eval):
        if self.control.should_log:
            logs: Dict[str, float] = {}

            # all_gather + mean() to get average loss over all processes
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()

            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)

            if hasattr(self.control, 'extra_losses'):
                for k, v in self.control.extra_losses.items():
                    logs[k] = self._nested_gather(v).mean().item()
                    self.control.extra_losses[k] -= self.control.extra_losses[k]

                    logs[k] = round(logs[k] / (self.state.global_step - self._globalstep_last_logged), 4)
            logs["learning_rate"] = self._get_learning_rate()

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            self.log(logs)

        metrics = None
        if self.control.should_evaluate:
            if isinstance(self.eval_dataset, dict):
                for eval_dataset_name, eval_dataset in self.eval_dataset.items():
                    metrics = self.evaluate(
                        eval_dataset=eval_dataset,
                        ignore_keys=ignore_keys_for_eval,
                        metric_key_prefix=f"eval_{eval_dataset_name}",
                    )
            else:
                metrics = self.evaluate(ignore_keys=ignore_keys_for_eval)
            self._report_to_hp_search(trial, self.state.global_step, metrics)

        if self.control.should_save:
            self._save_checkpoint(model, trial, metrics=metrics)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        model.train()

        loss = torch.zeros([]).to(self.args.device)

        text_inputs = {
            'input_ids'           : inputs.pop('input_ids_1'),
            'attention_mask'      : inputs.pop('attention_mask_1'),
            'labels'              : inputs.pop('labels_1'),
            'token_type_ids'      : inputs.pop('token_type_ids_1'),
            'output_hidden_states': self.use_contrastive,
        }
        trans_inputs = {
            'input_ids'           : inputs.pop('input_ids_2'),
            'attention_mask'      : inputs.pop('attention_mask_2'),
            'labels'              : inputs.pop('labels_2'),
            'token_type_ids'      : inputs.pop('token_type_ids_2'),
            'output_hidden_states': self.use_contrastive,
        }

        special_tokens_mask_1 = inputs.pop('special_tokens_mask_1')
        special_tokens_mask_2 = inputs.pop('special_tokens_mask_2')

        text_inputs = self._prepare_inputs(text_inputs)
        results1 = model(**text_inputs)

        trans_inputs = self._prepare_inputs(trans_inputs)
        results2 = model(**trans_inputs)

        results_tlm = None

        if self.use_tlm:
            tlm_inputs = {
                'input_ids'           : inputs.pop('input_ids_tlm'),
                'attention_mask'      : inputs.pop('attention_mask_tlm'),
                'labels'              : inputs.pop('labels_tlm'),
                'token_type_ids'      : inputs.pop('token_type_ids_tlm'),
                'output_hidden_states': False,
            }

            if 'position_ids_tlm' in inputs:
                tlm_inputs['position_ids'] = inputs.pop('position_ids_tlm')

            tlm_inputs = self._prepare_inputs(tlm_inputs)
            results_tlm = model(**tlm_inputs)

        log_loss_lm_1, log_loss_lm_2, log_loss_tcm, log_loss_tlm = (
            torch.tensor(0.0).to(self.args.device) for _ in range(4))
        with self.compute_loss_context_manager():
            if self.use_lm:
                log_loss_lm_1 = results1['loss']
                log_loss_lm_2 = results2['loss']
                lm_loss = results1['loss'] + results2['loss']
                loss = loss + lm_loss
            if self.use_contrastive:
                tcm_loss = self.do_tcm_forward(
                    results1, results2, special_tokens_mask_1, special_tokens_mask_2)
                log_loss_tcm = tcm_loss
                loss = loss + self.tcm_loss_weight * tcm_loss
            if self.use_tlm:
                tlm_loss = results_tlm['loss']
                log_loss_tlm = tlm_loss
                loss = loss + tlm_loss

            # to avoid error because not all parameters contribute to the loss
            if not self.use_lm and not self.use_tlm:
                for p in model.parameters():
                    loss += 0.0 * p.sum()

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training
            log_loss_lm_1 = log_loss_lm_1.mean()
            log_loss_lm_2 = log_loss_lm_2.mean()
            log_loss_tcm = log_loss_tcm.mean()
            log_loss_tlm = log_loss_tlm.mean()

        if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
            # deepspeed handles loss scaling by gradient_accumulation_steps in its `backward`
            loss = loss / self.args.gradient_accumulation_steps

        if self.do_grad_scaling:
            self.scaler.scale(loss).backward()
        elif self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        elif self.deepspeed:
            # loss gets scaled under gradient_accumulation_steps in deepspeed
            loss = self.deepspeed.backward(loss)
        else:
            loss.backward()

        if hasattr(self.control, 'extra_losses'):
            self.control.extra_losses['lm_1'] += log_loss_lm_1.detach() / self.args.gradient_accumulation_steps
            self.control.extra_losses['lm_2'] += log_loss_lm_2.detach() / self.args.gradient_accumulation_steps
            self.control.extra_losses['tcm'] += log_loss_tcm.detach() / self.args.gradient_accumulation_steps
            self.control.extra_losses['tlm'] += log_loss_tlm.detach() / self.args.gradient_accumulation_steps

        return loss.detach()

    # doing forward for TCM loss
    def do_tcm_forward(self, results1, results2,
                       special_tokens_mask_1, special_tokens_mask_2):
        outputs1 = results1['hidden_states'][self.contrast_layer]
        outputs2 = results2['hidden_states'][self.contrast_layer]
        if self.train_cls:
            outputs1 = outputs1[:, 0, :]
            outputs2 = outputs2[:, 0, :]
        else:
            outputs1 = _mean_pool(outputs1, special_tokens_mask_1)
            outputs2 = _mean_pool(outputs2, special_tokens_mask_2)

        if dist.is_initialized() and dist.get_world_size() > 1:
            outputs1_list = [torch.zeros_like(outputs1) for _ in range(dist.get_world_size())]
            outputs2_list = [torch.zeros_like(outputs2) for _ in range(dist.get_world_size())]
            dist.all_gather(outputs1_list, outputs1.contiguous())
            dist.all_gather(outputs2_list, outputs2.contiguous())
            # fix the gradients
            outputs1_list[dist.get_rank()] = outputs1
            outputs2_list[dist.get_rank()] = outputs2
            outputs1 = torch.cat(outputs1_list, dim=0)
            outputs2 = torch.cat(outputs2_list, dim=0)

        tcm_loss = seq_contrast(outputs1, outputs2, self.temperature, use_cosine=self.use_cosine,
                                use_smoothing=self.use_smoothing)

        return tcm_loss


# from https://github.com/microsoft/COCO-LM/issues/2
def get_seq_label(sim_matrix):
    bsz = sim_matrix.size(0)
    seq_label = torch.arange(0, bsz, device=sim_matrix.device).view(-1, 2)
    seq_label[:, 0] = seq_label[:, 0] + 1
    seq_label[:, 1] = seq_label[:, 1] - 1
    # label is [1, 0, 3, 2, 5, 4, ...]
    seq_label = seq_label.view(-1)
    return seq_label


# from https://github.com/microsoft/COCO-LM/issues/2
def seq_contrast(out_1, out_2, temperature, use_cosine=False, use_smoothing=False):
    batch_size = out_1.size(0)
    global_out = torch.cat([out_1, out_2], dim=-1).view(2 * batch_size, -1)
    if use_cosine:
        global_out = global_out / global_out.norm(dim=1, keepdim=True)

    sim_matrix = torch.mm(global_out, global_out.t()) / temperature
    global_batch_size = sim_matrix.size(0)
    sim_matrix.masked_fill_(torch.eye(global_batch_size, device=sim_matrix.device, dtype=torch.bool), float('-inf'))
    truth = get_seq_label(sim_matrix)
    truth.requires_grad = False

    if use_smoothing:
        reg_size = 3 * batch_size
        reg_random = torch.normal(mean=0.0, std=1.0, size=(reg_size, global_out.size(1)), device=global_out.device)
        if use_cosine:
            reg_random = reg_random / reg_random.norm(dim=1, keepdim=True)
        reg_sim_matrix = torch.mm(global_out, reg_random.t()) / temperature
        sim_matrix = torch.cat((sim_matrix, reg_sim_matrix), dim=-1)

    # Using torch.log_softmax and torch.nn.NLLLoss
    log_softmax_sim_matrix = torch.log_softmax(sim_matrix, dim=-1, dtype=torch.float32)
    nll_loss = torch.nn.NLLLoss(reduction='mean')
    contrast_loss = nll_loss(log_softmax_sim_matrix, truth) * 0.5

    return contrast_loss


# 1 is the sequence token and 0 is the special token
def _mean_pool(data, special_tokens_mask):
    special_tokens_mask = special_tokens_mask.to(data.device)
    special_tokens_mask.requires_grad = False
    sequence_mask = 1 - special_tokens_mask
    sequence_mask.requires_grad = False
    return (data * sequence_mask.unsqueeze(2).float()).sum(dim=1) / sequence_mask.sum(dim=1).view(-1, 1)
