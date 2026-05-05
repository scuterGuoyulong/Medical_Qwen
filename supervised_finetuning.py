# -*- coding: utf-8 -*-
# Copyright 2023 XuMing(xuming624@qq.com) and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for causal language modeling (GPT, LLaMA, Bloom, ...) on a json file or a dataset.

part of code is modified from https://github.com/shibing624/textgen
"""

import json
import math
import os
import sys
import re
import csv
import shutil
from dataclasses import dataclass, field
from glob import glob
from types import MethodType
from typing import Literal, Optional, Tuple

import numpy as np
import torch
import torch.utils.data
from datasets import load_dataset
from loguru import logger
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from peft import LoraConfig, TaskType, get_peft_model, PeftModel, prepare_model_for_kbit_training
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    Seq2SeqTrainingArguments,
    set_seed,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
)

try:
    from transformers import EarlyStoppingCallback
except ImportError:  # 旧版 transformers
    from transformers.trainer_callback import EarlyStoppingCallback
from transformers.trainer import TRAINING_ARGS_NAME
from transformers.trainer_pt_utils import LabelSmoother
from transformers.utils.versions import require_version

from transformers.integrations import is_deepspeed_zero3_enabled
from transformers.trainer_callback import TrainerCallback

is_flash_attn_2_available = False
try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import pad_input, unpad_input

    is_flash_attn_2_available = True
except ImportError:
    is_flash_attn_2_available = False

from template import get_conv_template


def _dir_has_tokenizer_files(path: str) -> bool:
    if not path or not os.path.isdir(path):
        return True
    markers = ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "tokenizer.model")
    return any(os.path.isfile(os.path.join(path, n)) for n in markers)


def resolve_tokenizer_pretrained_path(model_args: "ModelArguments") -> str:
    """checkpoint 常不含 tokenizer，需单独指定基座目录或环境变量 TOKENIZER_NAME_OR_PATH。"""
    if model_args.tokenizer_name_or_path:
        return model_args.tokenizer_name_or_path
    mp = model_args.model_name_or_path
    if not mp:
        raise ValueError("model_name_or_path 不能为空")
    if os.path.isdir(mp) and not _dir_has_tokenizer_files(mp):
        fb = os.environ.get("TOKENIZER_NAME_OR_PATH") or os.environ.get("BASE_MODEL_FOR_TOKENIZER")
        if fb:
            logger.warning(f"模型路径 {mp} 无 tokenizer 文件，改用环境变量 TOKENIZER_NAME_OR_PATH={fb}")
            return fb
        raise ValueError(
            f"模型路径 {mp} 下未发现 tokenizer 文件（常见于 Trainer checkpoint）。"
            f"请添加参数: --tokenizer_name_or_path /path/to/Qwen3___5-4B "
            f"或设置环境变量 TOKENIZER_NAME_OR_PATH 指向基座模型目录。"
        )
    return mp


def load_tokenizer_for_training(model_args: "ModelArguments") -> AutoTokenizer:
    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "trust_remote_code": model_args.trust_remote_code,
    }
    src = resolve_tokenizer_pretrained_path(model_args)
    want_fast = model_args.use_fast_tokenizer
    try:
        return AutoTokenizer.from_pretrained(src, use_fast=want_fast, **tokenizer_kwargs)
    except Exception as e1:
        logger.warning(f"Tokenizer use_fast={want_fast} 加载失败，尝试切换: {e1}")
        try:
            return AutoTokenizer.from_pretrained(src, use_fast=not want_fast, **tokenizer_kwargs)
        except Exception as e2:
            raise RuntimeError(
                f"无法加载 tokenizer: {src}。可安装 pip install sentencepiece tiktoken 后重试。"
                f" 原始错误: {e2}"
            ) from e2


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )
    load_in_8bit: bool = field(default=False, metadata={"help": "Whether to load the model in 8bit mode or not."})
    load_in_4bit: bool = field(default=False, metadata={"help": "Whether to load the model in 4bit mode or not."})
    tokenizer_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The tokenizer for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    model_revision: Optional[str] = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    hf_hub_token: Optional[str] = field(default=None, metadata={"help": "Auth token to log in with Hugging Face Hub."})
    use_fast_tokenizer: bool = field(
        default=False,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    torch_dtype: Optional[str] = field(
        default="float16",
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    device_map: Optional[str] = field(
        default="auto",
        metadata={"help": "Device to map model to. If `auto` is passed, the device will be selected automatically. "},
    )
    trust_remote_code: bool = field(
        default=True,
        metadata={"help": "Whether to trust remote code when loading a model from a remote checkpoint."},
    )
    rope_scaling: Optional[Literal["linear", "dynamic"]] = field(
        default=None,
        metadata={"help": "Adopt scaled rotary positional embeddings."}
    )
    flash_attn: Optional[bool] = field(
        default=False,
        metadata={"help": "Enable FlashAttention-2 for faster training."}
    )
    shift_attn: Optional[bool] = field(
        default=False,
        metadata={"help": "Enable shifted sparse attention (S^2-Attn) proposed by LongLoRA."}
    )
    neft_alpha: Optional[float] = field(
        default=0,
        metadata={"help": "The alpha parameter to control the noise magnitude in NEFTune. value can be 5."}
    )

    def __post_init__(self):
        if self.model_name_or_path is None:
            raise ValueError("You must specify a valid model_name_or_path to run training.")


@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file_dir: Optional[str] = field(default=None, metadata={"help": "The train jsonl data file folder."})
    validation_file_dir: Optional[str] = field(default=None, metadata={"help": "The evaluation jsonl file folder."})
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_test_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker testing, truncate the number of test examples to this value if "
                "set."
            )
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={"help": "If only pad tokens should be ignored. This assumes that `config.pad_token_id` is defined."},
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=1,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    test_split_percentage: Optional[int] = field(
        default=1,
        metadata={
            "help": "The percentage of the train set used as test set in case there's no test split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )

    def __post_init__(self):
        if self.max_train_samples is not None and 0 < self.max_train_samples <= 1000:
            logger.warning("You may set max_train_samples = -1 to run all samples in production.")


@dataclass
class ScriptArguments:
    use_peft: bool = field(default=True, metadata={"help": "Whether to use peft"})
    train_on_inputs: bool = field(default=False, metadata={"help": "Whether to train on inputs"})
    target_modules: Optional[str] = field(default="all")
    lora_rank: Optional[int] = field(default=8)
    lora_dropout: Optional[float] = field(default=0.05)
    lora_alpha: Optional[float] = field(default=32.0)
    modules_to_save: Optional[str] = field(default=None)
    peft_path: Optional[str] = field(default=None, metadata={"help": "The path to the peft model"})
    qlora: bool = field(default=False, metadata={"help": "Whether to use qlora"})
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum model context length. suggest: 8192 * 4, 8192 * 2, 8192, 4096, 2048, 1024, 512"}
    )
    template_name: Optional[str] = field(default="vicuna", metadata={"help": "The prompt template name."})
    disable_thinking: bool = field(
        default=True,
        metadata={"help": "Disable thinking mode in prompt format by injecting empty think block for qwen template."},
    )
    early_stopping_patience: int = field(
        default=0,
        metadata={
            "help": (
                "Early stopping: stop after this many evaluations without improvement on "
                "`metric_for_best_model` (e.g. eval_bleu4). 0 disables early stopping."
            )
        },
    )
    early_stopping_threshold: float = field(
        default=0.0,
        metadata={
            "help": (
                "Minimum change in the monitored metric to qualify as an improvement "
                "(passed to EarlyStoppingCallback)."
            )
        },
    )

    def __post_init__(self):
        if self.model_max_length < 60:
            raise ValueError("You must specify a valid model_max_length >= 60 to run training")


class SavePeftModelTrainer(Trainer):
    """
    Trainer for lora models
    """

    def save_model(self, output_dir=None, _internal_call=False):
        """Save the LoRA model."""
        os.makedirs(output_dir, exist_ok=True)
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
        self.model.save_pretrained(output_dir)


def save_model(model, tokenizer, args):
    """Save the model and the tokenizer."""
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Take care of distributed/parallel training
    model_to_save = model.module if hasattr(model, "module") else model
    model_to_save.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


def save_model_zero3(model, tokenizer, args, trainer):
    """Save the model for deepspeed zero3.
    refer https://github.com/lm-sys/FastChat/blob/main/fastchat/train/train_lora.py#L209
    """
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    state_dict_zero3 = trainer.model_wrapped._zero3_consolidated_16bit_state_dict()
    model_to_save = model.module if hasattr(model, "module") else model
    model_to_save.save_pretrained(args.output_dir, state_dict=state_dict_zero3)
    tokenizer.save_pretrained(output_dir)


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


def find_all_linear_names(peft_model, int4=False, int8=False):
    """Find all linear layer names in the model. reference from qlora paper."""
    cls = torch.nn.Linear
    if int4 or int8:
        import bitsandbytes as bnb
        if int4:
            cls = bnb.nn.Linear4bit
        elif int8:
            cls = bnb.nn.Linear8bitLt
    lora_module_names = set()
    for name, module in peft_model.named_modules():
        if isinstance(module, cls):
            # last layer is not add to lora_module_names
            if 'lm_head' in name:
                continue
            if 'output_layer' in name:
                continue
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    return sorted(lora_module_names)


def check_and_optimize_memory():
    """检查并优化GPU内存使用"""
    if not torch.cuda.is_available():
        return

    logger.info("🔍 检查GPU内存状态...")

    # 清理缓存
    torch.cuda.empty_cache()

    # 检查每个GPU的内存状态
    num_gpus = torch.cuda.device_count()
    for i in range(num_gpus):
        props = torch.cuda.get_device_properties(i)
        total_memory = props.total_memory / 1024 ** 3
        allocated = torch.cuda.memory_allocated(i) / 1024 ** 3
        cached = torch.cuda.memory_reserved(i) / 1024 ** 3
        free = total_memory - allocated - cached

        logger.info(f"GPU {i} ({props.name}):")
        logger.info(f"  总内存: {total_memory:.1f}GB")
        logger.info(f"  已分配: {allocated:.1f}GB")
        logger.info(f"  已缓存: {cached:.1f}GB")
        logger.info(f"  可用: {free:.1f}GB")

    # 设置内存优化选项
    if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
        torch.backends.cuda.enable_flash_sdp(True)
        logger.info("✅ 启用Flash Attention优化")

    # 启用内存高效的注意力机制
    if hasattr(torch.backends.cuda, 'enable_mem_efficient_sdp'):
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        logger.info("✅ 启用内存高效注意力机制")


def clean_text_for_training(text: str) -> str:
    """
    清洗训练文本中的无效噪声，避免模型学习到格式污染：
    - 去掉 <think>...</think> 思维链标记
    - 去掉残留 <think> 标签
    - 去掉控制标记（如 <|im_start|>）
    - 将转义换行转为真实换行
    """
    if not isinstance(text, str) or not text:
        return ""

    cleaned = text
    cleaned = re.sub(r"```[\s\S]*?```", " ", cleaned)
    cleaned = re.sub(r"<think>[\s\S]*?</think>", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"</?think>", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"<\|[^>]+?\|>", " ", cleaned)
    cleaned = re.sub(r"(?m)^\s{0,3}#{1,6}\s*", " ", cleaned)
    cleaned = re.sub(r"(?m)^\s*[-*+]\s+", " ", cleaned)
    cleaned = re.sub(r"(?m)^\s*\d+\.\s+", " ", cleaned)
    cleaned = re.sub(r"(?m)^\s*>\s*", " ", cleaned)
    cleaned = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1", cleaned)
    cleaned = cleaned.replace("\\n", "\n").replace("\r", "\n")
    cleaned = re.sub(r"[`*_#]+", " ", cleaned)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned


def inject_empty_think_block(source_text: str, template_name: str, disable_thinking: bool) -> str:
    if not disable_thinking or template_name != "qwen":
        return source_text
    assistant_prefix = "<|im_start|>assistant\n"
    if source_text.endswith(assistant_prefix):
        return source_text + "<think>\n\n</think>\n\n"
    return source_text


def normalize_text_for_metrics(text: str) -> str:
    if not text:
        return ""
    cleaned = markdown_to_plaintext(text, keep_newlines=False)
    cleaned = cleaned.replace("\r", " ")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def markdown_to_plaintext(text: str, keep_newlines: bool = True) -> str:
    """
    将常见 Markdown/格式符清洗为纯文本：
    - 去掉 ```code```、标题(#)、列表符号(-/*/1.)、引用(>)、链接[text](url)、加粗/斜体等
    - 表格竖线 | 会转为空格
    - 可选择保留换行（用于样例文件可读性）
    """
    if not isinstance(text, str) or not text:
        return ""

    cleaned = text
    cleaned = re.sub(r"```[\s\S]*?```", " ", cleaned)
    cleaned = re.sub(r"<think>[\s\S]*?</think>", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"</?think>", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"<\|[^>]+?\|>", " ", cleaned)
    cleaned = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1", cleaned)  # link -> text

    # 标题/引用/列表（按行处理）
    cleaned = re.sub(r"(?m)^\s{0,3}#{1,6}\s*", "", cleaned)
    cleaned = re.sub(r"(?m)^\s*>\s*", "", cleaned)
    cleaned = re.sub(r"(?m)^\s*[-*+]\s+", "", cleaned)
    cleaned = re.sub(r"(?m)^\s*\d+\.\s+", "", cleaned)

    # 表格竖线与强调符号
    cleaned = cleaned.replace("|", " ")
    cleaned = re.sub(r"[`*_~]+", "", cleaned)

    # 处理转义换行
    cleaned = cleaned.replace("\\n", "\n")
    cleaned = cleaned.replace("\r", "\n")

    if keep_newlines:
        cleaned = re.sub(r"[ \t]+", " ", cleaned)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    else:
        cleaned = cleaned.replace("\n", " ")
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def compute_bleu_metrics(references, predictions):
    smooth_fn = SmoothingFunction().method4
    scores = {"bleu1": [], "bleu2": [], "bleu3": [], "bleu4": []}
    for ref, pred in zip(references, predictions):
        ref = normalize_text_for_metrics(ref)
        pred = normalize_text_for_metrics(pred)
        if not ref or not pred:
            continue
        ref_tokens = [list(ref)]
        pred_tokens = list(pred)
        scores["bleu1"].append(sentence_bleu(ref_tokens, pred_tokens, weights=(1, 0, 0, 0), smoothing_function=smooth_fn))
        scores["bleu2"].append(
            sentence_bleu(ref_tokens, pred_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smooth_fn)
        )
        scores["bleu3"].append(
            sentence_bleu(ref_tokens, pred_tokens, weights=(1 / 3, 1 / 3, 1 / 3, 0), smoothing_function=smooth_fn)
        )
        scores["bleu4"].append(
            sentence_bleu(ref_tokens, pred_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth_fn)
        )
    return {k: float(np.mean(v)) if len(v) > 0 else 0.0 for k, v in scores.items()}


class TopKMetricsCheckpointsCallback(TrainerCallback):
    """
    仅保留指定评估指标的 Top-K checkpoints（其余自动删除）。
    设计为在 on_evaluate 记录指标，在 on_save 时根据该 step 的指标决定是否保留 checkpoint。
    """

    def __init__(self, metrics_topk: dict[str, int]):
        self.metrics_topk = metrics_topk  # e.g. {"eval_bleu4": 3, "eval_bleu1": 3}
        self.step_metrics: dict[int, dict] = {}
        self.best: dict[str, list[tuple[float, int, str]]] = {k: [] for k in metrics_topk.keys()}

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None:
            return control
        self.step_metrics[int(state.global_step)] = dict(metrics)
        return control

    def on_save(self, args, state, control, **kwargs):
        # 只在主进程删除文件
        if not state.is_world_process_zero:
            return control

        step = int(state.global_step)
        ckpt_path = os.path.join(args.output_dir, f"checkpoint-{step}")
        if not os.path.isdir(ckpt_path):
            return control

        metrics = self.step_metrics.get(step) or {}

        # 更新各指标的 topk
        keep_paths: set[str] = set()
        for metric_key, k in self.metrics_topk.items():
            val = metrics.get(metric_key, None)
            if val is None:
                # 没有该指标就沿用历史 best
                pass
            else:
                try:
                    score = float(val)
                except Exception:
                    score = None
                if score is not None:
                    self.best.setdefault(metric_key, [])
                    self.best[metric_key].append((score, step, ckpt_path))
                    # 分数高优先；step 大作为次序稳定因子
                    self.best[metric_key] = sorted(self.best[metric_key], key=lambda x: (x[0], x[1]), reverse=True)
                    # 去重（同一路径只留一次）
                    dedup = []
                    seen = set()
                    for s, st, p in self.best[metric_key]:
                        if p in seen:
                            continue
                        seen.add(p)
                        dedup.append((s, st, p))
                    self.best[metric_key] = dedup[:k]

            for _, _, p in self.best.get(metric_key, []):
                keep_paths.add(p)

        # 兜底：当前 checkpoint 总是保留（避免误删）
        keep_paths.add(ckpt_path)

        # 删除非保留的 checkpoints
        try:
            for name in os.listdir(args.output_dir):
                if not name.startswith("checkpoint-"):
                    continue
                p = os.path.join(args.output_dir, name)
                if os.path.isdir(p) and p not in keep_paths:
                    shutil.rmtree(p, ignore_errors=True)
        except Exception as e:
            logger.warning(f"TopK checkpoint 清理失败（不影响训练继续）: {e}")

        return control


class SaveTrainingCurvesOnEvaluateCallback(TrainerCallback):
    """
    每次验证结束后，根据当前 log_history 重写 train_curve.csv / eval_curve.csv，
    并更新 training_curves.png。避免仅训练结束才落盘导致长训中途看不到曲线或进程中断丢失文件。
    """

    def __init__(self, output_dir: str):
        self.output_dir = output_dir

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not state.is_world_process_zero:
            return control
        try:
            save_training_curves(self.output_dir, state.log_history)
        except Exception as e:
            logger.warning(f"验证后更新曲线/CSV 失败（不影响训练）: {e}")
        return control


def save_training_curves(output_dir: str, log_history):
    """Save train/eval curves to csv and png."""
    os.makedirs(output_dir, exist_ok=True)

    train_rows = []
    eval_rows = []
    for item in log_history:
        step = item.get("step")
        if step is None:
            continue
        if "loss" in item and "eval_loss" not in item:
            train_rows.append({"step": step, "train_loss": item.get("loss")})
        if "eval_loss" in item or "eval_bleu4" in item:
            eval_rows.append(
                {
                    "step": step,
                    "eval_loss": item.get("eval_loss"),
                    "eval_bleu1": item.get("eval_bleu1"),
                    "eval_bleu2": item.get("eval_bleu2"),
                    "eval_bleu3": item.get("eval_bleu3"),
                    "eval_bleu4": item.get("eval_bleu4"),
                }
            )

    train_csv = os.path.join(output_dir, "train_curve.csv")
    eval_csv = os.path.join(output_dir, "eval_curve.csv")
    eval_fieldnames = ["step", "eval_loss", "eval_bleu1", "eval_bleu2", "eval_bleu3", "eval_bleu4"]

    with open(train_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "train_loss"])
        writer.writeheader()
        writer.writerows(train_rows)

    with open(eval_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=eval_fieldnames)
        writer.writeheader()
        writer.writerows(eval_rows)

    try:
        import matplotlib.pyplot as plt

        fig, ax1 = plt.subplots(figsize=(10, 6))
        if train_rows:
            ax1.plot(
                [x["step"] for x in train_rows],
                [x["train_loss"] for x in train_rows],
                label="train_loss",
                color="#1f77b4",
                linewidth=1.5,
            )
        if eval_rows:
            eval_steps = [x["step"] for x in eval_rows if x.get("eval_loss") is not None]
            eval_loss_values = [x["eval_loss"] for x in eval_rows if x.get("eval_loss") is not None]
            if eval_steps and eval_loss_values:
                ax1.plot(eval_steps, eval_loss_values, label="eval_loss", color="#ff7f0e", linewidth=1.5)
        ax1.set_xlabel("step")
        ax1.set_ylabel("loss")
        ax1.grid(alpha=0.3)

        ax2 = ax1.twinx()
        bleu_specs = [
            ("eval_bleu1", "#2ca02c"),
            ("eval_bleu2", "#d62728"),
            ("eval_bleu3", "#9467bd"),
            ("eval_bleu4", "#17becf"),
        ]
        for key, color in bleu_specs:
            steps_b = [x["step"] for x in eval_rows if x.get(key) is not None]
            vals_b = [x[key] for x in eval_rows if x.get(key) is not None]
            if steps_b and vals_b:
                ax2.plot(steps_b, vals_b, label=key, color=color, linewidth=1.5)
        ax2.set_ylabel("eval BLEU (1-4)")

        handles1, labels1 = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(handles1 + handles2, labels1 + labels2, loc="best", fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "training_curves.png"), dpi=160)
        plt.close(fig)
        logger.info(f"训练曲线已保存到: {os.path.join(output_dir, 'training_curves.png')}")
    except Exception as e:
        logger.warning(f"绘制训练曲线失败（已保存csv，可用TensorBoard查看）: {e}")


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, Seq2SeqTrainingArguments, ScriptArguments))

    # 使用 parse_args_into_dataclasses 时忽略未知参数
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # 如果我们传递了一个 JSON 文件，让我们用它来配置参数
        model_args, data_args, training_args, script_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1]))
    else:
        # 部分 transformers 版本 HfArgumentParser 未收录 TrainingArguments 的个别字段（如 --overwrite_output_dir），
        # 使用 return_remaining_strings 接回后手动处理，其余未知参数仍报错以防拼写错误。
        model_args, data_args, training_args, script_args, remaining = parser.parse_args_into_dataclasses(
            look_for_args_file=False,
            return_remaining_strings=True,
        )
        remaining = list(remaining)
        if "--overwrite_output_dir" in remaining:
            if hasattr(training_args, "overwrite_output_dir"):
                training_args.overwrite_output_dir = True
            remaining = [x for x in remaining if x != "--overwrite_output_dir"]
        if remaining:
            raise ValueError(f"Some specified arguments are not used by the HfArgumentParser: {remaining}")

    # 确保 DeepSpeed 配置正确加载
    if training_args.deepspeed is not None:
        training_args.distributed_state.deepspeed_plugin = None

    # The Trainer will handle distributed training setup
    is_main_process = training_args.local_rank in [-1, 0]

    # Only log on main process
    if is_main_process:
        logger.info(f"Model args: {model_args}")
        logger.info(f"Data args: {data_args}")
        logger.info(f"Training args: {training_args}")
        logger.info(f"Script args: {script_args}")
        logger.info(
            f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
            + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
        )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Load tokenizer（支持从仅含权重的 checkpoint 继续训：用 --tokenizer_name_or_path 或 TOKENIZER_NAME_OR_PATH）
    prompt_template = get_conv_template(script_args.template_name)
    tokenizer = load_tokenizer_for_training(model_args)
    if tokenizer.eos_token_id is None:
        tokenizer.eos_token = prompt_template.stop_str  # eos token is required
        tokenizer.add_special_tokens({"eos_token": tokenizer.eos_token})
        logger.info(f"Add eos_token: {tokenizer.eos_token}, eos_token_id: {tokenizer.eos_token_id}")
    if tokenizer.bos_token_id is None:
        tokenizer.add_special_tokens({"bos_token": tokenizer.eos_token})
        tokenizer.bos_token_id = tokenizer.eos_token_id
        logger.info(f"Add bos_token: {tokenizer.bos_token}, bos_token_id: {tokenizer.bos_token_id}")
    if tokenizer.pad_token_id is None:
        if tokenizer.unk_token_id is not None:
            tokenizer.pad_token = tokenizer.unk_token
        else:
            tokenizer.pad_token = tokenizer.eos_token
        logger.info(f"Add pad_token: {tokenizer.pad_token}, pad_token_id: {tokenizer.pad_token_id}")
    logger.debug(f"Tokenizer: {tokenizer}")

    IGNORE_INDEX = LabelSmoother.ignore_index if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id

    # Get datasets
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
        )
        if "validation" not in raw_datasets.keys():
            shuffled_train_dataset = raw_datasets["train"].shuffle(seed=42)
            # Split the shuffled train dataset into training and validation sets
            split = shuffled_train_dataset.train_test_split(
                test_size=data_args.validation_split_percentage / 100,
                seed=42
            )
            # Assign the split datasets back to raw_datasets
            raw_datasets["train"] = split["train"]
            raw_datasets["validation"] = split["test"]
    else:
        # Loading a dataset from local files.
        data_files = {}
        if data_args.train_file_dir is not None and os.path.exists(data_args.train_file_dir):
            train_data_files = glob(f'{data_args.train_file_dir}/**/*.json', recursive=True) + glob(
                f'{data_args.train_file_dir}/**/*.jsonl', recursive=True)
            logger.info(f"train files: {train_data_files}")
            data_files["train"] = train_data_files
        if data_args.validation_file_dir is not None and os.path.exists(data_args.validation_file_dir):
            eval_data_files = glob(f'{data_args.validation_file_dir}/**/*.json', recursive=True) + glob(
                f'{data_args.validation_file_dir}/**/*.jsonl', recursive=True)
            logger.info(f"eval files: {eval_data_files}")
            data_files["validation"] = eval_data_files
        raw_datasets = load_dataset(
            'json',
            data_files=data_files,
            cache_dir=model_args.cache_dir,
        )
    # Ensure train/validation/test splits are all available.
    if "validation" not in raw_datasets.keys() and "train" in raw_datasets.keys():
        train_dataset_for_split = raw_datasets["train"].shuffle(seed=42)
        val_pct = max(0.0, float(data_args.validation_split_percentage / 100))
        test_pct = max(0.0, float(data_args.test_split_percentage / 100))
        holdout_pct = val_pct + test_pct
        if holdout_pct <= 0:
            holdout_pct = 0.02
            val_pct, test_pct = 0.01, 0.01

        split = train_dataset_for_split.train_test_split(
            test_size=holdout_pct,
            seed=42,
        )
        raw_datasets["train"] = split["train"]
        holdout = split["test"]

        if val_pct > 0 and test_pct > 0:
            test_ratio_in_holdout = test_pct / holdout_pct
            holdout_split = holdout.train_test_split(test_size=test_ratio_in_holdout, seed=42)
            raw_datasets["validation"] = holdout_split["train"]
            raw_datasets["test"] = holdout_split["test"]
        elif val_pct > 0:
            raw_datasets["validation"] = holdout
            raw_datasets["test"] = holdout
        else:
            raw_datasets["test"] = holdout
            raw_datasets["validation"] = holdout

    if "validation" in raw_datasets.keys() and "test" not in raw_datasets.keys():
        val_dataset_for_split = raw_datasets["validation"].shuffle(seed=42)
        test_pct = max(0.0, float(data_args.test_split_percentage / 100))
        if test_pct > 0 and len(val_dataset_for_split) > 1:
            test_ratio = min(max(test_pct, 1.0 / len(val_dataset_for_split)), 0.5)
            val_split = val_dataset_for_split.train_test_split(test_size=test_ratio, seed=42)
            raw_datasets["validation"] = val_split["train"]
            raw_datasets["test"] = val_split["test"]
        else:
            raw_datasets["test"] = raw_datasets["validation"]

    logger.info(f"Raw datasets: {raw_datasets}")

    # Preprocessing the datasets
    max_length = script_args.model_max_length

    def preprocess_function(examples):
        """
        Preprocessing the datasets.
            part of code modified from https://github.com/lm-sys/FastChat
        """
        input_ids_list = []
        attention_mask_list = []
        targets_list = []
        roles = ["human", "gpt"]

        def get_dialog(examples):
            system_prompts = examples.get("system_prompt", "")
            for i, source in enumerate(examples['conversations']):
                system_prompt = ""
                if len(source) < 2:
                    continue
                data_role = source[0].get("from", "")
                if data_role == "system":
                    # Skip the first one if it is from system
                    system_prompt = source[0]["value"]
                    source = source[1:]
                    data_role = source[0].get("from", "")
                if data_role not in roles or data_role != roles[0]:
                    # Skip the first one if it is not from human
                    source = source[1:]
                if len(source) < 2:
                    continue
                messages = []
                for j, sentence in enumerate(source):
                    data_role = sentence.get("from", "")
                    if data_role not in roles:
                        logger.warning(f"unknown role: {data_role}, {i}. (ignored)")
                        break
                    if data_role == roles[j % 2]:
                        cleaned_value = clean_text_for_training(sentence.get("value", ""))
                        if cleaned_value:
                            messages.append(cleaned_value)
                if len(messages) % 2 != 0:
                    continue
                # Convert the list to pairs of elements
                history_messages = [[messages[k], messages[k + 1]] for k in range(0, len(messages), 2)]
                if not system_prompt:
                    system_prompt = system_prompts[i] if system_prompts else ""
                style_constraint = "请用纯中文短段落回答，不要用 Markdown。"
                system_prompt = (system_prompt.strip() + "\n\n" + style_constraint).strip() if system_prompt else style_constraint
                yield prompt_template.get_dialog(history_messages, system_prompt=system_prompt)

        for dialog in get_dialog(examples):
            input_ids, labels = [], []

            for i in range(len(dialog) // 2):
                source_text = inject_empty_think_block(
                    dialog[2 * i],
                    template_name=script_args.template_name,
                    disable_thinking=script_args.disable_thinking,
                )
                source_ids = tokenizer.encode(text=source_text, add_special_tokens=(i == 0))
                target_ids = tokenizer.encode(text=dialog[2 * i + 1], add_special_tokens=False)

                total_len = len(source_ids) + len(target_ids)
                max_source_len = int(max_length * (len(source_ids) / total_len))
                max_target_len = int(max_length * (len(target_ids) / total_len))

                if len(source_ids) > max_source_len:
                    source_ids = source_ids[:max_source_len]
                if len(target_ids) > max_target_len - 1:  # eos token
                    target_ids = target_ids[:max_target_len - 1]
                if len(source_ids) > 0 and source_ids[0] == tokenizer.eos_token_id:
                    source_ids = source_ids[1:]
                if len(target_ids) > 0 and target_ids[-1] == tokenizer.eos_token_id:
                    target_ids = target_ids[:-1]
                if len(input_ids) + len(source_ids) + len(target_ids) + 1 > max_length:
                    break

                input_ids += source_ids + target_ids + [tokenizer.eos_token_id]  # add eos token for each turn
                if script_args.train_on_inputs:
                    labels += source_ids + target_ids + [tokenizer.eos_token_id]
                else:
                    labels += [IGNORE_INDEX] * len(source_ids) + target_ids + [tokenizer.eos_token_id]

            input_ids_list.append(input_ids)
            attention_mask_list.append([1] * len(input_ids))
            targets_list.append(labels)

        return dict(
            input_ids=input_ids_list,
            attention_mask=attention_mask_list,
            labels=targets_list,
        )

    def filter_empty_labels(example):
        """Remove empty labels dataset."""
        return not all(label == IGNORE_INDEX for label in example["labels"])

    train_dataset = None
    max_train_samples = 0
    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets['train'].shuffle(seed=42)
        max_train_samples = len(train_dataset)
        if data_args.max_train_samples is not None and data_args.max_train_samples > 0:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))

        if is_main_process:
            logger.debug(f"Example train_dataset[0]: {train_dataset[0]}")

        with training_args.main_process_first(desc="Train dataset tokenization"):
            tokenized_dataset = train_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=train_dataset.column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on dataset" if is_main_process else None,
            )
            train_dataset = tokenized_dataset.filter(
                filter_empty_labels,
                num_proc=data_args.preprocessing_num_workers
            )

            if is_main_process:
                logger.debug(f"Num train_samples: {len(train_dataset)}")
                logger.debug("Tokenized training example:")
                logger.debug(f"Decode input_ids[0]:\n{tokenizer.decode(train_dataset[0]['input_ids'])}")
                replaced_labels = [label if label != IGNORE_INDEX else tokenizer.pad_token_id
                                   for label in list(train_dataset[0]['labels'])]
                logger.debug(f"Decode labels[0]:\n{tokenizer.decode(replaced_labels)}")

    eval_dataset = None
    max_eval_samples = 0
    if training_args.do_eval:
        with training_args.main_process_first(desc="Eval dataset tokenization"):
            if "validation" not in raw_datasets:
                raise ValueError("--do_eval requires a validation dataset")
            eval_dataset = raw_datasets["validation"]
            max_eval_samples = len(eval_dataset)
            if data_args.max_eval_samples is not None and data_args.max_eval_samples > 0:
                max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
                eval_dataset = eval_dataset.select(range(max_eval_samples))
            eval_size = len(eval_dataset)
            logger.debug(f"Num eval_samples: {eval_size}")
            if eval_size > 500:
                logger.warning(f"Num eval_samples is large: {eval_size}, "
                               f"training slow, consider reduce it by `--max_eval_samples=50`")
            logger.debug(f"Example eval_dataset[0]: {eval_dataset[0]}")
            eval_dataset = eval_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=eval_dataset.column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on validation dataset",
            )
            eval_dataset = eval_dataset.filter(filter_empty_labels, num_proc=data_args.preprocessing_num_workers)
            # 在 eval_dataset.map(...) 完成之后，trainer 初始化之前加这一行：
            logger.debug(f"Num eval_samples: {len(eval_dataset)}")
            logger.debug("Tokenized eval example:")
            logger.debug(tokenizer.decode(eval_dataset[0]['input_ids']))

    test_dataset = None
    max_test_samples = 0
    if "test" in raw_datasets:
        with training_args.main_process_first(desc="Test dataset tokenization"):
            test_dataset = raw_datasets["test"]
            max_test_samples = len(test_dataset)
            if data_args.max_test_samples is not None and data_args.max_test_samples > 0:
                max_test_samples = min(len(test_dataset), data_args.max_test_samples)
                test_dataset = test_dataset.select(range(max_test_samples))
            test_dataset = test_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=test_dataset.column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on test dataset",
            )
            test_dataset = test_dataset.filter(filter_empty_labels, num_proc=data_args.preprocessing_num_workers)
            logger.debug(f"Num test_samples: {len(test_dataset)}")

    # Load model
    if model_args.model_name_or_path:
        torch_dtype = (
            model_args.torch_dtype
            if model_args.torch_dtype in ["auto", None]
            else getattr(torch, model_args.torch_dtype)
        )
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        ddp = world_size != 1
        if ddp:
            model_args.device_map = {"": int(os.environ.get("LOCAL_RANK", "0"))}
            training_args.gradient_accumulation_steps = training_args.gradient_accumulation_steps // world_size or 1
        if script_args.qlora and (len(training_args.fsdp) > 0 or is_deepspeed_zero3_enabled()):
            logger.warning("FSDP and DeepSpeed ZeRO-3 are both currently incompatible with QLoRA.")

        config_kwargs = {
            "trust_remote_code": model_args.trust_remote_code,
            "cache_dir": model_args.cache_dir,
            "revision": model_args.model_revision,
            "token": model_args.hf_hub_token,
        }
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)

        # Set RoPE scaling
        if model_args.rope_scaling is not None:
            if hasattr(config, "rope_scaling"):
                if model_args.rope_scaling == "dynamic":
                    logger.warning(
                        "Dynamic NTK may not work well with fine-tuning. "
                        "See: https://github.com/huggingface/transformers/pull/24653"
                    )
                current_max_length = getattr(config, "max_position_embeddings", None)
                if current_max_length and script_args.model_max_length > current_max_length:
                    scaling_factor = float(math.ceil(script_args.model_max_length / current_max_length))
                else:
                    logger.warning(f"The model_max_length({script_args.model_max_length}) is smaller than max "
                                   f"length({current_max_length}). Consider increase model_max_length.")
                    scaling_factor = 1.0

                setattr(config, "rope_scaling", {"type": model_args.rope_scaling, "factor": scaling_factor})
                logger.info("Using {} scaling strategy and setting scaling factor to {}".format(
                    model_args.rope_scaling, scaling_factor
                ))
            else:
                logger.warning("Current model does not support RoPE scaling.")

        # Set FlashAttention-2
        if model_args.flash_attn:
            if is_flash_attn_2_available:
                config_kwargs["use_flash_attention_2"] = True
                logger.info("Using FlashAttention-2 for faster training and inference.")
            else:
                logger.warning("FlashAttention-2 is not installed.")
        elif model_args.shift_attn and getattr(config, "model_type", None) == "llama":
            logger.warning("Using `--flash_attn` for faster training in large context length, enable if your GPU"
                           " is RTX3090, RTX4090, A100 or H100.")

        # Set shifted sparse attention (S^2-Attn)
        if model_args.shift_attn:
            if getattr(config, "model_type", None) == "llama":
                setattr(config, "group_size_ratio", 0.25)
                logger.info("Using shifted sparse attention with group_size_ratio=1/4.")
            else:
                logger.warning("Current model does not support shifted sparse attention.")

        load_in_4bit = model_args.load_in_4bit
        load_in_8bit = model_args.load_in_8bit
        quantization_config = None
        if load_in_4bit and load_in_8bit:
            raise ValueError("Error, load_in_4bit and load_in_8bit cannot be set at the same time")
        elif load_in_8bit or load_in_4bit:
            logger.info(f"Quantizing model, load_in_4bit: {load_in_4bit}, load_in_8bit: {load_in_8bit}")
            if is_deepspeed_zero3_enabled():
                raise ValueError("DeepSpeed ZeRO-3 is incompatible with quantization.")
            if load_in_8bit:
                quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            elif load_in_4bit:
                if script_args.qlora:
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch_dtype,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4"
                    )
                else:
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch_dtype,
                    )

        model_kwargs = {
            "config": config,
            "torch_dtype": torch_dtype,
            "trust_remote_code": model_args.trust_remote_code,
            "quantization_config": quantization_config,
            "low_cpu_mem_usage": True,  # 减少CPU内存使用
            "device_map": model_args.device_map,
        }

        # 设置device_map
        num_gpus = torch.cuda.device_count()
        if model_args.device_map == 'auto':
            if num_gpus > 1 and not ddp:
                # 大模型多GPU：使用auto进行张量并行
                model_kwargs["device_map"] = "auto"
                # 设置最大内存使用
                max_memory = {}
                for i in range(num_gpus):
                    # 为每个GPU预留一些内存给梯度和优化器
                    gpu_props = torch.cuda.get_device_properties(i)
                    total_mem = gpu_props.total_memory
                    # 预留20%内存给训练时的梯度、优化器状态等
                    usable_mem = int(total_mem * 0.8)
                    max_memory[i] = f"{usable_mem // (1024 ** 3)}GiB"

                model_kwargs["max_memory"] = max_memory

        logger.info(f"🔧 大模型训练配置:")
        logger.info(f"  model_kwargs: {model_kwargs}")

        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            **model_kwargs
        )

        logger.info("✅ 模型加载完成")

        # 显示模型分布信息
        logger.info("📊 模型分布情况:")
        if hasattr(model, 'hf_device_map') and model.hf_device_map:
            logger.info("🔧 使用HuggingFace设备映射:")
            for module_name, device in model.hf_device_map.items():
                logger.info(f"  {module_name}: {device}")

            # 统计每个GPU上的模块数量
            device_count = {}
            for device in model.hf_device_map.values():
                device_str = str(device)
                device_count[device_str] = device_count.get(device_str, 0) + 1

            logger.info("📈 设备使用统计:")
            for device, count in device_count.items():
                logger.info(f"  {device}: {count} 个模块")
        else:
            # 检查模型参数的设备分布
            device_params = {}
            total_params = 0
            for name, param in model.named_parameters():
                device = str(param.device)
                if device not in device_params:
                    device_params[device] = {'count': 0, 'size': 0}
                device_params[device]['count'] += 1
                device_params[device]['size'] += param.numel()
                total_params += param.numel()

            logger.info("📈 参数设备分布:")
            for device, info in device_params.items():
                param_size_gb = info['size'] * 4 / 1024 ** 3  # 假设float32
                percentage = info['size'] / total_params * 100
                logger.info(f"  {device}: {info['count']} 个参数组, {param_size_gb:.2f}GB ({percentage:.1f}%)")

        # 显示GPU内存使用情况
        if torch.cuda.is_available():
            logger.info("💾 GPU内存使用情况:")
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / 1024 ** 3
                cached = torch.cuda.memory_reserved(i) / 1024 ** 3
                total = torch.cuda.get_device_properties(i).total_memory / 1024 ** 3
                logger.info(f"  GPU {i}: 已分配={allocated:.1f}GB, 缓存={cached:.1f}GB, 总计={total:.1f}GB")

        # Fix ChatGLM2 and ChatGLM3 and internlm2 LM head
        if getattr(config, "model_type", None) == "chatglm" or getattr(config, "model_type", None) == "internlm2":
            setattr(model, "lm_head", model.transformer.output_layer)
            setattr(model, "_keys_to_ignore_on_save", ["lm_head.weight"])

        # Set NEFTune trick for fine-tuning
        if model_args.neft_alpha > 0:
            input_embed = model.get_input_embeddings()
            if isinstance(input_embed, torch.nn.Embedding):
                def noisy_forward(self: torch.nn.Embedding, x: torch.Tensor) -> torch.Tensor:
                    embeddings = input_embed.__class__.forward(self, x)
                    dims = self.num_embeddings * self.embedding_dim
                    mag_norm = model_args.neft_alpha / (dims ** 0.5)
                    embeddings += torch.zeros_like(embeddings).uniform_(-mag_norm, mag_norm)
                    return embeddings

                input_embed.forward = MethodType(noisy_forward, input_embed)
                logger.info("Using noisy embedding with alpha={:.2f}".format(model_args.neft_alpha))
            else:
                logger.warning("Input embeddings are not normal nn.Embedding, cannot transform into noisy embedding.")

        # Patch Mixtral MOE model
        if getattr(config, "model_type", None) == "mixtral" and is_deepspeed_zero3_enabled():
            require_version("deepspeed>=0.13.0", "To fix: pip install deepspeed>=0.13.0")
            from deepspeed.utils import set_z3_leaf_modules  # type: ignore
            from transformers.models.mixtral.modeling_mixtral import MixtralSparseMoeBlock  # type: ignore

            set_z3_leaf_modules(model, [MixtralSparseMoeBlock])

        # Patch DeepSeek-V3 MoE module
        if getattr(config, "model_type", None) == "deepseek_v3" and is_deepspeed_zero3_enabled():
            require_version("deepspeed>=0.13.0", "To fix: pip install deepspeed>=0.13.0")
            # deepseek_v3 moe module set as leaf node
            for layer in model.model.layers:
                if 'DeepseekV3MoE' in str(type(layer.mlp)):
                    layer.mlp._z3_leaf = True
    else:
        raise ValueError(f"Error, model_name_or_path is None, SFT must be loaded from a pre-trained model")

    if script_args.use_peft:
        logger.info("Fine-tuning method: LoRA(PEFT)")

        # Set fp32 forward hook for lm_head
        output_layer = getattr(model, "lm_head")
        if isinstance(output_layer, torch.nn.Linear) and output_layer.weight.dtype != torch.float32:
            def fp32_forward_post_hook(module: torch.nn.Module, args: Tuple[torch.Tensor], output: torch.Tensor):
                return output.to(torch.float32)

            output_layer.register_forward_hook(fp32_forward_post_hook)

        # Load LoRA model
        if script_args.peft_path is not None:
            logger.info(f"Peft from pre-trained model: {script_args.peft_path}")
            model = PeftModel.from_pretrained(model, script_args.peft_path, is_trainable=True)
        else:
            logger.info("Init new peft model")
            if load_in_8bit or load_in_4bit:
                model = prepare_model_for_kbit_training(model, training_args.gradient_checkpointing)
            target_modules = script_args.target_modules.split(',') if script_args.target_modules else None
            if target_modules and 'all' in target_modules:
                target_modules = find_all_linear_names(model, int4=load_in_4bit, int8=load_in_8bit)
            modules_to_save = script_args.modules_to_save
            if modules_to_save is not None:
                modules_to_save = modules_to_save.split(',')
            logger.info(f"Peft target_modules: {target_modules}")
            logger.info(f"Peft lora_rank: {script_args.lora_rank}")
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                target_modules=target_modules,
                inference_mode=False,
                r=script_args.lora_rank,
                lora_alpha=script_args.lora_alpha,
                lora_dropout=script_args.lora_dropout,
                modules_to_save=modules_to_save)
            model = get_peft_model(model, peft_config)
        for param in filter(lambda p: p.requires_grad, model.parameters()):
            param.data = param.data.to(torch.float32)
        model.print_trainable_parameters()
    else:
        logger.info("Fine-tuning method: Full parameters training")
        model = model.float()
        print_trainable_parameters(model)

    # Initialize our Trainer
    if training_args.gradient_checkpointing and getattr(model, "supports_gradient_checkpointing", False):
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
        logger.info("Gradient checkpointing enabled.")
    else:
        model.config.use_cache = True
        logger.info("Gradient checkpointing disabled.")
    model.enable_input_require_grads()
    if not ddp and torch.cuda.device_count() > 1:
        # Keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=IGNORE_INDEX,
        pad_to_multiple_of=4 if tokenizer.padding_side == "right" else None,  # for shifted sparse attention
    )
    # trainer：global_step；eval_sample_tag：验证/测试样例文件名；generation_eval_dataset：当前做 generate 的数据集（含测试集）
    trainer_ref: dict = {
        "trainer": None,
        "eval_sample_tag": "val",
        "generation_eval_dataset": None,
    }

    def preprocess_logits_for_metrics(logits, labels):
        if isinstance(logits, tuple):
            logits = logits[0]
        return torch.argmax(logits, dim=-1)

    def compute_metrics(eval_preds):
        if training_args.local_rank not in [-1, 0]:
            return {"bleu1": 0.0, "bleu2": 0.0, "bleu3": 0.0, "bleu4": 0.0}

        _preds, _labels = eval_preds  # Trainer 传入；这里用 generation_eval_dataset 做生成评估

        ds = trainer_ref.get("generation_eval_dataset")
        if ds is None:
            ds = eval_dataset

        decoded_labels = []
        decoded_preds = []

        orig_use_cache = model.config.use_cache
        model.config.use_cache = True

        gen_batch_size = 4
        for start in range(0, len(ds), gen_batch_size):
            batch = ds[start: start + gen_batch_size]
            batch_input_ids = batch["input_ids"]
            batch_labels = batch["labels"]

            prompt_input_ids = []
            ref_token_lens = []
            for inp, lab in zip(batch_input_ids, batch_labels):
                lab = list(lab)
                target_ids = [x for x in lab if x != IGNORE_INDEX and x != tokenizer.pad_token_id]
                if not target_ids:
                    continue
                decoded_labels.append(tokenizer.decode(target_ids, skip_special_tokens=True).strip())

                prompt_len = sum(1 for x in lab if x == IGNORE_INDEX)
                prompt_input_ids.append(inp[:prompt_len])
                ref_token_lens.append(len(target_ids))

            if not prompt_input_ids:
                continue

            # 统一 max_new_tokens：对一个 batch 取最大 ref 长度
            batch_max_new_tokens = max(1, max(ref_token_lens))

            max_prompt_len = max(len(x) for x in prompt_input_ids)
            padded = [
                [tokenizer.pad_token_id] * (max_prompt_len - len(x)) + list(x)
                for x in prompt_input_ids
            ]
            input_tensor = torch.tensor(padded, dtype=torch.long).to(model.device)
            attention_mask = (input_tensor != tokenizer.pad_token_id).long()

            with torch.no_grad():
                output_ids = model.generate(
                    input_ids=input_tensor,
                    attention_mask=attention_mask,
                    max_new_tokens=batch_max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            for out in output_ids:
                gen_ids = out[max_prompt_len:].tolist()
                gen_ids = [x for x in gen_ids if x != tokenizer.pad_token_id]
                decoded_preds.append(tokenizer.decode(gen_ids, skip_special_tokens=True).strip())

        model.config.use_cache = orig_use_cache

        if not decoded_preds or not decoded_labels:
            return {"bleu1": 0.0, "bleu2": 0.0, "bleu3": 0.0, "bleu4": 0.0}

        min_len = min(len(decoded_preds), len(decoded_labels))
        decoded_preds = decoded_preds[:min_len]
        decoded_labels = decoded_labels[:min_len]

        tr = trainer_ref.get("trainer")
        step = int(tr.state.global_step) if tr is not None else 0
        n_save = min(10, len(decoded_preds), len(decoded_labels))
        if n_save > 0:
            samples = [
                {
                    "index": i,
                    "reference": markdown_to_plaintext(decoded_labels[i], keep_newlines=True),
                    "prediction": markdown_to_plaintext(decoded_preds[i], keep_newlines=True),
                }
                for i in range(n_save)
            ]
            out_dir = training_args.output_dir
            os.makedirs(out_dir, exist_ok=True)
            tag = trainer_ref.get("eval_sample_tag") or "val"
            out_path = os.path.join(out_dir, f"eval_samples_{tag}_step_{step}.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(samples, f, ensure_ascii=False, indent=2)
            label_cn = "验证集" if tag == "val" else "测试集"
            logger.info(f"{label_cn}样本（前 {n_save} 条 GT / 生成）已保存: {out_path}")

        return compute_bleu_metrics(decoded_labels, decoded_preds)


    trainer_init_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        data_collator=data_collator,
        compute_metrics=compute_metrics if training_args.do_eval else None,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics if training_args.do_eval else None,
    )

    callbacks = []
    if (
        training_args.do_train
        and training_args.do_eval
        and script_args.early_stopping_patience > 0
    ):
        if not getattr(training_args, "load_best_model_at_end", False):
            logger.warning(
                "early_stopping_patience > 0 but load_best_model_at_end is False; "
                "enabling load_best_model_at_end for early stopping."
            )
            training_args.load_best_model_at_end = True
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=script_args.early_stopping_patience,
                early_stopping_threshold=script_args.early_stopping_threshold,
            )
        )
        logger.info(
            f"Early stopping enabled: patience={script_args.early_stopping_patience}, "
            f"threshold={script_args.early_stopping_threshold}, "
            f"metric={getattr(training_args, 'metric_for_best_model', None)}"
        )
    if callbacks:
        trainer_init_kwargs["callbacks"] = callbacks

    # 仅保留 eval_bleu4 Top3 + eval_bleu1 Top3 的 checkpoints（其余自动删除）
    # 建议配合 save_steps == eval_steps，确保每次评估对应一个 checkpoint。
    topk_cb = TopKMetricsCheckpointsCallback({"eval_bleu4": 3, "eval_bleu1": 3})
    curves_cb = SaveTrainingCurvesOnEvaluateCallback(training_args.output_dir)
    if "callbacks" in trainer_init_kwargs:
        trainer_init_kwargs["callbacks"].extend([topk_cb, curves_cb])
    else:
        trainer_init_kwargs["callbacks"] = [topk_cb, curves_cb]

    # transformers>=4.56 uses processing_class instead of tokenizer in Trainer.__init__
    trainer_init_params = SavePeftModelTrainer.__init__.__code__.co_varnames
    if "processing_class" in trainer_init_params:
        trainer_init_kwargs["processing_class"] = tokenizer
    else:
        trainer_init_kwargs["tokenizer"] = tokenizer
    trainer = SavePeftModelTrainer(**trainer_init_kwargs)
    trainer_ref["trainer"] = trainer
    if eval_dataset is not None:
        trainer_ref["generation_eval_dataset"] = eval_dataset

    # Training
    if training_args.do_train:
        if trainer.is_world_process_zero():
            logger.info("*** Train ***")
            sample = next(iter(trainer.get_train_dataloader()))
            logger.debug(f"Train dataloader example: {sample}")
            logger.debug(f"input_ids:\n{list(sample['input_ids'])[:3]}, \nlabels:\n{list(sample['labels'])[:3]}")
            logger.debug(f"Decode input_ids[0]:\n{tokenizer.decode(sample['input_ids'][0])}")
            replaced_labels = [label if label != IGNORE_INDEX else tokenizer.pad_token_id for label in
                               sample['labels'][0]]
            logger.debug(f"Decode labels[0]:\n{tokenizer.decode(replaced_labels)}")
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)

        metrics = train_result.metrics
        metrics["train_samples"] = max_train_samples
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

        model.config.use_cache = True  # enable cache after training
        tokenizer.padding_side = "left"  # restore padding side
        tokenizer.init_kwargs["padding_side"] = "left"

        if trainer.is_world_process_zero():
            logger.debug(f"Training metrics: {metrics}")
            logger.info(f"Saving model checkpoint to {training_args.output_dir}")
            if is_deepspeed_zero3_enabled():
                save_model_zero3(model, tokenizer, training_args, trainer)
            else:
                save_model(model, tokenizer, training_args)

    # Evaluation
    if training_args.do_eval:
        if trainer.is_world_process_zero():
            logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(metric_key_prefix="eval")

        metrics["eval_samples"] = max_eval_samples
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
        if trainer.is_world_process_zero():
            logger.debug(f"Eval metrics: {metrics}")

    if test_dataset is not None:
        if trainer.is_world_process_zero():
            logger.info("*** Test ***")
        trainer_ref["eval_sample_tag"] = "test"
        trainer_ref["generation_eval_dataset"] = test_dataset
        try:
            test_metrics = trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix="test")
        finally:
            trainer_ref["eval_sample_tag"] = "val"
            trainer_ref["generation_eval_dataset"] = eval_dataset if eval_dataset is not None else None
        test_metrics["test_samples"] = max_test_samples
        if "test_loss" in test_metrics:
            try:
                test_metrics["test_perplexity"] = math.exp(test_metrics["test_loss"])
            except OverflowError:
                test_metrics["test_perplexity"] = float("inf")
        trainer.log_metrics("test", test_metrics)
        trainer.save_metrics("test", test_metrics)
        if trainer.is_world_process_zero():
            logger.debug(f"Test metrics: {test_metrics}")

    if trainer.is_world_process_zero():
        save_training_curves(training_args.output_dir, trainer.state.log_history)


if __name__ == "__main__":
    main()
