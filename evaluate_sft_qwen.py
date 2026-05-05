# -*- coding: utf-8 -*-
import os
import sys
import json
import argparse
import re
import tempfile
from glob import glob
from typing import List, Dict, Tuple, Optional

import torch
import nltk
from loguru import logger
from tqdm.auto import tqdm
from datasets import Dataset, load_dataset
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    GenerationConfig,
)
from peft import PeftModel

from template import get_conv_template


# ======================== 日志：实时输出 ========================
logger.remove()
logger.add(
    sys.stderr,
    level="INFO",
    colorize=True,
    enqueue=False,
    backtrace=False,
    diagnose=False,
)


# ======================== NLTK 资源 ========================
# nltk.download("punkt", quiet=True)
# nltk.download("wordnet", quiet=True)
# nltk.download("omw-1.4", quiet=True)


# ======================== 指标函数 ========================
def chinese_char_split(text: str) -> List[str]:
    return [ch for ch in text if ch.strip()]


def calculate_bleu(reference: str, candidate: str, n: int, use_char_split: bool = True) -> float:
    if use_char_split:
        ref_tokens = [chinese_char_split(reference)]
        cand_tokens = chinese_char_split(candidate)
    else:
        ref_tokens = [nltk.word_tokenize(reference.lower())]
        cand_tokens = nltk.word_tokenize(candidate.lower())

    weights = [1.0 / n] * n + [0.0] * (4 - n)
    smooth_fn = SmoothingFunction().method4

    try:
        return sentence_bleu(ref_tokens, cand_tokens, weights=weights, smoothing_function=smooth_fn)
    except Exception:
        return 0.0


def calculate_rouge_l(reference: str, candidate: str, use_char_split: bool = True) -> float:
    if use_char_split:
        ref_text = " ".join(chinese_char_split(reference))
        cand_text = " ".join(chinese_char_split(candidate))
    else:
        ref_text = reference
        cand_text = candidate

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    scores = scorer.score(ref_text, cand_text)
    return scores["rougeL"].fmeasure


def calculate_meteor(reference: str, candidate: str, use_char_split: bool = True) -> float:
    if use_char_split:
        ref_tokens = chinese_char_split(reference)
        cand_tokens = chinese_char_split(candidate)
    else:
        ref_tokens = nltk.word_tokenize(reference.lower())
        cand_tokens = nltk.word_tokenize(candidate.lower())

    try:
        return meteor_score([ref_tokens], cand_tokens)
    except Exception:
        return 0.0


def extract_answer_content(text: str) -> str:
    """
    尽量从模型输出中提取“最终回答”部分，剔除前置思考/分析模板。
    """
    if not text:
        return ""

    cleaned = text

    # 去掉可能残留的 role 前缀（如 "system ... user ... assistant ..."）
    cleaned = re.sub(r"(?is)^\s*system\b[\s\S]*?\bassistant\b[:：]?\s*", " ", cleaned)

    # 优先截取显式“最终回答”标记之后的内容
    final_markers = [
        r"final\s+answer\s*[:：]",
        r"final\s+response\s*[:：]",
        r"suggested\s+response\s*[:：]",
        r"最终回答\s*[:：]",
        r"最终答复\s*[:：]",
        r"答复如下\s*[:：]",
        r"回答如下\s*[:：]",
        r"建议回复\s*[:：]",
    ]
    for marker in final_markers:
        m = re.search(marker, cleaned, flags=re.IGNORECASE)
        if m:
            cleaned = cleaned[m.end():]
            break

    # 去掉常见英文思维链前缀（兼容不同措辞）
    cleaned = re.sub(
        r"(?is)^\s*here(?:'| i|’)?s\s+a\s+thinking\s+process[\s\S]{0,200}?:\s*",
        " ",
        cleaned,
    )

    # 如果开头仍是英文分析模板，则截到第一处中文正文
    if re.match(r"(?is)^\s*(analy[sz]e|intent|emotional state|situation|symptoms|question|topic|tone|target audience)\b", cleaned):
        m = re.search(r"[\u4e00-\u9fff]", cleaned)
        if m:
            cleaned = cleaned[m.start():]

    # 通用兜底：若明显是英文思考/分析前导，并且后续含中文正文，直接截到首个中文字符
    if re.search(r"(?is)\b(thinking process|analy[sz]e|step-by-step|reasoning)\b", cleaned):
        m = re.search(r"[\u4e00-\u9fff]", cleaned)
        if m and m.start() > 0:
            cleaned = cleaned[m.start():]

    return cleaned.strip()


def normalize_text_for_metrics(text: str) -> str:
    """
    清洗用于指标计算的文本，尽量去除与语义无关的噪声：
    - <think>...</think> 思维链标签
    - 转义换行与真实换行
    - 常见 markdown 格式符号
    - 多余空白
    """
    if not text:
        return ""

    cleaned = extract_answer_content(text)
    # 0) 去掉 markdown 代码块
    cleaned = re.sub(r"```[\s\S]*?```", " ", cleaned)
    # 1) 去掉 think 内容块（含多行）
    cleaned = re.sub(r"<think>[\s\S]*?</think>", " ", cleaned, flags=re.IGNORECASE)
    # 2) 去掉残余 think 标签
    cleaned = re.sub(r"</?think>", " ", cleaned, flags=re.IGNORECASE)
    # 3) 去掉常见特殊控制标记（如果偶发残留）
    cleaned = re.sub(r"<\|[^>]+?\|>", " ", cleaned)
    # 4) 去掉 markdown 标题/列表/引用等结构标记
    cleaned = re.sub(r"(?m)^\s{0,3}#{1,6}\s*", " ", cleaned)
    cleaned = re.sub(r"(?m)^\s*[-*+]\s+", " ", cleaned)
    cleaned = re.sub(r"(?m)^\s*\d+\.\s+", " ", cleaned)
    cleaned = re.sub(r"(?m)^\s*>\s*", " ", cleaned)
    # 5) 将 markdown 链接转成纯文本
    cleaned = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1", cleaned)
    # 6) 将转义换行与真实换行统一成空格
    cleaned = cleaned.replace("\\n", " ").replace("\n", " ").replace("\r", " ")
    # 7) 去掉常见 markdown 样式符号（不影响主要语义）
    cleaned = re.sub(r"[`*_#]+", " ", cleaned)
    # 8) 压缩空白
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def normalize_text_keep_think(text: str) -> str:
    """
    与 normalize_text_for_metrics 类似，但**保留 <think>...</think>**，用于“思考模式”结果展示/对照。
    注意：该文本不用于指标计算。
    """
    if not text:
        return ""
    cleaned = extract_answer_content(text)
    # 去掉 markdown 代码块（避免思考里夹杂代码块导致 JSON 过大/难读）
    cleaned = re.sub(r"```[\s\S]*?```", " ", cleaned)
    # 保留 think 块，仅清理残留控制标记与空白
    cleaned = re.sub(r"<\|[^>]+?\|>", " ", cleaned)
    cleaned = cleaned.replace("\\n", "\n")
    cleaned = re.sub(r"\r\n", "\n", cleaned)
    return cleaned.strip()


def compute_all_metrics(
    references: List[str],
    candidates: List[str],
    use_chinese_char_split: bool = True,
    enable_meteor: bool = True,
    enable_bertscore: bool = False,
    bertscore_model_type: str = "bert-base-chinese",
    bertscore_device: Optional[str] = None,
) -> Dict[str, float]:
    logger.info(f"开始计算 {len(references)} 条样本的生成指标...")

    metrics = {
        "bleu1": [],
        "bleu2": [],
        "bleu3": [],
        "bleu4": [],
        "rougeL": [],
    }
    if enable_meteor:
        metrics["meteor"] = []

    progress = tqdm(
        range(len(references)),
        total=len(references),
        desc="Scoring",
        dynamic_ncols=True,
        ncols=120,
    )

    for idx in progress:
        ref = normalize_text_for_metrics(references[idx])
        cand = normalize_text_for_metrics(candidates[idx])

        if not ref or not cand:
            logger.warning(
                f"跳过空文本（第{idx}条）: ref={ref[:20] if ref else '空'}, cand={cand[:20] if cand else '空'}"
            )
            continue

        metrics["bleu1"].append(calculate_bleu(ref, cand, n=1, use_char_split=use_chinese_char_split))
        metrics["bleu2"].append(calculate_bleu(ref, cand, n=2, use_char_split=use_chinese_char_split))
        metrics["bleu3"].append(calculate_bleu(ref, cand, n=3, use_char_split=use_chinese_char_split))
        metrics["bleu4"].append(calculate_bleu(ref, cand, n=4, use_char_split=use_chinese_char_split))
        metrics["rougeL"].append(calculate_rouge_l(ref, cand, use_char_split=use_chinese_char_split))

        if enable_meteor:
            metrics["meteor"].append(calculate_meteor(ref, cand, use_char_split=use_chinese_char_split))

    avg_metrics = {k: round(sum(v) / len(v), 4) if v else 0.0 for k, v in metrics.items()}

    if enable_bertscore:
        try:
            from bert_score import score as bert_score
        except ImportError as e:
            raise SystemExit(
                "缺少 BERTScore 依赖，请执行: pip install bert-score\n" f"原始错误: {e}"
            ) from e

        # BERTScore 期望原始字符串列表；这里用与 BLEU/ROUGE 相同的清洗后文本
        cleaned_refs = [normalize_text_for_metrics(t) for t in references]
        cleaned_cands = [normalize_text_for_metrics(t) for t in candidates]
        # 过滤空串，避免 bert-score 报错；同时保持对齐
        pair_refs = []
        pair_cands = []
        for r, c in zip(cleaned_refs, cleaned_cands, strict=True):
            if r and c:
                pair_refs.append(r)
                pair_cands.append(c)
        if pair_refs:
            P, R, F1 = bert_score(
                cands=pair_cands,
                refs=pair_refs,
                lang="zh",
                model_type=bertscore_model_type,
                device=bertscore_device,
                rescale_with_baseline=True,
                verbose=False,
            )
            avg_metrics["bertscore_precision"] = round(float(P.mean().item()), 4)
            avg_metrics["bertscore_recall"] = round(float(R.mean().item()), 4)
            avg_metrics["bertscore_f1"] = round(float(F1.mean().item()), 4)
        else:
            avg_metrics["bertscore_precision"] = 0.0
            avg_metrics["bertscore_recall"] = 0.0
            avg_metrics["bertscore_f1"] = 0.0

    logger.info("===== 生成指标结果 =====")
    for k, v in avg_metrics.items():
        logger.info(f"{k.upper()}: {v}")

    return avg_metrics


# ======================== 数据处理 ========================
def normalize_conversation(conv: List[Dict]) -> List[Dict]:
    valid_roles = {"system", "human", "gpt"}
    new_conv = []
    for msg in conv:
        role = msg.get("from", "")
        value = msg.get("value", "")
        if role in valid_roles and isinstance(value, str) and value.strip():
            new_conv.append({"from": role, "value": value.strip()})
    return new_conv


def build_eval_sample_from_conversation(
    conversations: List[Dict],
) -> Optional[Tuple[str, List[List[str]], str]]:
    """
    输出:
    - system_prompt
    - history_pairs: [[human1, gpt1], ..., [last_human, ""]]
    - target_answer: 最后一轮 gpt
    """
    conversations = normalize_conversation(conversations)
    if len(conversations) < 2:
        return None

    system_prompt = ""
    if conversations and conversations[0]["from"] == "system":
        system_prompt = conversations[0]["value"]
        conversations = conversations[1:]

    while conversations and conversations[0]["from"] != "human":
        conversations = conversations[1:]

    if len(conversations) < 2:
        return None

    messages = []
    expected = "human"
    for msg in conversations:
        if msg["from"] != expected:
            continue
        messages.append(msg["value"])
        expected = "gpt" if expected == "human" else "human"

    if len(messages) < 2 or len(messages) % 2 != 0:
        return None

    pairs = [[messages[i], messages[i + 1]] for i in range(0, len(messages), 2)]
    if len(pairs) < 1:
        return None

    target_answer = pairs[-1][1]
    history_pairs = pairs[:-1] + [[pairs[-1][0], ""]]
    return system_prompt, history_pairs, target_answer


def build_prompt_from_history(
    prompt_template,
    history_pairs: List[List[str]],
    system_prompt: str = "",
    tokenizer: Optional[AutoTokenizer] = None,
    disable_thinking: bool = True,
) -> str:
    # Preferred path: use tokenizer.apply_chat_template to align with model-native template.
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        for user_text, assistant_text in history_pairs:
            messages.append({"role": "user", "content": user_text})
            if assistant_text:
                messages.append({"role": "assistant", "content": assistant_text})

        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=not disable_thinking,
            )
        except TypeError:
            # Some transformers/tokenizers don't accept enable_thinking.
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            if disable_thinking:
                if prompt.endswith("<|im_start|>assistant\n<think>\n"):
                    prompt += "\n</think>\n\n"
                elif prompt.endswith("<|im_start|>assistant\n"):
                    prompt += "<think>\n\n</think>\n\n"
            return prompt

    # Fallback path: legacy prompt template construction.
    dialog = prompt_template.get_dialog(history_pairs, system_prompt=system_prompt)
    prompt = "".join(dialog)
    if disable_thinking:
        prompt += "<think>\n\n</think>\n\n"
    return prompt


def load_test_data(
    data_path: str,
    template_name: str,
    tokenizer: Optional[AutoTokenizer] = None,
    disable_thinking: bool = True,
    max_samples: int = None,
    cache_dir: Optional[str] = None,
) -> Tuple[List[str], List[str], List[Dict]]:
    logger.info(f"[数据] 开始加载测试数据: {data_path}")

    if os.path.isdir(data_path):
        data_files = glob(os.path.join(data_path, "**/*.json"), recursive=True)
        data_files += glob(os.path.join(data_path, "**/*.jsonl"), recursive=True)
        if not data_files:
            raise ValueError(f"文件夹 {data_path} 中未找到 json/jsonl 文件")
    else:
        data_files = [data_path]

    logger.info(f"[数据] 找到 {len(data_files)} 个文件")

    # 单文件 .jsonl：逐行读入内存，避免 datasets 在本地盘写大量 Arrow 缓存（磁盘满时常报错）
    if len(data_files) == 1 and os.path.isfile(data_files[0]) and data_files[0].endswith(".jsonl"):
        records = []
        with open(data_files[0], "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
        raw_dataset = Dataset.from_list(records)
        logger.info(f"[数据] jsonl 直读（无磁盘缓存），原始样本数: {len(raw_dataset)}")
    else:
        cache_root = cache_dir or os.environ.get("HF_DATASETS_CACHE") or os.path.join(
            tempfile.gettempdir(), "medical_qwen_eval_datasets_cache"
        )
        os.makedirs(cache_root, exist_ok=True)
        raw_dataset = load_dataset("json", data_files=data_files, cache_dir=cache_root)["train"]
        logger.info(f"[数据] 原始样本数: {len(raw_dataset)}（cache_dir={cache_root}）")

    if max_samples and max_samples > 0:
        raw_dataset = raw_dataset.select(range(min(max_samples, len(raw_dataset))))
        logger.info(f"[数据] 截取后样本数: {len(raw_dataset)}")

    prompt_template = get_conv_template(template_name)

    input_prompts = []
    reference_texts = []
    meta_infos = []

    progress = tqdm(
        raw_dataset,
        total=len(raw_dataset),
        desc="Loading data",
        dynamic_ncols=True,
        ncols=120,
    )

    for idx, sample in enumerate(progress):
        conversations = sample.get("conversations", [])
        if not conversations:
            logger.warning(f"第{idx}条样本无 conversations 字段，跳过")
            continue

        parsed = build_eval_sample_from_conversation(conversations)
        if parsed is None:
            logger.warning(f"第{idx}条样本无法解析为有效对话，跳过")
            continue

        system_prompt, history_pairs, target_answer = parsed
        prompt = build_prompt_from_history(
            prompt_template=prompt_template,
            history_pairs=history_pairs,
            system_prompt=system_prompt,
            tokenizer=tokenizer,
            disable_thinking=disable_thinking,
        )

        input_prompts.append(prompt)
        reference_texts.append(target_answer)
        meta_infos.append(
            {
                "system_prompt": system_prompt,
                "history_pairs": history_pairs,
            }
        )

    logger.info(f"[数据] 成功加载 {len(input_prompts)} 条有效评测样本")
    return input_prompts, reference_texts, meta_infos


# ======================== 模型加载 ========================
def load_tokenizer_with_fallback(
    tokenizer_path: str,
    trust_remote_code: bool = True,
) -> AutoTokenizer:
    """
    Trainer 保存的 checkpoint 常不含 tokenizer 文件；需从基座目录加载。
    优先 fast（tokenizer.json），失败再 slow；均需环境中有 tiktoken/sentencepiece 等依赖。
    """
    last_err = None
    for use_fast in (True, False):
        try:
            tok = AutoTokenizer.from_pretrained(
                tokenizer_path,
                trust_remote_code=trust_remote_code,
                padding_side="left",
                use_fast=use_fast,
            )
            logger.info(f"[模型] tokenizer 已从 {tokenizer_path} 加载 (use_fast={use_fast})")
            return tok
        except Exception as e:
            last_err = e
            logger.warning(f"[模型] tokenizer use_fast={use_fast} 失败: {e}")
    raise RuntimeError(
        f"无法从 {tokenizer_path} 加载 tokenizer。请安装: pip install sentencepiece tiktoken；"
        f"或检查路径是否为完整基座模型目录。最后错误: {last_err}"
    ) from last_err


def load_model_and_tokenizer(
    base_model_path: str,
    peft_model_path: str = None,
    tokenizer_path: Optional[str] = None,
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
    device_map: str = "auto",
    trust_remote_code: bool = True,
    torch_dtype: str = "bfloat16",
    disable_fla: bool = True,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    tok_src = tokenizer_path or base_model_path
    logger.info(f"[模型 1/4] 开始加载 tokenizer: {tok_src}")

    if load_in_4bit and load_in_8bit:
        raise ValueError("load_in_4bit 和 load_in_8bit 不能同时启用")

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    real_torch_dtype = dtype_map.get(torch_dtype, torch.bfloat16)

    quantization_config = None
    if load_in_4bit:
        logger.info("[模型] 启用 4bit 量化加载")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=real_torch_dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    elif load_in_8bit:
        logger.info("[模型] 启用 8bit 量化加载")
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    tokenizer = load_tokenizer_with_fallback(tok_src, trust_remote_code=trust_remote_code)
    logger.info("[模型 2/4] tokenizer 加载完成")

    if tokenizer.eos_token_id is None:
        tokenizer.add_special_tokens({"eos_token": "</s>"})
        logger.info(f"[模型] 补充 eos_token: {tokenizer.eos_token}")

    if tokenizer.bos_token_id is None:
        tokenizer.add_special_tokens({"bos_token": tokenizer.eos_token})
        tokenizer.bos_token_id = tokenizer.eos_token_id
        logger.info(f"[模型] 补充 bos_token: {tokenizer.bos_token}")

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.unk_token if tokenizer.unk_token_id is not None else tokenizer.eos_token
        logger.info(f"[模型] 补充 pad_token: {tokenizer.pad_token}")

    logger.info("[模型 3/4] 开始加载 base model ...")
    if disable_fla:
        # Qwen3.5 的部分 attention 实现会尝试导入 flash-linear-attention(fla) 并触发 triton JIT。
        # 在某些环境下 fla/triton 组合不兼容会导致模型类加载失败。这里提供“强制禁用”开关，
        # 让其回退到 PyTorch/SDPA 等更稳的实现路径（若后端支持）。
        os.environ.setdefault("QWEN_DISABLE_FLA", "1")
        os.environ.setdefault("FLASH_LINEAR_ATTENTION_DISABLE", "1")
        os.environ.setdefault("FLA_DISABLE", "1")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            quantization_config=quantization_config,
            device_map=device_map,
            torch_dtype=real_torch_dtype,
            trust_remote_code=trust_remote_code,
            low_cpu_mem_usage=True,
        )
    except Exception as e:
        msg = repr(e)
        if "fla" in msg.lower() or "triton" in msg.lower() or "Qwen3_5ForCausalLM" in msg:
            raise RuntimeError(
                "加载 Qwen3.5 checkpoint 失败，常见原因是环境里安装了 flash-linear-attention(fla) "
                "但与 triton 版本不兼容，导入时触发 triton JIT 报错，进而导致 Qwen3_5ForCausalLM 类无法加载。\n\n"
                "建议修复方式（二选一）：\n"
                "1) 卸载 fla / flash-linear-attention，让模型回退到 torch 实现：\n"
                "   pip uninstall -y fla flash-linear-attention\n"
                "2) 升级/重装与当前 CUDA/PyTorch 匹配的 triton 与 fla（需要你们环境的固定版本策略）。\n\n"
                f"原始错误: {e}"
            ) from e
        raise
    logger.info("[模型 3/4] base model 加载完成")

    if peft_model_path:
        logger.info(f"[模型 4/4] 开始加载 LoRA: {peft_model_path}")
        model = PeftModel.from_pretrained(model, peft_model_path)
        logger.info("[模型 4/4] LoRA 加载完成")

    model.eval()
    logger.info("[模型] 模型加载完成，进入评测")
    return model, tokenizer


def get_input_device(model) -> torch.device:
    try:
        return model.get_input_embeddings().weight.device
    except Exception:
        try:
            return next(model.parameters()).device
        except StopIteration:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ======================== 生成 ========================
@torch.no_grad()
def batch_generate(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    input_prompts: List[str],
    reference_texts: Optional[List[str]] = None,
    max_input_length: int = 4096,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    batch_size: int = 4,
    do_sample: bool = False,
    repetition_penalty: float = 1.05,
    print_every_batch: bool = True,
    show_examples_per_batch: int = 1,
    print_full_raw_output: bool = False,
    keep_think_in_outputs: bool = False,
) -> Tuple[List[str], List[str]]:
    logger.info(f"[生成] 开始批量生成，共 {len(input_prompts)} 条，batch_size={batch_size}")

    generation_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature if do_sample else 1.0,
        top_p=top_p if do_sample else 1.0,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        repetition_penalty=repetition_penalty,
    )

    input_device = get_input_device(model)
    logger.info(f"[生成] 输入张量设备: {input_device}")

    predictions = []
    raw_predictions = []
    think_preserved_predictions = []

    total_batches = (len(input_prompts) + batch_size - 1) // batch_size
    progress_bar = tqdm(
        range(0, len(input_prompts), batch_size),
        total=total_batches,
        desc="Generating",
        ncols=120,
        dynamic_ncols=True,
    )

    for batch_idx, i in enumerate(progress_bar, start=1):
        batch_prompts = input_prompts[i:i + batch_size]
        batch_refs = reference_texts[i:i + batch_size] if reference_texts is not None else None

        encoded = tokenizer(
            batch_prompts,
            return_tensors="pt",
            truncation=True,
            max_length=max_input_length,
            padding=True,
        )

        input_ids = encoded["input_ids"].to(input_device)
        attention_mask = encoded["attention_mask"].to(input_device)
        # IMPORTANT:
        # For left padding, attention_mask.sum() is the "non-pad length", not the
        # absolute generation start index in the batched tensor.
        # We should cut generated tokens from the full input sequence length.
        prompt_length = input_ids.size(1)

        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            generation_config=generation_config,
        )

        batch_predictions = []
        batch_raw_predictions = []
        for j in range(outputs.size(0)):
            gen_ids = outputs[j][prompt_length:]
            raw_pred_text = tokenizer.decode(gen_ids, skip_special_tokens=False).strip()
            pred_text = normalize_text_for_metrics(raw_pred_text)
            if keep_think_in_outputs:
                think_preserved_predictions.append(normalize_text_keep_think(raw_pred_text))
            raw_predictions.append(raw_pred_text)
            predictions.append(pred_text)
            batch_predictions.append(pred_text)
            batch_raw_predictions.append(raw_pred_text)

        progress_bar.set_postfix({
            "batch": f"{batch_idx}/{total_batches}",
            "done": len(predictions),
        })

        if print_every_batch:
            logger.info(f"\n========== Batch {batch_idx}/{total_batches} ==========")
            for k in range(min(show_examples_per_batch, len(batch_prompts))):
                prompt_preview = batch_prompts[k].replace("\n", "\\n")
                pred_preview = batch_predictions[k].replace("\n", "\\n")
                logger.info(f"[Prompt {k}] {prompt_preview}")
                if batch_refs is not None:
                    ref_preview = batch_refs[k].replace("\n", "\\n")
                    logger.info(f"[Ref    {k}] {ref_preview}")
                logger.info(f"[Pred   {k}] {pred_preview}")

            if print_full_raw_output:
                for k, raw_text in enumerate(batch_raw_predictions):
                    logger.info(f"[PredRawFull {k}]")
                    logger.info(raw_text)
                    logger.info(f"[PredRawRepr {k}] {repr(raw_text)}")
            logger.info("========================================")

            # 强制刷新，避免 bash 下“看起来没输出”
            sys.stdout.flush()
            sys.stderr.flush()

    logger.info("[生成] 批量生成完成")
    # 将“保留 think”的结果通过 raw_predictions 的并行保存输出到文件层
    # 这里不改函数返回签名，避免大范围改动；在 main() 里按需从闭包变量取值并写入。
    return predictions, raw_predictions, think_preserved_predictions


# ======================== 主函数 ========================
def main():
    print("main() started", flush=True)

    parser = argparse.ArgumentParser(description="评测 SFT/LoRA 微调后的 Qwen 模型")

    # 模型参数
    parser.add_argument("--base_model_path", required=True, help="基础模型路径（全参微调可为 checkpoint 目录）")
    parser.add_argument("--peft_model_path", default=None, help="LoRA checkpoint 路径")
    parser.add_argument(
        "--tokenizer_path",
        default=None,
        help="tokenizer 所在目录；checkpoint 常无 tokenizer 文件，需指向原始基座模型目录",
    )
    parser.add_argument("--load_in_4bit", action="store_true", help="4bit 量化加载")
    parser.add_argument("--load_in_8bit", action="store_true", help="8bit 量化加载")
    parser.add_argument("--device_map", default="auto", help="设备映射")
    parser.add_argument("--torch_dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument(
        "--disable_fla",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否禁用 flash-linear-attention(fla)（默认禁用，避免 triton JIT 兼容性问题）",
    )

    # 数据参数
    parser.add_argument("--test_data_path", required=True, help="测试集路径（json/jsonl 文件或目录）")
    parser.add_argument("--template_name", default="qwen", help="必须与训练时保持一致")
    parser.add_argument(
        "--disable_thinking",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否关闭 thinking（默认关闭，自动注入空 think 块）",
    )
    parser.add_argument("--max_samples", type=int, default=None, help="最大测试样本数")
    parser.add_argument(
        "--cache_dir",
        default=None,
        help="datasets 缓存目录（多文件 json 合并加载时用）；单文件 .jsonl 默认直读内存不写缓存",
    )
    parser.add_argument("--use_chinese_char_split", action="store_true", default=True, help="中文按字评测")

    # 生成参数
    parser.add_argument("--max_input_length", type=int, default=4096, help="输入最大长度")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="最大生成长度")
    parser.add_argument("--temperature", type=float, default=0.7, help="采样温度")
    parser.add_argument("--top_p", type=float, default=0.9, help="top_p")
    parser.add_argument("--batch_size", type=int, default=4, help="生成 batch size")
    parser.add_argument("--do_sample", action="store_true", help="是否采样生成")
    parser.add_argument("--repetition_penalty", type=float, default=1.05)

    # 日志打印参数
    parser.add_argument("--print_every_batch", action="store_true", help="每个 batch 打印样例")
    parser.add_argument("--show_examples_per_batch", type=int, default=1, help="每个 batch 打印几个样例")
    parser.add_argument(
        "--print_full_raw_output",
        action="store_true",
        help="打印每条样本的完整原始输出（未清洗）及repr格式，便于分析思考前缀模式",
    )

    # 结果参数
    parser.add_argument("--output_dir", default="./test_results", help="结果保存目录")
    parser.add_argument("--enable_meteor", action="store_true", help="是否计算 meteor")
    parser.add_argument("--enable_bertscore", action="store_true", help="是否计算 BERTScore（P/R/F1）")
    parser.add_argument(
        "--bertscore_model_type",
        default="bert-base-chinese",
        help="BERTScore 使用的模型名或本地目录（离线环境建议先下载到本地）",
    )
    parser.add_argument(
        "--bertscore_device",
        default="",
        help='BERTScore 计算设备，如 "cuda:0"；默认自动选择',
    )
    parser.add_argument(
        "--save_think_traces",
        action="store_true",
        help="当启用 thinking（--no-disable_thinking）时，额外在结果 JSON 中写入 prediction_with_think 字段",
    )

    args = parser.parse_args()

    logger.info(f"评测参数: {args}")
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. 加载模型与 tokenizer（tokenizer 用于 apply_chat_template）
    model, tokenizer = load_model_and_tokenizer(
        base_model_path=args.base_model_path,
        peft_model_path=args.peft_model_path,
        tokenizer_path=args.tokenizer_path,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
        disable_fla=args.disable_fla,
    )

    # 2. 加载测试数据并构造 prompt
    input_prompts, reference_texts, meta_infos = load_test_data(
        data_path=args.test_data_path,
        template_name=args.template_name,
        tokenizer=tokenizer,
        disable_thinking=args.disable_thinking,
        max_samples=args.max_samples,
        cache_dir=args.cache_dir,
    )
    if len(input_prompts) == 0:
        logger.error("未加载到有效测试数据，退出")
        return

    # 统一清洗参考答案，避免换行/markdown噪声影响打印、保存与评测
    cleaned_reference_texts = [normalize_text_for_metrics(t) for t in reference_texts]

    prompt_preview = input_prompts[0].replace("\n", "\\n")
    ref_preview = cleaned_reference_texts[0].replace("\n", "\\n")
    logger.info(f"[样例] 第一条 prompt 预览: {prompt_preview}")
    logger.info(f"[样例] 第一条 reference(清洗后): {ref_preview}")


    # 3. 批量生成
    keep_think = (not args.disable_thinking) and bool(args.save_think_traces)
    pred_texts, raw_pred_texts, pred_texts_with_think = batch_generate(
        model=model,
        tokenizer=tokenizer,
        input_prompts=input_prompts,
        reference_texts=cleaned_reference_texts,
        max_input_length=args.max_input_length,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        batch_size=args.batch_size,
        do_sample=args.do_sample,
        repetition_penalty=args.repetition_penalty,
        print_every_batch=args.print_every_batch,
        show_examples_per_batch=args.show_examples_per_batch,
        print_full_raw_output=args.print_full_raw_output,
        keep_think_in_outputs=keep_think,
    )

    # 4. 计算指标
    metrics = compute_all_metrics(
        references=cleaned_reference_texts,
        candidates=pred_texts,
        use_chinese_char_split=args.use_chinese_char_split,
        enable_meteor=args.enable_meteor,
        enable_bertscore=args.enable_bertscore,
        bertscore_model_type=args.bertscore_model_type,
        bertscore_device=(args.bertscore_device or None),
    )

    # 5. 保存指标
    metrics_path = os.path.join(args.output_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    logger.info(f"指标结果已保存到: {metrics_path}")

    # 6. 保存样例
    examples_path = os.path.join(args.output_dir, "generation_examples.json")
    examples = []
    for i in range(min(50, len(input_prompts))):
        examples.append(
            {
                "prompt": input_prompts[i],
                "reference": cleaned_reference_texts[i],
                "raw_prediction": raw_pred_texts[i],
                "prediction": pred_texts[i],
                "meta": meta_infos[i],
            }
        )
    with open(examples_path, "w", encoding="utf-8") as f:
        json.dump(examples, f, ensure_ascii=False, indent=2)
    logger.info(f"生成样例已保存到: {examples_path}")

    # 7. 保存完整结果
    all_results_path = os.path.join(args.output_dir, "all_results.json")
    all_results = []
    for i in range(len(input_prompts)):
        row = {
            "prompt": input_prompts[i],
            "reference": cleaned_reference_texts[i],
            "raw_prediction": raw_pred_texts[i],
            "prediction": pred_texts[i],
            "meta": meta_infos[i],
        }
        if keep_think:
            row["prediction_with_think"] = pred_texts_with_think[i]
        all_results.append(row)
    with open(all_results_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    logger.info(f"所有生成结果已保存到: {all_results_path}")

    logger.info("测试完成！")
    print("evaluation finished", flush=True)


if __name__ == "__main__":
    main()
