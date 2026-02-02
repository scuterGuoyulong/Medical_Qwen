## 📖 Introduction

**Medical_Qwen** 训练医疗大模型，实现了包括增量预训练、有监督微调、RLHF(奖励建模、强化学习训练)和DPO(直接偏好优化)、GRPO等,还有对应的测试代码，包括BLEU1-4、ROUGE-L等指标的计算。

TODO: 尝试改进模型的框架，在已有模型上增加RAG和输入流的因果推断
<img src="https://github.com/shibing624/MedicalGPT/blob/main/docs/dpo.jpg" width="860" />


## 😊 Features


基于ChatGPT Training Pipeline，本项目实现了领域模型--医疗行业语言大模型的训练：


- 第一阶段：PT(Continue PreTraining)增量预训练，在海量领域文档数据上二次预训练GPT模型，以适应领域数据分布（可选）
- 第二阶段：SFT(Supervised Fine-tuning)有监督微调，构造指令微调数据集，在预训练模型基础上做指令精调，以对齐指令意图，并注入领域知识
- 第三阶段
  - RLHF(Reinforcement Learning from Human Feedback)基于人类反馈对语言模型进行强化学习，分为两步：
    - RM(Reward Model)奖励模型建模，构造人类偏好排序数据集，训练奖励模型，用来建模人类偏好，主要是"HHH"原则，具体是"helpful, honest, harmless"
    - RL(Reinforcement Learning)强化学习，用奖励模型来训练SFT模型，生成模型使用奖励或惩罚来更新其策略，以便生成更高质量、更符合人类偏好的文本
  - [DPO(Direct Preference Optimization)](https://arxiv.org/pdf/2305.18290.pdf)直接偏好优化方法，DPO通过直接优化语言模型来实现对其行为的精确控制，而无需使用复杂的强化学习，也可以有效学习到人类偏好，DPO相较于RLHF更容易实现且易于训练，效果更好
  - [GRPO](https://arxiv.org/pdf/2402.03300)通过组内相对奖励来估计基线，从而避免使用额外的价值函数模型


## ▶️ Demo


我们提供了一个简洁的基于gradio的交互式web界面，启动服务后，可通过浏览器访问，输入问题，模型会返回答案。

启动服务，命令如下：
```shell
CUDA_VISIBLE_DEVICES=0 python gradio_demo.py --base_model path_to_llama_hf_dir --lora_model path_to_lora_dir
```

参数说明：

- `--base_model {base_model}`：存放HF格式的LLaMA模型权重和配置文件的目录，也可使用HF Model Hub模型调用名称
- `--lora_model {lora_model}`：LoRA文件所在目录，也可使用HF Model Hub模型调用名称。若lora权重已经合并到预训练模型，则删除--lora_model参数
- `--tokenizer_path {tokenizer_path}`：存放对应tokenizer的目录。若不提供此参数，则其默认值与--base_model相同
- `--template_name`：模板名称，如`vicuna`、`alpaca`等。若不提供此参数，则其默认值是vicuna
- `--only_cpu`: 仅使用CPU进行推理
- `--resize_emb`：是否调整embedding大小，若不调整，则使用预训练模型的embedding大小，默认不调整


## 💾 Install
#### Updating the requirements

```markdown
git clone https://github.com/shibing624/MedicalGPT
cd MedicalGPT
pip install -r requirements.txt --upgrade
```

#### Hardware Requirement (显存/VRAM)


\* *估算值*

| 训练方法  | 精度          |   7B  |  13B  |  30B  |   70B  |  110B  |  8x7B |  8x22B |
|-------|-------------| ----- | ----- | ----- | ------ | ------ | ----- | ------ |
| 全参数   | AMP(自动混合精度) | 120GB | 240GB | 600GB | 1200GB | 2000GB | 900GB | 2400GB |
| 全参数   | 16          |  60GB | 120GB | 300GB |  600GB |  900GB | 400GB | 1200GB |
| LoRA  | 16          |  16GB |  32GB |  64GB |  160GB |  240GB | 120GB |  320GB |
| QLoRA | 8           |  10GB |  20GB |  40GB |   80GB |  140GB |  60GB |  160GB |
| QLoRA | 4           |   6GB |  12GB |  24GB |   48GB |   72GB |  30GB |   96GB |
| QLoRA | 2           |   4GB |   8GB |  16GB |   24GB |   48GB |  18GB |   48GB |

## 🚀 Training Pipeline

Training Stage:

| Stage                          | Introduction | Python script                                                                                           | Shell script                                                                  |
|:-------------------------------|:-------------|:--------------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------|
| Continue Pretraining           | 增量预训练        | [pretraining.py]                     | [run_pt.sh]   |
| Supervised Fine-tuning         | 有监督微调        | [supervised_finetuning.py]           | [run_sft.sh]   |
| Direct Preference Optimization | 直接偏好优化      | [dpo_training.py]                    | [run_dpo.sh]   |


- 提供基于知识库文件的LLM问答功能（RAG）：[chatpdf.py]



## 💻 Inference
训练完成后，现在我们加载训练好的模型，验证模型生成文本的效果。

```shell
CUDA_VISIBLE_DEVICES=0 python inference.py \
    --base_model path_to_model_hf_dir \
    --lora_model path_to_lora \
    --interactive
```

参数说明：

- `--base_model {base_model}`：存放HF格式的LLaMA模型权重和配置文件的目录
- `--tokenizer_path {base_model}`：存放HF格式的LLaMA模型权重和配置文件的目录
- `--lora_model {lora_model}`：LoRA解压后文件所在目录，也可使用HF Model Hub模型调用名称。如果已经合并了LoRA权重到预训练模型，则可以不提供此参数
- `--tokenizer_path {tokenizer_path}`：存放对应tokenizer的目录。若不提供此参数，则其默认值与--base_model相同
- `--template_name`：模板名称，如`vicuna`、`alpaca`等。若不提供此参数，则其默认值是vicuna
- `--interactive`：以交互方式启动多轮问答，使用流式推理
- `--data_file {file_name}`：非交互方式启动下，读取file_name中的的内容进行batch预测
- `--output_file {file_name}`：非交互式方式下，将预测的结果以jsonl格式写入file_name
- `--resize_emb`：是否调整embedding大小，若不调整，则使用预训练模型的embedding大小，默认不调整
- `--only_cpu`：仅使用CPU进行推理
- `--gpus {gpu_ids}`：指定使用的GPU设备编号，默认为0。如使用多张GPU，以逗号分隔，如0,1,2

#### 多卡推理
多卡数据并行，batch推理
```shell
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node 2 inference_multigpu_demo.py --base_model shibing624/vicuna-baichuan-13b-chat
```
#### vllm多卡部署
```shell
sh vllm_deployment.sh
```


## 📚 Dataset
### 医疗数据集

- 240万条中文医疗数据集(包括预训练、指令微调和奖励数据集)：[shibing624/medical](https://huggingface.co/datasets/shibing624/medical)
- 22万条中文医疗对话数据集(华佗项目)：[shibing624/huatuo_medical_qa_sharegpt](https://huggingface.co/datasets/shibing624/huatuo_medical_qa_sharegpt) 【本项目支持格式】

### 通用数据集

#### Pretraining datasets(预训练数据集)
- 16GB中英文无监督、平行语料[Linly-AI/Chinese-pretraining-dataset](https://huggingface.co/datasets/Linly-AI/Chinese-pretraining-dataset)
- 524MB中文维基百科语料[wikipedia-cn-20230720-filtered](https://huggingface.co/datasets/pleisto/wikipedia-cn-20230720-filtered)
#### Supervised fine-tuning datasets(指令微调数据集)
- 10万条多语言ShareGPT GPT4多轮对话数据集：[shibing624/sharegpt_gpt4](https://huggingface.co/datasets/shibing624/sharegpt_gpt4) 【本项目支持格式】
- 9万条英文ShareGPT多轮对话数集：[anon8231489123/ShareGPT_Vicuna_unfiltered](https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered) 【本项目支持格式】
- 50万条中文ChatGPT指令Belle数据集：[BelleGroup/train_0.5M_CN](https://huggingface.co/datasets/BelleGroup/train_0.5M_CN)
- 100万条中文ChatGPT指令Belle数据集：[BelleGroup/train_1M_CN](https://huggingface.co/datasets/BelleGroup/train_1M_CN)
- 5万条英文ChatGPT指令Alpaca数据集：[50k English Stanford Alpaca dataset](https://github.com/tatsu-lab/stanford_alpaca#data-release)
- 2万条中文ChatGPT指令Alpaca数据集：[shibing624/alpaca-zh](https://huggingface.co/datasets/shibing624/alpaca-zh)
- 69万条中文指令Guanaco数据集(Belle50万条+Guanaco19万条)：[Chinese-Vicuna/guanaco_belle_merge_v1.0](https://huggingface.co/datasets/Chinese-Vicuna/guanaco_belle_merge_v1.0)
- 5万条英文ChatGPT多轮对话数据集：[RyokoAI/ShareGPT52K](https://huggingface.co/datasets/RyokoAI/ShareGPT52K)
- 80万条中文ChatGPT多轮对话数据集：[BelleGroup/multiturn_chat_0.8M](https://huggingface.co/datasets/BelleGroup/multiturn_chat_0.8M)
- 116万条中文ChatGPT多轮对话数据集：[fnlp/moss-002-sft-data](https://huggingface.co/datasets/fnlp/moss-002-sft-data)
- 3.8万条中文ShareGPT多轮对话数据集：[FreedomIntelligence/ShareGPT-CN](https://huggingface.co/datasets/FreedomIntelligence/ShareGPT-CN)
- 130万条中文微调数据集（汇总）：[zhuangxialie/Llama3-Chinese-Dataset](https://modelscope.cn/datasets/zhuangxialie/Llama3-Chinese-Dataset/dataPeview) 【本项目支持格式】
- 7千条中文角色扮演多轮对话数据集：[shibing624/roleplay-zh-sharegpt-gpt4-data](https://huggingface.co/datasets/shibing624/roleplay-zh-sharegpt-gpt4-data) 【本项目支持格式】

#### Preference datasets(偏好数据集)
- 2万条中英文偏好数据集：[shibing624/DPO-En-Zh-20k-Preference](https://huggingface.co/datasets/shibing624/DPO-En-Zh-20k-Preference) 【本项目支持格式】
- 原版的oasst1数据集：[OpenAssistant/oasst1](https://huggingface.co/datasets/OpenAssistant/oasst1)
- 2万条多语言oasst1的reward数据集：[tasksource/oasst1_pairwise_rlhf_reward](https://huggingface.co/datasets/tasksource/oasst1_pairwise_rlhf_reward)
- 11万条英文hh-rlhf的reward数据集：[Dahoas/full-hh-rlhf](https://huggingface.co/datasets/Dahoas/full-hh-rlhf)
- 9万条英文reward数据集(来自Anthropic's Helpful Harmless dataset)：[Dahoas/static-hh](https://huggingface.co/datasets/Dahoas/static-hh)
- 7万条英文reward数据集（来源同上）：[Dahoas/rm-static](https://huggingface.co/datasets/Dahoas/rm-static)
- 7万条繁体中文的reward数据集（翻译自rm-static）[liswei/rm-static-m2m100-zh](https://huggingface.co/datasets/liswei/rm-static-m2m100-zh)
- 7万条英文Reward数据集：[yitingxie/rlhf-reward-datasets](https://huggingface.co/datasets/yitingxie/rlhf-reward-datasets)
- 3千条中文知乎问答偏好数据集：[liyucheng/zhihu_rlhf_3k](https://huggingface.co/datasets/liyucheng/zhihu_rlhf_3k)




## 💕 Acknowledgements

- [MedicalGPT: Training Medical GPT Model]((https://github.com/shibing624/MedicalGPT))
- [Direct Preference Optimization:Your Language Model is Secretly a Reward Model](https://arxiv.org/pdf/2305.18290.pdf)
- [tloen/alpaca-lora](https://github.com/tloen/alpaca-lora/blob/main/finetune.py)
- [ymcui/Chinese-LLaMA-Alpaca](https://github.com/ymcui/Chinese-LLaMA-Alpaca)
- [hiyouga/LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)
- [dvlab-research/LongLoRA](https://github.com/dvlab-research/LongLoRA)

Thanks for their great work!


