"""
The system trains BERT (or any other transformer model like RoBERTa, DistilBERT etc.) on the SNLI + MultiNLI (AllNLI) dataset
with MatryoshkaLoss using MultipleNegativesRankingLoss. This trains a model at output dimensions [768, 512, 256, 128, 64].
Entailments are positive pairs and the contradiction on AllNLI dataset is added as a hard negative.
At every 10% training steps, the model is evaluated on the STS benchmark dataset at the different output dimensions.

Usage:
python matryoshka_nli.py

OR
python matryoshka_nli.py pretrained_transformer_model_name
"""

import logging
import sys
import traceback
from datetime import datetime

from datasets import load_dataset

from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    losses,
)
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator, SequentialEvaluator, SimilarityFunction
from sentence_transformers.training_args import BatchSamplers

# 设置 log level，记录信息
logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)
# 模型名称
model_name = sys.argv[1] if len(sys.argv) > 1 else "BAAI/bge-large-zh-v1.5"
batch_size = 128  # 根据显存设置 batch size，越大越好
num_train_epochs = 1  # 训练轮次
matryoshka_dims = [768, 512, 256, 128, 64]  # 俄罗斯套娃维度

# 模型存储路径
output_dir = f"output/matryoshka_nli_{model_name.replace('/', '-')}-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

# Step 1. 定义你的 SentenceTransformer 模型
model = SentenceTransformer(model_name)
# model.max_seq_length = 75  # 限制模型的最大序列长度
logging.info(model)

# Step 2. 加载 AllNLI 数据集: https://huggingface.co/datasets/sentence-transformers/all-nli
train_dataset = load_dataset("sentence-transformers/all-nli", "triplet", split="train")
eval_dataset = load_dataset("sentence-transformers/all-nli", "triplet", split="dev")
logging.info(train_dataset)
# print(train_dataset[0])
# {
#     'anchor': 'A person on a horse jumps over a broken down airplane.',
#     'negative': 'A person is at a diner, ordering an omelette.',
#     'positive': 'A person is outdoors, on a horse.'
# }
# train_dataset = train_dataset.select(range(5000))  # 限制训练集长度

# Step 3. 定义训练损失
inner_train_loss = losses.MultipleNegativesRankingLoss(model)
train_loss = losses.MatryoshkaLoss(model, inner_train_loss, matryoshka_dims=matryoshka_dims)

# Step 4. 定义 evaluators，跟踪评估损失
stsb_eval_dataset = load_dataset("sentence-transformers/stsb", split="validation")  # 评估基准数据集 STS
evaluators = []
for dim in matryoshka_dims:
    evaluators.append(
        EmbeddingSimilarityEvaluator(
            sentences1=stsb_eval_dataset["sentence1"],
            sentences2=stsb_eval_dataset["sentence2"],
            scores=stsb_eval_dataset["score"],
            main_similarity=SimilarityFunction.COSINE,
            name=f"sts-dev-{dim}",
            truncate_dim=dim,
        )
    )
dev_evaluator = SequentialEvaluator(evaluators, main_score_function=lambda scores: scores[0])

# Step 5. 设置模型训练参数
args = SentenceTransformerTrainingArguments(
    # 必填参数:
    output_dir=output_dir,
    # 可选参数:
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    warmup_ratio=0.1,
    fp16=True,  # 根据 GPU 支持设置
    bf16=False,
    batch_sampler=BatchSamplers.NO_DUPLICATES,  # .NO_DUPLICATES 筛除重复样本
    # debug 参数
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=2,
    logging_steps=100,
    run_name="matryoshka-nli",  # pip install wandb 训练可视化
)

# Step 6. 创建 trainer，训练模型
trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    loss=train_loss,
    evaluator=dev_evaluator,
)
trainer.train()

# Step 7. 模型性能评估
test_dataset = load_dataset("sentence-transformers/stsb", split="test")
evaluators = []
for dim in matryoshka_dims:
    evaluators.append(
        EmbeddingSimilarityEvaluator(
            sentences1=test_dataset["sentence1"],
            sentences2=test_dataset["sentence2"],
            scores=test_dataset["score"],
            main_similarity=SimilarityFunction.COSINE,
            name=f"sts-test-{dim}",
            truncate_dim=dim,
        )
    )
test_evaluator = SequentialEvaluator(evaluators)
test_evaluator(model)

# Step 8. 保存模型
final_output_dir = f"{output_dir}/final"
model.save(final_output_dir)

# Step 9. (可选) 将模型上传至 Hugging Face Hub，run `huggingface-cli login`
model_name = model_name if "/" not in model_name else model_name.split("/")[-1]
try:
    model.push_to_hub(f"{model_name}-nli-matryoshka")
except Exception:
    logging.error(
        f"Error uploading model to the Hugging Face Hub:\n{traceback.format_exc()}To upload it manually, you can run "
        f"`huggingface-cli login`, followed by loading the model using `model = SentenceTransformer({final_output_dir!r})` "
        f"and saving it using `model.push_to_hub('{model_name}-nli-matryoshka')`."
    )