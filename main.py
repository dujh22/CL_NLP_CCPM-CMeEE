# 加载本代码所依赖的通用外部包
import json
from datasets import load_dataset, load_metric
import torch
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding, \
    AutoTokenizer
import numpy as np


# 加载数据集
ccpm_train = load_dataset(path="./data_datasets_CCPM/", data_files="train.jsonl")
ccpm_valid = load_dataset(path="./data_datasets_CCPM/", data_files="valid.jsonl")
ccpm_test = load_dataset(path="./data_datasets_CCPM/", data_files="test_public.jsonl")

# 数据集分裂处理
ccpm_train_split = []
for i in range(0, len(ccpm_train["train"])):
    for j in range(0, 4):
        temp_json = {"translation": ccpm_train["train"][i]["translation"]}
        temp_json["choices"] = ccpm_train["train"][i]["choices"][j]
        if j == int(ccpm_train["train"][i]["answer"]):
            temp_json["answer"] = 1
        else:
            temp_json["answer"] = 0
        ccpm_train_split.append(temp_json)

ccpm_valid_split = []
for i in range(0, len(ccpm_valid["train"])):
    for j in range(0, 4):
        temp_json = {"translation": ccpm_valid["train"][i]["translation"]}
        temp_json["choices"] = ccpm_valid["train"][i]["choices"][j]
        if j == int(ccpm_valid["train"][i]["answer"]):
            temp_json["answer"] = 1
        else:
            temp_json["answer"] = 0
        ccpm_valid_split.append(temp_json)

ccpm_test_split = []
for i in range(0, len(ccpm_test["train"])):
    for j in range(0, 4):
        temp_json = {"translation": ccpm_test["train"][i]["translation"]}
        temp_json["choices"] = ccpm_test["train"][i]["choices"][j]
        temp_json["answer"] = 0
        ccpm_test_split.append(temp_json)

## 预处理1-5：

# 1.建模思路一：将翻译和候选诗行进行拼接
ccpm_train_concat = []
for i in range(0, len(ccpm_train_split)):
    temp_json = {"text": ccpm_train_split[i]["translation"] + "=" + ccpm_train_split[i]["choices"],
                 "label": ccpm_train_split[i]["answer"]}
    ccpm_train_concat.append(temp_json)

ccpm_valid_concat = []
for i in range(0, len(ccpm_valid_split)):
    temp_json = {"text": ccpm_valid_split[i]["translation"] + "=" + ccpm_valid_split[i]["choices"],
                 "label": ccpm_valid_split[i]["answer"]}
    ccpm_valid_concat.append(temp_json)

ccpm_test_concat = []
for i in range(0, len(ccpm_test_split)):
    temp_json = {"text": ccpm_test_split[i]["translation"] + "=" + ccpm_test_split[i]["choices"],
                 "label": ccpm_test_split[i]["answer"]}
    ccpm_test_concat.append(temp_json)

# 2.加载句子标记器
tokenizer = AutoTokenizer.from_pretrained("thu-cbert-character")

# 3.定义预处理函数
# 用于标记和截断序列，使其不超过模型的最大输入长度：text
def preproccess_function(examples):
    return tokenizer(examples["text"], truncation=True)

# 4.对整个数据集进行批量化预处理
# 使用数据集映射函数将预处理函数应用于整个数据集
ccpm_train_tokenized = list(map(preproccess_function, ccpm_train_concat))
ccpm_valid_tokenized = list(map(preproccess_function, ccpm_valid_concat))

ccpm_train_tokenized2 = []
for i in range(0, len(ccpm_train_concat)):
    temp_json = {"text": ccpm_train_concat[i]["text"],
                 "input_ids": ccpm_train_tokenized[i]["input_ids"],
                 "attention_mask":ccpm_train_tokenized[i]["attention_mask"],
                 "label": ccpm_train_concat[i]["label"]}
    ccpm_train_tokenized2.append(temp_json)

ccpm_valid_tokenized2 = []
for i in range(0, len(ccpm_valid_concat)):
    temp_json = {"text": ccpm_valid_concat[i]["text"],
                 "input_ids": ccpm_valid_tokenized[i]["input_ids"],
                 "attention_mask":ccpm_valid_tokenized[i]["attention_mask"],
                 "label": ccpm_valid_concat[i]["label"]}
    ccpm_valid_tokenized2.append(temp_json)

# 5.文本动态填充到统一长度：最长元素的长度
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# 训练
# 使用AutoModelForSequenceClassification加载模型以及标签的数量
model = AutoModelForSequenceClassification.from_pretrained("thu-cbert-character", num_labels=2)

metric = load_metric("accuracy")
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    results = metric.compute(predictions=predictions, references=labels)
    return results

# 在训练参数中定义训练超参数
training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=1,
    weight_decay=0.01,

    evaluation_strategy="steps",
    eval_steps=50,
    logging_strategy="steps",
    logging_steps=50,
    save_strategy="steps",
    save_steps=1000,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    report_to="tensorboard"
)

# 将训练参数与模型、数据集、分词器和数据整理器一起传递给训练器
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=ccpm_train_tokenized2,
    eval_dataset=ccpm_valid_tokenized2,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

trainer.train()

# 模型测试
ccpm_test_tokenized = list(map(preproccess_function, ccpm_test_concat))

ccpm_test_tokenized2 = []
for i in range(0, len(ccpm_test_concat)):
    temp_json = {"text": ccpm_test_concat[i]["text"],
                 "input_ids": ccpm_test_tokenized[i]["input_ids"],
                 "attention_mask":ccpm_test_tokenized[i]["attention_mask"],
                 }
    ccpm_test_tokenized2.append(temp_json)

test_predictions = trainer.predict(ccpm_test_tokenized2)
test_predictions_argmax = np.argmax(test_predictions[0], axis=1)
test_references = np.array()
for i in range(0, len(ccpm_test_tokenized2)):
    test_references.append(ccpm_test_tokenized2[i]["label"])
metric.compute(predictions=test_predictions_argmax, references=test_references)
