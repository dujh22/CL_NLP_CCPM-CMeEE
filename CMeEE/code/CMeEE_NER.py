#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import sys

import pandas as pd

sys.path.append('.')
import argparse
import torch
import numpy as np
import json
from tqdm import tqdm

from transformers import BertTokenizer, BertForSequenceClassification, AlbertForSequenceClassification, \
    BertForTokenClassification, AlbertForTokenClassification


from cblue.data import STSDataProcessor, STSDataset, QICDataset, QICDataProcessor, QQRDataset, \
    QQRDataProcessor, QTRDataset, QTRDataProcessor, CTCDataset, CTCDataProcessor, EEDataset, EEDataProcessor
# from cblue.trainer import STSTrainer, QICTrainer, QQRTrainer, QTRTrainer, CTCTrainer, EETrainer
from cblue.utils import init_logger, seed_everything
from cblue.models import ZenConfig, ZenNgramDict, ZenForSequenceClassification, ZenForTokenClassification

from cblue.metrics import sts_metric, qic_metric, qqr_metric, qtr_metric, \
    ctc_metric, ee_metric, er_metric, re_metric, cdn_cls_metric, cdn_num_metric

from cblue.utils import seed_everything, ProgressBar, TokenRematch
from cblue.models import convert_examples_to_features, save_zen_model
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup



class Trainer(object):
    def __init__(
            self,
            args,
            model,
            data_processor,
            tokenizer,
            logger,
            model_class,
            train_dataset=None,
            eval_dataset=None,
            ngram_dict=None
    ):

        self.args = args
        self.model = model
        self.data_processor = data_processor
        self.tokenizer = tokenizer

        if train_dataset is not None and isinstance(train_dataset, Dataset):
            self.train_dataset = train_dataset

        if eval_dataset is not None and isinstance(eval_dataset, Dataset):
            self.eval_dataset = eval_dataset

        self.logger = logger
        self.model_class = model_class
        self.ngram_dict = ngram_dict

    def train(self):
        args = self.args
        logger = self.logger
        model = self.model
        model.to(args.device)

        train_dataloader = self.get_train_dataloader()

        num_training_steps = len(train_dataloader) * args.epochs
        num_warmup_steps = num_training_steps * args.warmup_proportion
        num_examples = len(train_dataloader.dataset)

        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': self.args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps,
                                                    num_training_steps=num_training_steps)

        if args.task_name in ['qic', 'qqr', 'qtr', 'sts']:
            seed_everything(args.seed)
            model.zero_grad()

        logger.info("***** Running training *****")
        logger.info("Num samples %d", num_examples)
        logger.info("Num epochs %d", args.epochs)
        logger.info("Num training steps %d", num_training_steps)
        logger.info("Num warmup steps %d", num_warmup_steps)

        global_step = 0
        best_step = None
        best_score = .0
        cnt_patience = 0
        for i in range(args.epochs):
            pbar = ProgressBar(n_total=len(train_dataloader), desc='Training')
            for step, item in enumerate(train_dataloader):
                loss = self.training_step(model, item)
                pbar(step, {'loss': loss.item()})

                if args.max_grad_norm:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()
                if args.task_name in ['qic', 'qqr', 'qtr', 'sts']:
                    model.zero_grad()
                else:
                    optimizer.zero_grad()

                global_step += 1

                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    print("")
                    score = self.evaluate(model)
                    if score > best_score:
                        best_score = score
                        best_step = global_step
                        cnt_patience = 0
                        self._save_checkpoint(model, global_step)
                    else:
                        cnt_patience += 1
                        self.logger.info("Earlystopper counter: %s out of %s", cnt_patience, args.earlystop_patience)
                        if cnt_patience >= self.args.earlystop_patience:
                            break
            if cnt_patience >= args.earlystop_patience:
                break

        logger.info("Training Stop! The best step %s: %s", best_step, best_score)
        if args.device == 'cuda':
            torch.cuda.empty_cache()

        self._save_best_checkpoint(best_step=best_step)

        return global_step, best_step

    def evaluate(self, model):
        raise NotImplementedError

    def _save_checkpoint(self, model, step):
        raise NotImplementedError

    def _save_best_checkpoint(self, best_step):
        raise NotImplementedError

    def training_step(self, model, item):
        raise NotImplementedError

    def get_train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            shuffle=True
        )

    def get_eval_dataloader(self):
        return DataLoader(
            self.eval_dataset,
            batch_size=self.args.eval_batch_size,
            shuffle=False
        )

    def get_test_dataloader(self, test_dataset, batch_size=None):
        if not batch_size:
            batch_size = self.args.eval_batch_size

        return DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False
        )


class EETrainer(Trainer):
    def __init__(
            self,
            args,
            model,
            data_processor,
            tokenizer,
            logger,
            model_class,
            train_dataset=None,
            eval_dataset=None,
            ngram_dict=None
    ):
        super(EETrainer, self).__init__(
            args=args,
            model=model,
            data_processor=data_processor,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            logger=logger,
            model_class=model_class,
            ngram_dict=ngram_dict
        )

    def training_step(self, model, item):
        model.train()

        input_ids = item[0].to(self.args.device)
        token_type_ids = item[1].to(self.args.device)
        attention_mask = item[2].to(self.args.device)
        labels = item[3].to(self.args.device)

        if self.args.model_type == 'zen':
            input_ngram_ids = item[4].to(self.args.device)
            ngram_attention_mask = item[5].to(self.args.device)
            ngram_token_type_ids = item[6].to(self.args.device)
            ngram_position_matrix = item[7].to(self.args.device)

        if self.args.model_type == 'zen':
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                            labels=labels.long(), ngram_ids=input_ngram_ids, ngram_positions=ngram_position_matrix,
                            ngram_attention_mask=ngram_attention_mask, ngram_token_type_ids=ngram_token_type_ids)
        else:
            outputs = model(labels=labels.long(), input_ids=input_ids, token_type_ids=token_type_ids,
                            attention_mask=attention_mask)

        loss = outputs[0]
        loss.backward()

        return loss.detach()

    def evaluate(self, model):
        args = self.args
        logger = self.logger
        eval_dataloader = self.get_eval_dataloader()
        num_examples = len(eval_dataloader.dataset)

        preds = None
        eval_labels = None

        logger.info("***** Running evaluation *****")
        logger.info("Num samples %d", num_examples)
        for step, item in enumerate(eval_dataloader):
            model.eval()

            input_ids = item[0].to(self.args.device)
            token_type_ids = item[1].to(self.args.device)
            attention_mask = item[2].to(self.args.device)
            labels = item[3].to(self.args.device)

            if args.model_type == 'zen':
                input_ngram_ids = item[4].to(self.args.device)
                ngram_attention_mask = item[5].to(self.args.device)
                ngram_token_type_ids = item[6].to(self.args.device)
                ngram_position_matrix = item[7].to(self.args.device)

            with torch.no_grad():
                if self.args.model_type == 'zen':
                    outputs = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask,
                                    labels=labels.long(), ngram_ids=input_ngram_ids,
                                    ngram_positions=ngram_position_matrix,
                                    ngram_token_type_ids=ngram_token_type_ids,
                                    ngram_attention_mask=ngram_attention_mask)
                else:
                    outputs = model(labels=labels.long(), input_ids=input_ids, token_type_ids=token_type_ids,
                                    attention_mask=attention_mask)

                # outputs = model(labels=labels, **inputs)
                loss, logits = outputs[:2]
                # active_index = inputs['attention_mask'].view(-1) == 1
                active_index = attention_mask.view(-1) == 1
                active_labels = labels.view(-1)[active_index]
                logits = logits.argmax(dim=-1)
                active_logits = logits.view(-1)[active_index]

            if preds is None:
                preds = active_logits.detach().cpu().numpy()
                eval_labels = active_labels.detach().cpu().numpy()
            else:
                preds = np.append(preds, active_logits.detach().cpu().numpy(), axis=0)
                eval_labels = np.append(eval_labels, active_labels.detach().cpu().numpy(), axis=0)

        p, r, f1, _ = ee_metric(preds, eval_labels)
        # logger.info("%s-%s precision: %s - recall: %s - f1 score: %s", args.task_name, args.model_name, p, r, f1)
        return f1

    def predict(self, model, test_dataset):
        args = self.args
        logger = self.logger
        test_dataloader = self.get_test_dataloader(test_dataset)
        num_examples = len(test_dataloader.dataset)
        model.to(args.device)

        predictions = []

        # logger.info("***** Running prediction *****")
        # logger.info("Num samples %d", num_examples)
        pbar = ProgressBar(n_total=len(test_dataloader), desc='Prediction')
        for step, item in enumerate(test_dataloader):
            model.eval()

            input_ids = item[0].to(self.args.device)
            token_type_ids = item[1].to(self.args.device)
            attention_mask = item[2].to(self.args.device)

            if args.model_type == 'zen':
                input_ngram_ids = item[3].to(self.args.device)
                ngram_attention_mask = item[4].to(self.args.device)
                ngram_token_type_ids = item[5].to(self.args.device)
                ngram_position_matrix = item[6].to(self.args.device)

            with torch.no_grad():
                if self.args.model_type == 'zen':
                    outputs = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask,
                                    ngram_ids=input_ngram_ids,
                                    ngram_positions=ngram_position_matrix,
                                    ngram_token_type_ids=ngram_token_type_ids,
                                    ngram_attention_mask=ngram_attention_mask)
                else:
                    outputs = model(input_ids=input_ids, token_type_ids=token_type_ids,
                                    attention_mask=attention_mask)

                if args.model_type == 'zen':
                    logits = outputs.detach()
                else:
                    logits = outputs[0].detach()
                # active_index = (inputs['attention_mask'] == 1).cpu()
                active_index = attention_mask == 1
                preds = logits.argmax(dim=-1).cpu()

                for i in range(len(active_index)):
                    predictions.append(preds[i][active_index[i]].tolist())
            pbar(step=step, info="")

        # test_inputs = [list(text) for text in test_dataset.texts]
        test_inputs = test_dataset.texts
        predictions = [pred[1:-1] for pred in predictions]
        predicts = self.data_processor.extract_result(predictions, test_inputs)

        # ee_commit_prediction(dataset=test_dataset, preds=predicts, output_dir=args.result_output_dir)

        pred_result = {}
        pred_result['text'] = test_dataset.orig_text[0]
        pred_result['entities'] = predicts[0]

        # output = json.dumps(pred_result, indent=2, ensure_ascii=False)

        return pred_result



    def _save_checkpoint(self, model, step):
        output_dir = os.path.join(self.args.output_dir, 'checkpoint-{}'.format(step))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        if self.args.model_type == 'zen':
            save_zen_model(output_dir, model=model, tokenizer=self.tokenizer,
                           ngram_dict=self.ngram_dict, args=self.args)
        else:
            model.save_pretrained(output_dir)
            torch.save(self.args, os.path.join(output_dir, 'training_args.bin'))
            self.tokenizer.save_vocabulary(save_directory=output_dir)
        self.logger.info('Saving models checkpoint to %s', output_dir)

    def _save_best_checkpoint(self, best_step):
        model = self.model_class.from_pretrained(os.path.join(self.args.output_dir, f'checkpoint-{best_step}'),
                                                 num_labels=self.data_processor.num_labels)
        if self.args.model_type == 'zen':
            save_zen_model(self.args.output_dir, model=model, tokenizer=self.tokenizer,
                           ngram_dict=self.ngram_dict, args=self.args)
        else:
            model.save_pretrained(self.args.output_dir)
            torch.save(self.args, os.path.join(self.args.output_dir, 'training_args.bin'))
            self.tokenizer.save_vocabulary(save_directory=self.args.output_dir)
        self.logger.info('Saving models checkpoint to %s', self.args.output_dir)


TASK_DATASET_CLASS = {
    'ee': (EEDataset, EEDataProcessor),
    # 'ctc': (CTCDataset, CTCDataProcessor),
    # 'sts': (STSDataset, STSDataProcessor),
    # 'qqr': (QQRDataset, QQRDataProcessor),
    # 'qtr': (QTRDataset, QTRDataProcessor),
    # 'qic': (QICDataset, QICDataProcessor)
}

TASK_TRAINER = {
    'ee': EETrainer,
    # 'ctc': CTCTrainer,
    # 'sts': STSTrainer,
    # 'qic': QICTrainer,
    # 'qqr': QQRTrainer,
    # 'qtr': QTRTrainer
}

MODEL_CLASS = {
    'bert': (BertTokenizer, BertForSequenceClassification),
    'roberta': (BertTokenizer, BertForSequenceClassification),
    # 'albert': (BertTokenizer, AlbertForSequenceClassification),
    # 'zen': (BertTokenizer, ZenForSequenceClassification)
}

TOKEN_MODEL_CLASS = {
    'bert': (BertTokenizer, BertForTokenClassification),
    'roberta': (BertTokenizer, BertForTokenClassification),
    # 'albert': (BertTokenizer, AlbertForTokenClassification),
    # 'zen': (BertTokenizer, ZenForTokenClassification)
}


def loadEEmodel():
    parser = argparse.ArgumentParser()
    '''
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The task data directory.")
    parser.add_argument("--model_dir", default=None, type=str, required=True,
                        help="The directory of pretrained models")
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="The type of selected pretrained models.")
    parser.add_argument("--model_name", default=None, type=str, required=True,
                        help="The path of selected pretrained models. (e.g. chinese-bert-wwm)")
    parser.add_argument("--task_name", default=None, type=str, required=True,
                        help="The name of task to train")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The path of result data and models to be saved.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_predict", action='store_true',
                        help="Whether to run the models in inference mode on the test set.")
    parser.add_argument("--result_output_dir", default=None, type=str, required=True,
                        help="the directory of commit result to be saved")
    '''
    # models param
    parser.add_argument("--max_length", default=128, type=int,
                        help="the max length of sentence.")
    parser.add_argument("--train_batch_size", default=8, type=int,
                        help="Batch size for training.")
    parser.add_argument("--eval_batch_size", default=8, type=int,
                        help="Batch size for evaluation.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.01, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--epochs", default=3, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for, "
                             "E.g., 0.1 = 10% of training.")
    parser.add_argument("--earlystop_patience", default=2, type=int,
                        help="The patience of early stop")

    parser.add_argument('--logging_steps', type=int, default=10,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=1000,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument('--seed', type=int, default=2021,
                        help="random seed for initialization")

    args = parser.parse_args()

    args.data_dir = '../CBLUEDatasets'
    args.model_type = 'bert'
    args.model_name = 'chinese_roberta_wwm_large_ext'
    args.model_dir = '../data/model_data'
    args.task_name = 'ee'
    args.output_dir = '../data/output'
    args.result_output_dir = '../data/result_output'
    args.do_predict = True
    args.do_train = False


    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    args.output_dir = os.path.join(args.output_dir, args.task_name)
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    args.output_dir = os.path.join(args.output_dir, args.model_name)
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    if not os.path.exists(args.result_output_dir):
        os.mkdir(args.result_output_dir)

    logger = init_logger(os.path.join(args.output_dir, f'{args.task_name}_{args.model_name}.log'))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    seed_everything(args.seed)

    if 'albert' in args.model_name:
        args.model_type = 'albert'

    tokenizer_class, model_class = MODEL_CLASS[args.model_type]
    dataset_class, data_processor_class = TASK_DATASET_CLASS[args.task_name]
    trainer_class = TASK_TRAINER[args.task_name]

    if args.task_name == 'ee':
        tokenizer_class, model_class = TOKEN_MODEL_CLASS[args.model_type]

    logger.info("Training/evaluation parameters %s", args)

    if args.do_train:
        tokenizer = tokenizer_class.from_pretrained(os.path.join(args.model_dir, args.model_name))

        # compatible with 'ZEN' model
        ngram_dict = None
        if args.model_type == 'zen':
            ngram_dict = ZenNgramDict(os.path.join(args.model_dir, args.model_name), tokenizer=tokenizer)

        data_processor = data_processor_class(root=args.data_dir)
        train_samples = data_processor.get_train_sample()
        eval_samples = data_processor.get_dev_sample()

        if args.task_name == 'ee' or args.task_name == 'ctc':
            train_dataset = dataset_class(train_samples, data_processor, tokenizer, mode='train',
                                          model_type=args.model_type, ngram_dict=ngram_dict, max_length=args.max_length)
            eval_dataset = dataset_class(eval_samples, data_processor, tokenizer, mode='eval',
                                         model_type=args.model_type, ngram_dict=ngram_dict, max_length=args.max_length)
        else:
            train_dataset = dataset_class(train_samples, data_processor, mode='train')
            eval_dataset = dataset_class(eval_samples, data_processor, mode='eval')

        model = model_class.from_pretrained(os.path.join(args.model_dir, args.model_name),
                                            num_labels=data_processor.num_labels)

        trainer = trainer_class(args=args, model=model, data_processor=data_processor,
                                tokenizer=tokenizer, train_dataset=train_dataset, eval_dataset=eval_dataset,
                                logger=logger, model_class=model_class, ngram_dict=ngram_dict)

        global_step, best_step = trainer.train()

    if args.do_predict:
        tokenizer = tokenizer_class.from_pretrained(args.output_dir)
        data_processor = data_processor_class(root=args.data_dir)

        ngram_dict = None
        if args.model_type == 'zen':
            ngram_dict = ZenNgramDict(os.path.join(args.model_dir, args.model_name), tokenizer=tokenizer)

        model = model_class.from_pretrained(args.output_dir, num_labels=data_processor.num_labels)
        trainer = trainer_class(args=args, model=model, data_processor=data_processor,
                                tokenizer=tokenizer, logger=logger, model_class=model_class, ngram_dict=ngram_dict)


        return model, trainer, tokenizer ,data_processor
        '''
        ngram_dict = None
        if args.model_type == 'zen':
            ngram_dict = ZenNgramDict(os.path.join(args.model_dir, args.model_name), tokenizer=tokenizer)

        data_processor = data_processor_class(root=args.data_dir)
        test_samples = data_processor.get_test_sample()

        if args.task_name == 'ee' or args.task_name == 'ctc':
            test_dataset = dataset_class(test_samples, data_processor, tokenizer, mode='test', ngram_dict=ngram_dict,
                                         max_length=args.max_length, model_type=args.model_type)
        else:
            test_dataset = dataset_class(test_samples, data_processor, mode='test')

        model = model_class.from_pretrained(args.output_dir, num_labels=data_processor.num_labels)
        trainer = trainer_class(args=args, model=model, data_processor=data_processor,
                                tokenizer=tokenizer, logger=logger, model_class=model_class, ngram_dict=ngram_dict)
        trainer.predict(test_dataset=test_dataset, model=model)
        '''



def useNERModelToPredict(model, trainer, tokenizer, data_processor, inputstring):
    # 以下是OnlineService需要更改的部分
    # inputstring = '患者诉右下腹腹痛，要求住院'
    # inputstring = orgitext
    MAX_LENGTH = 500
    MODEL_TYPE='bert'

    # 按原算法要求将输入文本封装成dict
    input_data = {}

    text_a = ["，" if t == " " or t == "\n" or t == "\t" else t
              for t in list(inputstring)]

    input_data['text'] = [text_a]
    input_data['label'] = []
    input_data['orig_text'] = [inputstring]

    # 此处获取测试数据
    # dataset_class, data_processor_class = EEDataset, EEDataProcessor

    # 构建测试集
    test_dataset = EEDataset(input_data, data_processor, tokenizer, mode='test', ngram_dict=None,
                                 max_length=MAX_LENGTH, model_type=MODEL_TYPE)


    # 预测
    trainer.predict(test_dataset=test_dataset, model=model)
    pred_result = trainer.predict(test_dataset=test_dataset, model=model)


    return pred_result
    # output = json.dumps(pred_result, indent=2, ensure_ascii=False)


# 线上运行的代码
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources = r'/*')

# 接口
@app.route('/getMedicalNER',methods=['POST'])
def getMedicalNER():
    if request.method == "POST":
        # data = request.data
        inputstring = request.form.get("inputstring")

        return_dict = {}
        return_dict['result'] = 0
        return_dict['message'] = "Success"

        pred_result = useNERModelToPredict(model, trainer, tokenizer, data_processor, inputstring)
        return_dict['data']={}
        return_dict['data']['MedicalNER'] = pred_result

        # NER_json = json.dumps(pred_result, indent=2, ensure_ascii=False)


        return  json.dumps(return_dict)



if __name__ == '__main__':

    model, trainer, tokenizer ,data_processor = loadEEmodel()
    inputstring ='患者40年前发现血压升高，最高血压160/100mmHg，规律服用苯磺酸氨氯地平片2.5mgQd控制血压，血压波动在130/80mmHg左右；2015年于我院诊断为“脑梗死”，恢复尚可，生活不能自理；否认肝炎、结核、疟疾病史，否认糖尿病、精神疾病史，否认外伤史，否认食物、药物过敏史，预防接种史不详。'
    result = useNERModelToPredict(model, trainer, tokenizer, data_processor, inputstring)
    app.run(host='192.168.0.123',port=9285)