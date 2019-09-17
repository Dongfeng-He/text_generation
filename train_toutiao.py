import pytorch_transformers
from model import GPT2KWModel
import torch
from torch.utils import data
import os
import json
import re
import random
import numpy as np
import argparse
# from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from torch.nn import DataParallel
import time


class GPT2Trainer:
    def __init__(self, args, debug_mode=False):
        if args.no_wordpiece:
            from tokenizations import tokenization_bert_without_wordpiece as tokenization_bert
        elif args.segment:
            from tokenizations import tokenization_bert_word_level as tokenization_bert
        else:
            from tokenizations import tokenization_bert
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device  # 此处设置程序使用哪些显卡
        self.model_config = pytorch_transformers.modeling_gpt2.GPT2Config.from_json_file(args.model_config)
        self.n_ctx = self.model_config.n_ctx
        self.full_tokenizer = tokenization_bert.BertTokenizer(vocab_file=args.tokenizer_path)
        self.full_tokenizer.max_len = self.n_ctx
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.raw_data_path = args.raw_data_path
        self.tokenized_data_path = args.tokenized_data_path
        self.raw = args.raw  # 选择是否从零开始构建数据集
        self.epochs = args.epochs
        self.batch_size = args.batch_size
        self.lr = args.lr
        self.warmup_steps = args.warmup_steps
        self.log_step = args.log_step
        self.stride = args.stride
        self.gradient_accumulation = args.gradient_accumulation
        self.fp16 = args.fp16  # 不支持半精度的显卡请勿打开
        self.fp16_opt_level = args.fp16_opt_level
        self.max_grad_norm = args.max_grad_norm
        self.num_pieces = args.num_pieces
        self.min_length = args.min_length
        self.output_dir = args.output_dir
        self.pretrained_model = args.pretrained_model
        self.accumulation_steps = args.accumulation_steps
        # self.tb_writer = SummaryWriter(log_dir=args.writer_dir)
        self.debug_mode = debug_mode
        self.keywords_max_length = 64
        self.passage_max_length = 512
        self.passage_min_length = 128
        self.f_log = open("train_log.txt", "w")

    def process_data_dict(self, data_dict):
        title = data_dict["title"]
        content = data_dict["content"]
        keywords = data_dict["keywords"]
        sorted_keywords = sorted(keywords.items(), key=lambda x: x[1], reverse=True)
        keywords = list(map(lambda x: x[0].strip(), sorted_keywords))
        keywords = list(filter(lambda x: len(x) > 1, keywords))
        passage_ids_list = []
        keyword_ids_list = []
        if len(content) > self.min_length and len(keywords) > 0:
            # 如果标题为空字符，不要加上\n
            passage = title + "\n" + content if len(title) > 0 else content
            passage = passage.replace('\n', ' [SEP] ')
            passage = ' [MASK] ' + passage + ' [CLS] '  # [MASK] 表示文章开头，[CLS] 表示文章结束
            passage_tokens = self.full_tokenizer.tokenize(passage)
            keyword_tokens_list = [self.full_tokenizer.tokenize(keyword) for keyword in keywords]
            # 按 stride 分割文章
            start_point = 0
            sample_passage_list = []
            sample_keywords_list = []
            while start_point < len(passage_tokens) - self.passage_max_length:
                sample_passage = passage_tokens[start_point: start_point + self.passage_max_length]
                sample_keywords = list(filter(lambda x: " ".join(x) in " ".join(sample_passage), keyword_tokens_list))
                sample_passage_list.append(sample_passage)
                sample_keywords_list.append(sample_keywords)
                start_point += self.stride
            else:
                sample_passage = passage_tokens[-self.passage_max_length:]
                sample_keywords = list(filter(lambda x: " ".join(x) in " ".join(sample_passage), keyword_tokens_list))
                sample_passage_list.append(sample_passage)
                sample_keywords_list.append(sample_keywords)
            # 把 token 变 id
            for sample_passage, sample_keywords in zip(sample_passage_list, sample_keywords_list):
                passage_ids = self.full_tokenizer.convert_tokens_to_ids(sample_passage + ["[PAD]"] * (self.passage_max_length - len(sample_passage)))
                keyword_ids = []
                for keyword_tokens in sample_keywords:
                    single_keyword_ids = self.full_tokenizer.convert_tokens_to_ids(keyword_tokens + ["[SEP]"])
                    if len(keyword_ids) + len(single_keyword_ids) < self.keywords_max_length:
                        keyword_ids.extend(single_keyword_ids)
                keyword_ids.extend(self.full_tokenizer.convert_tokens_to_ids(["[PAD]"] * (self.keywords_max_length - len(keyword_ids))))
                passage_ids_list.append(passage_ids)
                keyword_ids_list.append(keyword_ids)
        return keyword_ids_list, passage_ids_list

    def create_dataloader(self):
        total_keyword_ids_list = []
        total_passage_ids_list = []
        with open(self.raw_data_path, "r") as f:
            for i, line in enumerate(f):
                if self.debug_mode and i == 200: break
                data_dict = json.loads(line)
                if (i + 1) % 10000 == 0: self.print_and_log("已加载训练样本 %d" % (i + 1))
                keyword_ids_list, passage_ids_list = self.process_data_dict(data_dict)
                if len(keyword_ids_list) != 0 and len(passage_ids_list) != 0:
                    total_keyword_ids_list.extend(keyword_ids_list)
                    total_passage_ids_list.extend(passage_ids_list)
        # 打乱，划分训练集和验证集
        random.seed(1234)
        random.shuffle(total_keyword_ids_list)
        random.seed(1234)
        random.shuffle(total_passage_ids_list)
        valid_num = int(len(total_keyword_ids_list) * 0.02)
        train_keyword_ids_list = total_keyword_ids_list[:-valid_num]
        train_passage_ids_list = total_passage_ids_list[:-valid_num]
        valid_keyword_ids_list = total_keyword_ids_list[-valid_num:]
        valid_passage_ids_list = total_passage_ids_list[-valid_num:]
        # 建立 dataset
        train_dataset = data.TensorDataset(torch.tensor(train_keyword_ids_list, dtype=torch.long),
                                           torch.tensor(train_passage_ids_list, dtype=torch.long))
        valid_dataset = data.TensorDataset(torch.tensor(valid_keyword_ids_list, dtype=torch.long),
                                           torch.tensor(valid_passage_ids_list, dtype=torch.long))
        if torch.cuda.is_available():
            pin_memory = True
            num_workers = 7
        else:
            pin_memory = False
            num_workers = 3
        # 建立 dataloader
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, pin_memory=pin_memory, num_workers=num_workers)
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=False, pin_memory=pin_memory, num_workers=num_workers)
        return train_loader, valid_loader

    def print_and_log(self, text):
        print(text)
        self.f_log.write(text + "\n")
        self.f_log.flush()

    def train(self):
        if not self.pretrained_model:
            model = GPT2KWModel(config=self.model_config)
        else:
            model = GPT2KWModel.from_pretrained(self.pretrained_model)
        model.train()
        model.to(self.device)
        # 计算模型参数量
        num_parameters = 0
        parameters = model.parameters()
        for parameter in parameters:
            num_parameters += parameter.numel()
        self.print_and_log('模型参数量: {}'.format(num_parameters))

        self.print_and_log("开始加载训练集")
        train_loader, valid_loader = self.create_dataloader()
        self.print_and_log("训练集加载完毕")

        epoch_steps = int(train_loader.sampler.num_samples / self.batch_size / self.accumulation_steps)
        total_steps = epoch_steps * self.epochs
        self.print_and_log('epoch 步数 = {}'.format(epoch_steps))
        self.print_and_log('总步数 = {}'.format(total_steps))

        optimizer = pytorch_transformers.AdamW(model.parameters(), lr=self.lr, correct_bias=True)
        scheduler = pytorch_transformers.WarmupLinearSchedule(optimizer, warmup_steps=self.warmup_steps, t_total=total_steps)

        if self.fp16:
            try:
                from apex import amp
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            model, optimizer = amp.initialize(model, optimizer, opt_level=self.fp16_opt_level)

        if torch.cuda.device_count() > 1:
            model = DataParallel(model)
            multi_gpu = True
        else:
            multi_gpu = False

        overall_step = 0
        model.train()
        for epoch in range(self.epochs):
            self.print_and_log('epoch {}'.format(epoch + 1))
            now = datetime.now()
            self.print_and_log('time: {}'.format(now))
            optimizer.zero_grad()
            running_loss = 0
            for i, batch_data in enumerate(train_loader):
                if torch.cuda.is_available():
                    keyword_ids = batch_data[0].to(self.device, non_blocking=True)
                    passage_ids = batch_data[1].to(self.device, non_blocking=True)
                    label_ids = passage_ids.clone().to(self.device, non_blocking=True)
                else:
                    keyword_ids = batch_data[0]
                    passage_ids = batch_data[1]
                    label_ids = passage_ids.clone()
                outputs = model(input_ids=passage_ids, keyword_ids=keyword_ids, labels=label_ids)
                loss, logits = outputs[:2]
                # 多 GPU 训练
                if multi_gpu:
                    loss = loss.mean()
                # 梯度累加
                if self.gradient_accumulation > 1:
                    loss = loss / self.gradient_accumulation
                #  混合精度训练或正常训练
                if self.fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), self.max_grad_norm)
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)
                # 更新权重
                if (i + 1) % self.gradient_accumulation == 0:
                    running_loss += loss.item()
                    scheduler.step()
                    optimizer.step()
                    optimizer.zero_grad()
                    overall_step += 1
                # 报告 train loss
                if (overall_step + 1) % self.log_step == 0 and running_loss != 0:
                    self.print_and_log('now time: {}:{}. Step {} of epoch {}, loss {}'.format(
                        datetime.now().hour,
                        datetime.now().minute,
                        overall_step + 1,
                        epoch + 1,
                        running_loss * self.gradient_accumulation / self.log_step))
                    running_loss = 0

            # 开始验证
            with torch.no_grad():
                valid_start_time = datetime.now()
                model.eval()
                valid_loss = 0
                valid_step = 0
                for i, valid_batch_data in enumerate(valid_loader):
                    if torch.cuda.is_available():
                        keyword_ids = valid_batch_data[0].to(self.device, non_blocking=True)
                        passage_ids = valid_batch_data[1].to(self.device, non_blocking=True)
                        label_ids = passage_ids.clone().to(self.device, non_blocking=True)
                    else:
                        keyword_ids = valid_batch_data[0]
                        passage_ids = valid_batch_data[1]
                        label_ids = passage_ids.clone()
                    outputs = model(input_ids=passage_ids, keyword_ids=keyword_ids, labels=label_ids)
                    loss, logits = outputs[:2]
                    valid_loss += loss
                    valid_step += 1
                valid_loss = valid_loss / valid_step
                self.print_and_log('valid duration: {}, valid loss: {}'.format(datetime.now() - valid_start_time, valid_loss))

            # 保存模型
            if (epoch + 1) % 1 == 0:
                if not os.path.exists(self.output_dir + 'model_epoch{}'.format(epoch + 1)):
                    os.makedirs(self.output_dir + 'model_epoch{}'.format(epoch + 1))
                model_to_save = model.module if hasattr(model, 'module') else model
                model_to_save.save_pretrained(self.output_dir + 'model_epoch{}'.format(epoch + 1))
                # torch.save(scheduler.state_dict(), output_dir + 'model_epoch{}/scheduler.pt'.format(epoch + 1))
                # torch.save(optimizer.state_dict(), output_dir + 'model_epoch{}/optimizer.pt'.format(epoch + 1))

            then = datetime.now()
            self.print_and_log('time: {}'.format(then))
            self.print_and_log('time for one epoch: {}'.format(then - now))
            model.train()

        self.print_and_log('training finished')
        self.f_log.close()
        if not os.path.exists(self.output_dir + 'final_model'):
            os.makedirs(self.output_dir + 'final_model')
        model_to_save = model.module if hasattr(model, 'module') else model
        model_to_save.save_pretrained(self.output_dir + 'final_model')
        # torch.save(scheduler.state_dict(), output_dir + 'final_model/scheduler.pt')
        # torch.save(optimizer.state_dict(), output_dir + 'final_model/optimizer.pt')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='0,1,2,3', type=str, required=False, help='设置使用哪些显卡')
    parser.add_argument('--model_config', default='config/model_config.json', type=str, required=False, help='选择模型参数')
    parser.add_argument('--tokenizer_path', default='cache/vocab.txt', type=str, required=False, help='选择词库')
    parser.add_argument('--raw_data_path', default='data/train_toutiao.json', type=str, required=False, help='原始训练语料')
    parser.add_argument('--tokenized_data_path', default='data/tokenized/', type=str, required=False, help='tokenized语料存放位置')
    parser.add_argument('--raw', action='store_true', help='是否先做tokenize')
    parser.add_argument('--epochs', default=10, type=int, required=False, help='训练循环')
    parser.add_argument('--batch_size', default=10, type=int, required=False, help='训练batch size')
    parser.add_argument('--accumulation_steps', default=1, type=int, required=False, help='梯度累加')
    parser.add_argument('--lr', default=1.5e-4, type=float, required=False, help='学习率')
    parser.add_argument('--warmup_steps', default=2000, type=int, required=False, help='warm up步数')
    parser.add_argument('--log_step', default=10000, type=int, required=False, help='多少步汇报一次loss')
    parser.add_argument('--stride', default=256, type=int, required=False, help='训练时取训练数据的窗口步长')
    parser.add_argument('--gradient_accumulation', default=1, type=str, required=False, help='梯度积累')
    parser.add_argument('--fp16', action='store_true', help='混合精度')
    parser.add_argument('--fp16_opt_level', default='O1', type=str, required=False)
    parser.add_argument('--max_grad_norm', default=1.0, type=float, required=False)
    parser.add_argument('--num_pieces', default=100, type=int, required=False, help='将训练语料分成多少份')
    parser.add_argument('--min_length', default=128, type=int, required=False, help='最短收录文章长度')
    parser.add_argument('--output_dir', default='model_toutiao/', type=str, required=False, help='模型输出路径')
    parser.add_argument('--pretrained_model', default='', type=str, required=False, help='模型训练起点路径')
    parser.add_argument('--writer_dir', default='tensorboard_summary/', type=str, required=False, help='Tensorboard路径')
    parser.add_argument('--no_wordpiece', action='store_true', help='不做word piece切词')
    parser.add_argument('--segment', action='store_true', help='中文以词为单位')
    args = parser.parse_args()
    if os.path.exists("/Volumes/移动硬盘/model/GPT2_pretrained"):
        args.fp16 = False
        args.raw_data_path = "/Volumes/移动硬盘/数据/头条历史/train_toutiao.json"
        args.pretrained_model = "/Volumes/移动硬盘/model/GPT2_pretrained"
    else:
        args.fp16 = True
        args.raw_data_path = "data/train_toutiao.json"
        args.pretrained_model = "/root/text_generation/pretrained_model/final_model"
    trainer = GPT2Trainer(args, debug_mode=True)
    auto_shutdown = False
    if auto_shutdown:
        try:
            trainer.train()
        except:
            pass
        os.system("sudo init 0")
    else:
        trainer.train()

