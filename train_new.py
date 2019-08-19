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

    def clean_content(self, content):
        bracket1 = list(map(lambda x: x.regs[0][0], list(re.finditer("\n\(", content))))
        bracket2 = list(map(lambda x: x.regs[0][0], list(re.finditer("\)\n", content))))
        if len(bracket1) > 0 and len(bracket2) > 0 and len(content) - bracket1[-1] < 20 and bracket1[-1] < bracket2[-1]:
            return content[:bracket1[-1]]
        else:
            return content

    def process_data_dict(self, data_dict):
        title = data_dict["title"]
        content = data_dict["content"]
        keywords = data_dict["keywords"]
        sorted_keywords = sorted(keywords.items(), key=lambda x: x[1], reverse=True)
        keywords = list(map(lambda x: x[0], sorted_keywords))
        keyword_ids_list = []
        passage_ids_list = []
        if len(content) > self.min_length and len(keywords) > 0:
            # 处理关键词
            keyword_ids = []
            for keyword in keywords:
                keyword_tokens = self.full_tokenizer.tokenize(keyword)
                single_keyword_ids = self.full_tokenizer.convert_tokens_to_ids(keyword_tokens + ["[SEP]"])
                if len(keyword_ids) + len(single_keyword_ids) < self.keywords_max_length:
                    keyword_ids.extend(single_keyword_ids)
            keyword_ids.extend(self.full_tokenizer.convert_tokens_to_ids(["[PAD]"] * (self.keywords_max_length - len(keyword_ids))))
            # 处理正文
            content = self.clean_content(content)
            passage = title + "\n" + content
            passage = passage.replace('\n', ' [SEP] ')
            passage = ' [MASK] ' + passage + ' [CLS] '  # [MASK] 表示文章开头，[CLS] 表示文章结束
            passage_tokens = self.full_tokenizer.tokenize(passage)
            while len(passage_tokens) > self.passage_min_length - 1:
                current_tokens = passage_tokens[: self.passage_max_length - 1]
                if len(passage_tokens) > self.passage_max_length - 1:
                    passage_tokens = passage_tokens[self.passage_max_length - 1:]
                else:
                    passage_tokens = []
                if current_tokens[-1] != "[SEP]": current_tokens += ["[SEP]"]
                passage_ids = self.full_tokenizer.convert_tokens_to_ids(current_tokens)
                passage_ids.extend(self.full_tokenizer.convert_tokens_to_ids(["[PAD]"] * (self.passage_max_length - len(passage_ids))))
                keyword_ids_list.append(keyword_ids)
                passage_ids_list.append(passage_ids)
        return keyword_ids_list, passage_ids_list

    def create_dataloader(self):
        total_keyword_ids_list = []
        total_passage_ids_list = []
        with open(self.raw_data_path, "r") as f:
            for i, line in enumerate(f):
                if self.debug_mode and i == 10: break
                data_dict = json.loads(line)
                if (i + 1) % 10000 == 0: print("已加载训练样本 %d" % (i + 1))
                keyword_ids_list, passage_ids_list = self.process_data_dict(data_dict)
                if len(keyword_ids_list) != 0 and len(passage_ids_list) != 0:
                    total_keyword_ids_list.extend(keyword_ids_list)
                    total_passage_ids_list.extend(passage_ids_list)
        train_dataset = data.TensorDataset(torch.tensor(total_keyword_ids_list, dtype=torch.long),
                                           torch.tensor(total_passage_ids_list, dtype=torch.long))
        if torch.cuda.is_available():
            pin_memory = True
            num_workers = 7
        else:
            pin_memory = False
            num_workers = 3
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, pin_memory=pin_memory, num_workers=num_workers)
        return train_loader

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
        print('模型参数量: {}'.format(num_parameters))

        print("开始加载训练集")
        train_loader = self.create_dataloader()
        print("训练集加载完毕")

        epoch_steps = int(train_loader.sampler.num_samples / self.batch_size / self.accumulation_steps)
        total_steps = epoch_steps * self.epochs
        print('epoch 步数 = {}'.format(epoch_steps))
        print('总步数 = {}'.format(total_steps))

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
        f_log = open("train_log.txt", "w")
        for epoch in range(self.epochs):
            print('epoch {}'.format(epoch + 1))
            now = datetime.now()
            print('time: {}'.format(now))
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

                if multi_gpu:
                    loss = loss.mean()

                if self.gradient_accumulation > 1:
                    loss = loss / self.gradient_accumulation

                #  loss backward
                if self.fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), self.max_grad_norm)
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)

                if (i + 1) % self.gradient_accumulation == 0:
                    running_loss += loss.item()
                    scheduler.step()
                    optimizer.step()
                    optimizer.zero_grad()
                    overall_step += 1
                    #if (overall_step + 1) % self.log_step == 0:
                    #    self.tb_writer.add_scalar('loss', loss.item(), overall_step)

                if (overall_step + 1) % self.log_step == 0 and running_loss != 0:
                    print('now time: {}:{}. Step {} of epoch {}, loss {}'.format(
                        datetime.now().hour,
                        datetime.now().minute,
                        overall_step + 1,
                        epoch + 1,
                        running_loss * self.gradient_accumulation / self.log_step))
                    f_log.write('now time: {}:{}. Step {} of epoch {}, loss {}\n'.format(
                        datetime.now().hour,
                        datetime.now().minute,
                        overall_step + 1,
                        epoch + 1,
                        running_loss * self.gradient_accumulation / self.log_step))
                    running_loss = 0

            if not os.path.exists(self.output_dir + 'model_epoch{}'.format(epoch + 1)):
                os.makedirs(self.output_dir + 'model_epoch{}'.format(epoch + 1))
            model_to_save = model.module if hasattr(model, 'module') else model
            model_to_save.save_pretrained(self.output_dir + 'model_epoch{}'.format(epoch + 1))
            # torch.save(scheduler.state_dict(), output_dir + 'model_epoch{}/scheduler.pt'.format(epoch + 1))
            # torch.save(optimizer.state_dict(), output_dir + 'model_epoch{}/optimizer.pt'.format(epoch + 1))

            then = datetime.now()
            print('time: {}'.format(then))
            print('time for one epoch: {}'.format(then - now))

        print('training finished')
        if not os.path.exists(self.output_dir + 'final_model'):
            os.makedirs(self.output_dir + 'final_model')
        model_to_save = model.module if hasattr(model, 'module') else model
        model_to_save.save_pretrained(self.output_dir + 'final_model')
        # torch.save(scheduler.state_dict(), output_dir + 'final_model/scheduler.pt')
        # torch.save(optimizer.state_dict(), output_dir + 'final_model/optimizer.pt')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='0,1,2,3', type=str, required=False, help='设置使用哪些显卡')
    parser.add_argument('--model_config', default='config/model_config_small.json', type=str, required=False,
                        help='选择模型参数')
    parser.add_argument('--tokenizer_path', default='cache/vocab_small.txt', type=str, required=False, help='选择词库')
    parser.add_argument('--raw_data_path', default='data/train.json', type=str, required=False, help='原始训练语料')
    parser.add_argument('--tokenized_data_path', default='data/tokenized/', type=str, required=False,
                        help='tokenized语料存放位置')
    parser.add_argument('--raw', action='store_true', help='是否先做tokenize')
    parser.add_argument('--epochs', default=5, type=int, required=False, help='训练循环')
    parser.add_argument('--batch_size', default=12, type=int, required=False, help='训练batch size')
    parser.add_argument('--accumulation_steps', default=1, type=int, required=False, help='梯度累加')
    parser.add_argument('--lr', default=1.5e-4, type=float, required=False, help='学习率')
    parser.add_argument('--warmup_steps', default=2000, type=int, required=False, help='warm up步数')
    parser.add_argument('--log_step', default=1000, type=int, required=False, help='多少步汇报一次loss')
    parser.add_argument('--stride', default=768, type=int, required=False, help='训练时取训练数据的窗口步长')
    parser.add_argument('--gradient_accumulation', default=1, type=str, required=False, help='梯度积累')
    parser.add_argument('--fp16', action='store_true', help='混合精度')
    parser.add_argument('--fp16_opt_level', default='O1', type=str, required=False)
    parser.add_argument('--max_grad_norm', default=1.0, type=float, required=False)
    parser.add_argument('--num_pieces', default=100, type=int, required=False, help='将训练语料分成多少份')
    parser.add_argument('--min_length', default=128, type=int, required=False, help='最短收录文章长度')
    parser.add_argument('--output_dir', default='model/', type=str, required=False, help='模型输出路径')
    parser.add_argument('--pretrained_model', default='', type=str, required=False, help='模型训练起点路径')
    parser.add_argument('--writer_dir', default='tensorboard_summary/', type=str, required=False, help='Tensorboard路径')
    parser.add_argument('--no_wordpiece', action='store_true', help='不做word piece切词')
    parser.add_argument('--segment', action='store_true', help='中文以词为单位')
    args = parser.parse_args()
    trainer = GPT2Trainer(args, debug_mode=False)
    trainer.train()
