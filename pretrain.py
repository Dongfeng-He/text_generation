import pytorch_transformers
import torch
from torch.utils import data
import os
import json
import re
import random
import numpy as np
import argparse
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
        self.n_ctx = 512
        self.full_tokenizer = tokenization_bert.BertTokenizer(vocab_file=args.tokenizer_path)
        # self.full_tokenizer.max_len = self.n_ctx
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
        self.debug_mode = debug_mode
        if os.path.exists("/Volumes/移动硬盘/"):
            self.wiki_dir = "/Volumes/移动硬盘/语料/1.中文维基"
            self.thu_news_dir = "/Users/hedongfeng/Desktop/下载/THUCNews"
            self.zhihu_path = "/Volumes/移动硬盘/语料/4.社区问答/web_text_zh_valid.json"
            self.baike_path = "/Volumes/移动硬盘/语料/3.百科问答/baike_qa_train.json"
            self.news_path = "/Volumes/移动硬盘/语料/2.新闻语料/news2016zh_train.json"
        else:
            self.wiki_dir = "/root/text_generation/data/wiki_zh"
            self.thu_news_dir = "/root/text_generation/data/THUCNews"
        self.f_log = open("train_log.txt", "w")

    def get_wiki(self):
        content_list = []
        cnt = 0
        for root, dirs, files in os.walk(self.wiki_dir):
            for dir in dirs:
                for subroot, _, files in os.walk(os.path.join(root, dir)):
                    for file in files:
                        with open(os.path.join(subroot, file), 'r', encoding='utf8') as f:
                            for line in f:
                                line = json.loads(line)
                                title = line["title"]
                                content = line["text"]
                                content = re.sub("\(.*?\)|（.*?）|\[\d*\]", "", content)
                                content = content.replace("<br>", "").replace("</div>", "").replace("-{", "") \
                                    .replace("}-", "").replace("：；", "").replace("，。", "").replace("()", "") \
                                    .replace("《》", "").replace("（）", "").replace("\n\n\n\n\n\n", "\n") \
                                    .replace("\n\n\n\n\n", "\n").replace("\n\n\n\n", "\n").replace("\n\n\n", "\n") \
                                    .replace("\n\n", "\n")
                                content = content.replace('\n', ' [SEP] ')
                                content_list.append(content)
                                cnt += 1
                                if cnt == 5000 and self.debug_mode:
                                    return content_list
        return content_list

    def get_thu_news(self):
        content_list = []
        cnt = 0
        for root, dirs, files in os.walk(self.thu_news_dir):
            for dir in dirs:
                for subroot, _, files in os.walk(os.path.join(root, dir)):
                    for file in files:
                        with open(os.path.join(subroot, file), 'r', encoding='utf8') as f:
                            lines = f.readlines()
                            content = "".join(lines)
                            content = content.replace("\u3000", "")
                            content = re.sub("\(.*?\)|（.*?）|\[\d*\]|\d、|\d ", "", content)
                            content = content.replace("\n\n\n\n\n\n", "\n").replace("\n\n\n\n\n", "\n")\
                                .replace("\n\n\n\n", "\n").replace("\n\n\n", "\n").replace("\n\n", "\n")
                            content = content.replace('\n', ' [SEP] ')
                            content_list.append(content)
                            cnt += 1
                            if cnt == 5000 and self.debug_mode:
                                return content_list

        return content_list

    def get_zhihu(self):
        with open(self.zhihu_path, 'r', encoding='utf8') as f:
            for line in f:
                line = json.loads(line)
                question = line["title"]
                topic = line["topic"]
                stars = line["star"]
                content = line["content"]
                length = len(content)
                print()

    def get_baike(self):
        with open(self.baike_path, 'r', encoding='utf8') as f:
            for line in f:
                line = json.loads(line)
                print()

    def get_news(self):
        with open(self.news_path, 'r', encoding='utf8') as f:
            for line in f:
                line = json.loads(line)
                print()

    def create_dataloader(self):
        wiki_content_list = self.get_wiki()
        thu_content_list = self.get_thu_news()
        content_list = wiki_content_list + thu_content_list
        all_ids = []
        for content in content_list:
            if len(content) < self.min_length: continue
            tokens = self.full_tokenizer.tokenize(content)
            ids = self.full_tokenizer.convert_tokens_to_ids(['[MASK]'] + tokens + ['[CLS]'])
            all_ids.extend(ids)
        self.print_and_log("总文章数: %d" % len(content_list))
        self.print_and_log("总字数: %d" % len(all_ids))
        start_point = 0
        samples = []
        while start_point < len(all_ids) - self.n_ctx:
            samples.append(all_ids[start_point: start_point + self.n_ctx])
            start_point += self.stride
        self.print_and_log("总样本数: %d" % len(samples))
        train_dataset = data.TensorDataset(torch.tensor(samples, dtype=torch.long))
        if torch.cuda.is_available():
            pin_memory = True
            num_workers = 7
        else:
            pin_memory = False
            num_workers = 3
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, pin_memory=pin_memory, num_workers=num_workers)
        return train_loader

    def print_and_log(self, text):
        print(text)
        self.f_log.write(text + "\n")
        self.f_log.flush()

    def train(self):
        if not self.pretrained_model:
            model = pytorch_transformers.modeling_gpt2.GPT2LMHeadModel(config=self.model_config)
        else:
            model = pytorch_transformers.modeling_gpt2.GPT2LMHeadModel.from_pretrained(self.pretrained_model)
        model.train()
        model.to(self.device)
        # 计算模型参数量
        num_parameters = 0
        parameters = model.parameters()
        for parameter in parameters:
            num_parameters += parameter.numel()
        self.print_and_log('模型参数量: {}'.format(num_parameters))

        self.print_and_log("开始加载训练集")
        train_loader = self.create_dataloader()
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

        for epoch in range(self.epochs):
            self.print_and_log('epoch {}'.format(epoch + 1))
            now = datetime.now()
            self.print_and_log('time: {}'.format(now))
            optimizer.zero_grad()
            running_loss = 0
            for i, batch_data in enumerate(train_loader):
                if torch.cuda.is_available():
                    passage_ids = batch_data[0].to(self.device, non_blocking=True)
                    label_ids = passage_ids.clone().to(self.device, non_blocking=True)
                else:
                    passage_ids = batch_data[0]
                    label_ids = passage_ids.clone()
                outputs = model(input_ids=passage_ids, labels=label_ids)
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

                if (overall_step + 1) % self.log_step == 0 and running_loss != 0:
                    self.print_and_log('now time: {}:{}. Step {} of epoch {}, loss {}'.format(
                        datetime.now().hour,
                        datetime.now().minute,
                        overall_step + 1,
                        epoch + 1,
                        running_loss * self.gradient_accumulation / self.log_step))
                    running_loss = 0

            if not os.path.exists(self.output_dir + 'model_epoch{}'.format(epoch + 1)):
                os.makedirs(self.output_dir + 'model_epoch{}'.format(epoch + 1))
            gpt2_model = model.transformer
            model_to_save = gpt2_model.module if hasattr(gpt2_model, 'module') else gpt2_model
            model_to_save.save_pretrained(self.output_dir + 'model_epoch{}'.format(epoch + 1))
            # torch.save(scheduler.state_dict(), output_dir + 'model_epoch{}/scheduler.pt'.format(epoch + 1))
            # torch.save(optimizer.state_dict(), output_dir + 'model_epoch{}/optimizer.pt'.format(epoch + 1))

            then = datetime.now()
            self.print_and_log('time: {}'.format(then))
            self.print_and_log('time for one epoch: {}'.format(then - now))

        self.print_and_log('training finished')
        self.f_log.close()
        if not os.path.exists(self.output_dir + 'final_model'):
            os.makedirs(self.output_dir + 'final_model')
        gpt2_model = model.transformer
        model_to_save = gpt2_model.module if hasattr(gpt2_model, 'module') else gpt2_model
        model_to_save.save_pretrained(self.output_dir + 'final_model')
        # torch.save(scheduler.state_dict(), output_dir + 'final_model/scheduler.pt')
        # torch.save(optimizer.state_dict(), output_dir + 'final_model/optimizer.pt')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='0,1,2,3', type=str, required=False, help='设置使用哪些显卡')
    parser.add_argument('--model_config', default='config/model_config.json', type=str, required=False,
                        help='选择模型参数')
    parser.add_argument('--tokenizer_path', default='cache/vocab.txt', type=str, required=False, help='选择词库')
    parser.add_argument('--raw_data_path', default='data/train.json', type=str, required=False, help='原始训练语料')
    parser.add_argument('--tokenized_data_path', default='data/tokenized/', type=str, required=False,
                        help='tokenized语料存放位置')
    parser.add_argument('--raw', action='store_true', help='是否先做tokenize')
    parser.add_argument('--epochs', default=2, type=int, required=False, help='训练循环')
    parser.add_argument('--batch_size', default=8, type=int, required=False, help='训练batch size')
    parser.add_argument('--accumulation_steps', default=1, type=int, required=False, help='梯度累加')
    parser.add_argument('--lr', default=1.5e-4, type=float, required=False, help='学习率')
    parser.add_argument('--warmup_steps', default=10000, type=int, required=False, help='warm up步数')
    parser.add_argument('--log_step', default=10000, type=int, required=False, help='多少步汇报一次loss')
    parser.add_argument('--stride', default=384, type=int, required=False, help='训练时取训练数据的窗口步长')
    parser.add_argument('--gradient_accumulation', default=1, type=str, required=False, help='梯度积累')
    parser.add_argument('--fp16', action='store_true', help='混合精度')
    parser.add_argument('--fp16_opt_level', default='O1', type=str, required=False)
    parser.add_argument('--max_grad_norm', default=1.0, type=float, required=False)
    parser.add_argument('--num_pieces', default=100, type=int, required=False, help='将训练语料分成多少份')
    parser.add_argument('--min_length', default=128, type=int, required=False, help='最短收录文章长度')
    parser.add_argument('--output_dir', default='pretrained_model/', type=str, required=False, help='模型输出路径')
    parser.add_argument('--pretrained_model', default='', type=str, required=False, help='模型训练起点路径')
    parser.add_argument('--no_wordpiece', action='store_true', help='不做word piece切词')
    parser.add_argument('--segment', action='store_true', help='中文以词为单位')
    args = parser.parse_args()
    trainer = GPT2Trainer(args, debug_mode=False)
    auto_shutdown = True
    if auto_shutdown:
        try:
            trainer.train()
        except:
            pass
        os.system("sudo init 0")
    else:
        trainer.train()

