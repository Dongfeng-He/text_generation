import torch
import pytorch_transformers
import torch.nn.functional as F
from model import GPT2KWModel
import os
from tokenizations import tokenization_bert
import argparse
import json
from tqdm import trange


class GPT2Generator:
    def __init__(self, args):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.batch_size = args.batch_size
        self.tokenizer = tokenization_bert.BertTokenizer(vocab_file=args.tokenizer_path)
        self.model = pytorch_transformers.modeling_gpt2.GPT2LMHeadModel.from_pretrained(args.model_path)
        self.model.to(self.device)
        self.model.eval()
        self.keywords_max_length = 64

    def is_word(self, word):
        for item in list(word):
            if item not in 'qwertyuiopasdfghjklzxcvbnm':
                return False
        return True

    def top_k_top_p_filtering(self, logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
        """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
            Args:
                logits: logits distribution shape (vocabulary size)
                top_k > 0: keep only top k tokens with highest probability (top-k filtering).
                top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                    Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
            From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317

            temperature: Float value controlling randomness in boltzmann distribution. Lower temperature results in
            less random completions. As the temperature approaches zero, the model will become deterministic and
            repetitive. Higher temperature results in more random completions.

            top_k: Integer value controlling diversity. 1 means only 1 word is considered for each step (token),
            resulting in deterministic completions, while 40 means 40 words are considered at each step. 0 is a
            special setting meaning no restrictions. 40 generally is a good value.
        """
        assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
        top_k = min(top_k, logits.size(-1))  # Safety check
        if top_k > 0:
            # Remove all tokens with a probability less than the last token of the top-k
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = filter_value

        if top_p > 0.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[indices_to_remove] = filter_value
        return logits

    def generate_sequence(self, context_ids, keyword_ids, length, num_samples=1, temperature=1, top_k=0, top_p=0.0, is_xlnet=False, window_size=512):
        context_ids = torch.tensor(context_ids, dtype=torch.long, device=self.device)
        context_ids = context_ids.unsqueeze(0).repeat(num_samples, 1)
        keyword_ids = torch.tensor(keyword_ids, dtype=torch.long, device=self.device)
        keyword_ids = keyword_ids.unsqueeze(0).repeat(num_samples, 1)
        generated = context_ids
        with torch.no_grad():
            for _ in range(length):
                inputs = {'input_ids': generated[:, -(window_size-1):]}
                outputs = self.model(**inputs)
                next_token_logits = outputs[0][0, -1, :] / temperature
                filtered_logits = self.top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
                next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
                generated = torch.cat((generated, next_token.unsqueeze(0)), dim=1)
                if next_token == 101:   # CLS 是结束符
                    break
        return generated

    def tokenization(self, content, keywords):
        keywords = keywords.split("，")
        # 处理关键词
        keyword_ids = []
        for keyword in keywords:
            keyword_tokens = self.tokenizer.tokenize(keyword)
            single_keyword_ids = self.tokenizer.convert_tokens_to_ids(keyword_tokens + ["[SEP]"])
            if len(keyword_ids) + len(single_keyword_ids) < self.keywords_max_length:
                keyword_ids.extend(single_keyword_ids)
        keyword_ids.extend(self.tokenizer.convert_tokens_to_ids(["[PAD]"] * (self.keywords_max_length - len(keyword_ids))))
        # 处理正文
        # passage = ' [MASK] ' + content + ' [SEP] ' # [MASK] 表示文章开头
        # passage = content + ' [SEP] '   # [MASK] 表示文章开头
        passage = content   # [MASK] 表示文章开头
        passage_tokens = self.tokenizer.tokenize(passage)
        passage_ids = self.tokenizer.convert_tokens_to_ids(passage_tokens)
        return passage_ids, keyword_ids

    def generate(self, raw_text, keywords, length, window_size, temperature, top_k, top_p, num_samples):
        print("开头: %s" % raw_text)
        print("关键词: %s" % keywords)
        context_ids, keyword_ids = self.tokenization(raw_text, keywords)
        generated = 0
        for _ in range(num_samples // self.batch_size):
            out = self.generate_sequence(
                context_ids=context_ids, keyword_ids=keyword_ids, length=length, num_samples=1,
                temperature=temperature, top_k=top_k, top_p=top_p, window_size=window_size
            )
            out = out.tolist()
            for i in range(self.batch_size):
                generated += 1
                text = self.tokenizer.convert_ids_to_tokens(out[0])
                for j, item in enumerate(text[:-1]):  # 确保英文前后有空格
                    if self.is_word(item) and self.is_word(text[j + 1]):
                        text[j] = item + ' '
                for j, item in enumerate(text):
                    if item == '[MASK]':
                        text[j] = ''
                    if item == '[CLS]' or item == '[SEP]':
                        text[j] = '\n'
                    if item == '[PAD]':
                        text[j] = ''
                print("=" * 40 + " SAMPLE " + str(generated) + " " + "=" * 40)
                text = ''.join(text).replace('##', '').strip()
                print(text)
        print("=" * 80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='0,1,2,3', type=str, required=False, help='生成设备')
    parser.add_argument('--length', default=512, type=int, required=False, help='生成长度')
    parser.add_argument('--batch_size', default=1, type=int, required=False, help='生成的batch size')
    parser.add_argument('--num_samples', default=1, type=int, required=False, help='生成几个样本')
    parser.add_argument('--temperature', default=1, type=float, required=False, help='生成温度')
    parser.add_argument('--topk', default=8, type=int, required=False, help='最高几选一')
    parser.add_argument('--topp', default=0, type=float, required=False, help='最高积累概率')
    parser.add_argument('--model_config', default='config/model_config.json', type=str, required=False,
                        help='模型参数')
    parser.add_argument('--tokenizer_path', default='cache/vocab.txt', type=str, required=False, help='词表路径')
    parser.add_argument('--model_path', default='model_toutiao_all_pretrain/model_epoch3', type=str, required=False, help='模型路径')
    parser.add_argument('--raw_text', default='', type=str, required=False, help='生成文章的开头')
    parser.add_argument('--keywords', default='中国男篮，王治郅，姚明', type=str, required=False, help='关键词，以中文逗号隔开')

    args = parser.parse_args()
    if torch.cuda.is_available() is False:
        args.model_path = "/Volumes/移动硬盘/model/model_toutiao_all_pretrain/model_epoch2"
    generator = GPT2Generator(args)

    raw_text_list = []
    keywords_list = []
    with open("gao_test.txt", "r") as f:
        for line in f:
            line = line.strip()
            if len(line) == 0: continue
            line = json.loads(line)
            raw_text_list.append(line[0])
            keywords_list.append(line[1])

    for raw_text, keywords in zip(raw_text_list, keywords_list):
        generator.generate(raw_text=raw_text,
                           keywords=keywords,
                           length=512,
                           window_size=512,
                           temperature=1,
                           top_k=8,
                           top_p=0,
                           num_samples=1)




