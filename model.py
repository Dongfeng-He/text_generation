import math
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from pytorch_transformers.modeling_utils import Conv1D
from pytorch_transformers.modeling_gpt2 import GPT2PreTrainedModel, gelu, GPT2Model


class MLPLayer(nn.Module):
    def __init__(self, output_size, input_size):
        super(MLPLayer, self).__init__()
        self.c_fc = Conv1D(output_size, input_size)
        self.act = gelu
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        h = self.act(self.c_fc(x))
        return self.dropout(h)


class Attention(nn.Module):
    def __init__(self, hidden_size=768, head_num=12, scale=False):
        super(Attention, self).__init__()

        assert hidden_size % head_num == 0  # 隐层要能整除头数，因为要直接对隐层均匀切分来进行分头
        self.head_num = head_num
        self.hidden_size = hidden_size
        self.scale = scale

        self.c_attn = Conv1D(hidden_size * 3, hidden_size)
        self.c_proj = Conv1D(hidden_size, hidden_size)
        self.attn_dropout = nn.Dropout(0.1)
        self.mlp = MLPLayer(hidden_size, hidden_size * 2)

    def _attn(self, query, key, value, attendee_mask=None):
        # query: attender
        # key、value: attendee
        attention_matrix = torch.matmul(query, key)     # (batch, head, attender_seq_length, attendee_seq_length)
        batch, head, attender_seq_length, attendee_seq_length = attention_matrix.shape
        if self.scale:
            attention_matrix = attention_matrix / math.sqrt(value.size(-1))

        # attendee_mask (batch, attendee_seq_length)
        if attendee_mask is not None:
            attendee_mask = attendee_mask.unsqueeze(1).unsqueeze(2).repeat(1, head, attender_seq_length, 1)
            attention_matrix = attention_matrix * attendee_mask

        attention_matrix = nn.Softmax(dim=-1)(attention_matrix)
        attention_matrix = self.attn_dropout(attention_matrix)

        attention = torch.matmul(attention_matrix, value)
        return attention

    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_x_shape)  # in Tensorflow implem: fct merge_states

    def split_heads(self, x, is_key=False):
        new_x_shape = x.size()[:-1] + (self.head_num, x.size(-1) // self.head_num)
        x = x.view(*new_x_shape)  # in Tensorflow implem: fct split_states
        if is_key:
            return x.permute(0, 2, 3, 1)  # (batch, head, head_features, seq_length)
        else:
            return x.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

    def forward(self, attender_seq, attendee_seq, attendee_mask=None):
        attender_seq_ext = self.c_attn(attender_seq)
        attendee_seq_ext = self.c_attn(attendee_seq)
        query, _, _ = attender_seq_ext.split(self.hidden_size, dim=2)
        _, key, value = attendee_seq_ext.split(self.hidden_size, dim=2)
        query = self.split_heads(query)                 # (batch, head, attender_seq_length, head_features), (2, 12, 512, 64)
        key = self.split_heads(key, is_key=True)        # (batch, head, head_features, attendee_seq_length)
        value = self.split_heads(value)                 # (batch, head, attendee_seq_length, head_features)

        attention = self._attn(query, key, value, attendee_mask)

        attention = self.merge_heads(attention)
        attention = self.c_proj(attention)
        cat_output = torch.cat([attender_seq, attention], 2)
        output = self.mlp(cat_output)
        return output


class GPT2KWModel(GPT2PreTrainedModel):
    def __init__(self, config):
        super(GPT2KWModel, self).__init__(config)
        self.transformer = GPT2Model(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.attention_layer = Attention(hidden_size=config.n_embd, head_num=config.n_head, scale=True)
        self.apply(self.init_weights)
        self.tie_weights()

    def tie_weights(self):
        self._tie_or_clone_weights(self.lm_head, self.transformer.wte)

    def forward(self, input_ids, keyword_ids, keyword_mask=None, position_ids=None, token_type_ids=None, labels=None, past=None, head_mask=None):
        transformer_outputs = self.transformer(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                                               past=past, head_mask=head_mask)
        keyword_outputs = self.transformer(keyword_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                                           past=past, head_mask=head_mask)
        lm_hidden_states = transformer_outputs[0]
        keyword_hidden_states = keyword_outputs[0]
        att_out = self.attention_layer(lm_hidden_states, keyword_hidden_states, keyword_mask)

        lm_logits = self.lm_head(att_out)

        outputs = (lm_logits,)
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                            shift_labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), lm_logits
