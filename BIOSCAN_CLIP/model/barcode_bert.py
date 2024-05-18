import math
from itertools import product

import torch
import torch.nn as nn
from torch import Tensor
from torchtext.vocab import build_vocab_from_iterator
from transformers import BertConfig, BertForMaskedLM
from util.util import PadSequence, KmerTokenizer, load_bert_model

device = "cuda" if torch.cuda.is_available() else "cpu"

def load_pre_trained_bioscan_bert(bioscan_bert_checkpoint, k=5):
    kmer_iter = (["".join(kmer)] for kmer in product("ACGT", repeat=k))
    vocab = build_vocab_from_iterator(kmer_iter, specials=["<MASK>", "<CLS>", "<UNK>"])
    vocab.set_default_index(vocab["<UNK>"])
    vocab_size = len(vocab)
    configuration = BertConfig(vocab_size=vocab_size, output_hidden_states=True)
    bert_model = BertForMaskedLM(configuration)
    load_bert_model(bert_model, bioscan_bert_checkpoint)
    return bert_model.to(device)

def get_sequence_pipeline(k=5):
    kmer_iter = (["".join(kmer)] for kmer in product("ACGT", repeat=k))
    vocab = build_vocab_from_iterator(kmer_iter, specials=["<MASK>", "<CLS>", "<UNK>"])
    vocab.set_default_index(vocab["<UNK>"])

    max_len = 660
    pad = PadSequence(max_len)
    tokenizer = KmerTokenizer(k, stride=k)
    sequence_pipeline = lambda x: [0, *vocab(tokenizer(pad(x)))]

    return sequence_pipeline



# MODIFIED FROM https://github.com/JamesQFreeman/LoRA-barcode_bert/blob/main/lora.py

class _LoRALayer(nn.Module):
    def __init__(self, w: nn.Module, w_a: nn.Module, w_b: nn.Module):
        super().__init__()
        self.w = w
        self.w_a = w_a
        self.w_b = w_b

    def forward(self, x):
        x = self.w(x) + self.w_b(self.w_a(x))
        return x


class LoRA_barcode_bert(nn.Module):
    def __init__(self, model, r: int, num_classes: int = 0, lora_layer=None):
        super(LoRA_barcode_bert, self).__init__()

        assert r > 0
        if lora_layer:
            self.lora_layer = lora_layer
        else:
            self.lora_layer = list(range(len(model.bert.encoder.layer)))

        # create for storage, then we can init them or load weights
        self.w_As = []  # These are linear layers
        self.w_Bs = []

        # lets freeze first
        for param in model.parameters():
            param.requires_grad = False

        for layer_idx, layer in enumerate(model.bert.encoder.layer):
            if layer_idx not in self.lora_layer:
                continue
            w_q_linear = layer.attention.self.query
            w_v_linear = layer.attention.self.value
            dim = layer.attention.self.query.in_features

            w_a_linear_q = nn.Linear(dim, r, bias=False)
            w_b_linear_q = nn.Linear(r, dim, bias=False)
            w_a_linear_v = nn.Linear(dim, r, bias=False)
            w_b_linear_v = nn.Linear(r, dim, bias=False)

            self.w_As.append(w_a_linear_q)
            self.w_Bs.append(w_b_linear_q)
            self.w_As.append(w_a_linear_v)
            self.w_Bs.append(w_b_linear_v)

            layer.attention.self.query = _LoRALayer(w_q_linear, w_a_linear_q, w_b_linear_q)
            layer.attention.self.value = _LoRALayer(w_v_linear, w_a_linear_v, w_b_linear_v)

        self.reset_parameters()
        self.lora_barcode_bert = model

        if num_classes > 0:
            self.lora_barcode_bert.cls.predictions.decoder = nn.Linear(self.lora_barcode_bert.cls.predictions.decoder.in_features, num_classes)


    def reset_parameters(self) -> None:
        for w_A in self.w_As:
            nn.init.kaiming_uniform_(w_A.weight, a=math.sqrt(5))
        for w_B in self.w_Bs:
            nn.init.zeros_(w_B.weight)

    def forward(self, x: Tensor) -> Tensor:

        return self.lora_barcode_bert(x).logits.softmax(dim=-1).mean(dim=1)
