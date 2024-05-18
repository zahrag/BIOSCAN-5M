import torch.nn.functional as F
import timm
import torch.nn as nn
import torch.optim as optim
from model.mlp import MLPEncoder
from model.image_encoder import LoRA_ViT_timm
import hydra
from model.dna_encoder import load_pre_trained_bioscan_bert, LoRA_barcode_bert, Freeze_DNA_Encoder
from model.language_encoder import load_pre_trained_bert, LoRA_bert
import numpy as np
from typing import Optional
import torch
import torch.multiprocessing as mp
import datetime


class SimpleCLIP(nn.Module):
    def __init__(self, image_encoder, dna_encoder, language_encoder):
        super(SimpleCLIP, self).__init__()
        self.image_encoder = image_encoder
        self.dna_encoder = dna_encoder
        self.language_encoder = language_encoder

    def forward(self, image_input, dna_input, language_input):
        image_output = None
        dna_output = None
        language_output = None

        if self.image_encoder is not None:
            image_output = F.normalize(self.image_encoder(image_input), p=2, dim=-1)
        if self.dna_encoder is not None:
            dna_output = F.normalize(self.dna_encoder(dna_input), p=2, dim=-1)
        if self.language_encoder is not None:
            language_output = F.normalize(self.language_encoder(language_input), p=2, dim=-1)

        return image_output, dna_output, language_output

# Modified from OpenCLIP code
class SimpleCLIP_With_Trainable_Temp(nn.Module):
    def __init__(self, image_encoder, dna_encoder, language_encoder, init_logit_scale: float = np.log(1 / 0.07),
            init_logit_bias: Optional[float] = None):
        super(SimpleCLIP_With_Trainable_Temp, self).__init__()
        self.image_encoder = image_encoder
        self.dna_encoder = dna_encoder
        self.language_encoder = language_encoder
        self.logit_scale = nn.Parameter(torch.ones([]) * init_logit_scale)
        if init_logit_bias is not None:
            self.logit_bias = nn.Parameter(torch.ones([]) * init_logit_bias)
        else:
            self.logit_bias = None


    def forward(self, image_input, dna_input, language_input):
        image_output = None
        dna_output = None
        language_output = None

        if self.image_encoder is not None:
            image_output = F.normalize(self.image_encoder(image_input), p=2, dim=-1)
        if self.dna_encoder is not None:
            dna_output = F.normalize(self.dna_encoder(dna_input), p=2, dim=-1)
        if self.language_encoder is not None:
            language_output = F.normalize(self.language_encoder(language_input), p=2, dim=-1)

        return image_output, dna_output, language_output, self.logit_scale.exp(), self.logit_bias

class SimpleCLIPWithClassificationHead(nn.Module):
    def __init__(self, clip_model, hidden_dime = 768, number_of_classes=1024):
        super(SimpleCLIPWithClassificationHead, self).__init__()
        self.image_encoder = clip_model.image_encoder
        self.dna_encoder = clip_model.dna_encoder
        self.language_encoder = clip_model.language_encoder

        if self.image_encoder is not None:
            for param in self.image_encoder.parameters():
                param.requires_grad = True
        if self.dna_encoder is not None:
            for param in self.dna_encoder.parameters():
                param.requires_grad = False
        if self.language_encoder is not None:
            for param in self.language_encoder.parameters():
                param.requires_grad = False

        self.classification_head = nn.Sequential(
            nn.Linear(768, hidden_dime),
            nn.ReLU(),
        nn.Linear(hidden_dime, hidden_dime),
            nn.ReLU(),
        nn.Linear(hidden_dime, number_of_classes),
        nn.Softmax(dim=1)
        )

    def forward(self, image_input, dna_input, language_input):
        image_output = None
        dna_output = None
        language_output = None

        if image_input is not None:
            image_output = F.normalize(self.image_encoder(image_input), p=2, dim=-1)
        if dna_input is not None:
            dna_output = F.normalize(self.dna_encoder(dna_input), p=2, dim=-1)
        if language_input is not None:
            language_output = F.normalize(self.language_encoder(language_input), p=2, dim=-1)

        output = self.classification_head(image_output)

        return image_output, dna_output, language_output, output




def load_clip_model(args, device=None):
    image_encoder = None
    dna_encoder = None
    language_encoder = None
    # For image part
    if args.model_config.image.input_type == "image":
        if args.model_config.image.model == "lora_vit":
            pre_trained_timm_vit = timm.create_model('vit_base_patch16_224', pretrained=True)
            image_encoder = LoRA_ViT_timm(vit_model=pre_trained_timm_vit, r=4, num_classes=args.model_config.output_dim)
        elif args.model_config.image.model == "vit_plus_mlp_with_warm_up":
            print("Coding for vit_plus_mlp_with_warm_up are not finished yet")
            exit()
    else:
        image_encoder = MLPEncoder(input_dim=args.model_config.image.input_dim,
                                   hidden_dim=args.model_config.image.hidden_dim,
                                   output_dim=args.model_config.output_dim)

    # For DNA part
    if hasattr(args.model_config, 'dna'):
        if hasattr(args.model_config.dna, 'freeze') and args.model_config.dna.freeze:
            dna_encoder = Freeze_DNA_Encoder()
        elif args.model_config.dna.input_type == "sequence":
            if args.model_config.dna.model == "lora_barcode_bert":
                pre_trained_barcode_bert = load_pre_trained_bioscan_bert(
                    bioscan_bert_checkpoint=args.bioscan_bert_checkpoint)
                dna_encoder = LoRA_barcode_bert(model=pre_trained_barcode_bert, r=4,
                                                num_classes=args.model_config.output_dim)
        else:
            dna_encoder = MLPEncoder(input_dim=args.model_config.dna.input_dim,
                                     hidden_dim=args.model_config.dna.hidden_dim,
                                     output_dim=args.model_config.output_dim)

    # For language
    if hasattr(args.model_config, 'language'):
        if args.model_config.language.input_type == "sequence":
            if args.model_config.language.model == "lora_bert":
                _, pre_trained_bert = load_pre_trained_bert()
                language_encoder = LoRA_bert(model=pre_trained_bert,  r=4, num_classes=args.model_config.output_dim)
            else:
                raise TypeError(f"{args.model_config.language.model} are not support yet.")
        else:
            raise TypeError(f"Using {args.model_config.language.input_type} as language input is not support yet.")

    if hasattr(args.model_config, 'open_clip_ver') and args.model_config.open_clip_ver:
        model = SimpleCLIP_With_Trainable_Temp(image_encoder=image_encoder, dna_encoder=dna_encoder, language_encoder=language_encoder)
    else:
        model = SimpleCLIP(image_encoder=image_encoder, dna_encoder=dna_encoder, language_encoder=language_encoder)

    if device is not None:
        model.to(device)
    return model
