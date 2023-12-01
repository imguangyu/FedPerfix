import torch
import torch.nn as nn
import copy
import math
from functools import reduce
from operator import mul

from timm.models.vision_transformer import VisionTransformer, PatchEmbed, Block
from copy import deepcopy

def build_adapter(basic_model, args):

    model = PAadapter(basic_model, no_pt=args.no_pt,local_parts=args.local_parts, mid_dim=args.mid_dim)
    return model


class PAadapter(nn.Module):
    def __init__(self, basic_model, local_parts=[], mid_dim=512, no_pt=True):

        super(PAadapter, self).__init__()
        self.no_pt = no_pt


        self.basic_model = deepcopy(basic_model)
        self.embed_dim = embed_dim = self.basic_model.embed_dim

        patch_size = basic_model.patch_embed.patch_size
        depth = len(basic_model.blocks)

        self.adapter_downsample = nn.Linear(
                embed_dim,
                mid_dim
            )
        self.adapter_upsample = nn.Linear(
                mid_dim,
                embed_dim
            )
        self.adapter_act_fn = nn.functional.tanh

        nn.init.zeros_(self.adapter_downsample.weight)
        nn.init.zeros_(self.adapter_downsample.bias)

        nn.init.zeros_(self.adapter_upsample.weight)
        nn.init.zeros_(self.adapter_upsample.bias)

        if not self.no_pt:
            self.freeze()
        
        self.local = self.set_local(local_parts)
    
    def set_local(self, local_part):
        locals = []
        for k, p in self.named_parameters():
            if any(s in k for s in local_part):
                locals.append(k)
        # self.local = locals

        return locals

    # Test code, not used
    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

        for param in self.adapter_downsample.parameters():
            param.requires_grad = True
        for param in self.adapter_upsample.parameters():
            param.requires_grad = True
        for param in self.basic_model.head.parameters():
            param.requires_grad = True

    # Test code, not used
    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True

    def forward_features(self, x): 
        x = self.basic_model.patch_embed(x)
        x = self.basic_model._pos_embed(x)

        for i in range(len(self.basic_model.blocks)):
            
            # forward normal blocks
            block = self.basic_model.blocks[i]
            x = x + block.drop_path1(block.ls1(block.attn(block.norm1(x))))
            h = x
            x = block.drop_path2(block.ls2(block.mlp(block.norm2(x))))
            # adapter
            adpt = self.adapter_downsample(x)
            adpt = self.adapter_act_fn(adpt)
            adpt = self.adapter_upsample(adpt)
            x = adpt + x

            x = x + h


        x = self.basic_model.norm(x)
        return x

    def forward(self,x):
        x = self.forward_features(x)
        x = self.basic_model.forward_head(x)
        return x
    
    def state_dict(self):

        if self.local is not None:
            state_dict = {}

            for k, p in self.named_parameters():
                if not k in self.local:
                    state_dict.update({k:p.data})

            return state_dict


    
    def get_local_state_dict(self):

        if self.local is not None:
            state_dict = {}

            for k, p in self.named_parameters():
                if k in self.local:
                    state_dict.update({k:p.data})

            return state_dict

    def load_state_dict(self, state_dict, strict=False):
        # self.head.load_state_dict(prompt_state_dict['head'], strict=strict)
        super().load_state_dict(state_dict, strict=False)