import torch
import torch.nn as nn
import copy
import math
from functools import reduce
from operator import mul

from timm.models.vision_transformer import VisionTransformer, PatchEmbed, Block
from copy import deepcopy

def build_prefix(basic_model, args):

    model = Prefix(basic_model, num_classes=args.num_classes, mid_dim=args.mid_dim, scale=args.scale, local_parts=args.local_parts, vanilla=args.vanilla, depth=args.prefix_depth)
    return model

# Side Adapter for FedPerfix
class Adapter(nn.Module):
    def __init__(self, dim, mid_dim=256, act_fn='tanh'):
        super().__init__()

        self.dim = dim
        self.mid_dim = mid_dim

        self.down = nn.Linear(dim, mid_dim)
        if act_fn == 'tanh':
            self.act_fn = nn.Tanh()
        else:
            raise NotImplementedError
        self.up = nn.Linear(mid_dim, dim * 2)

    def forward(self,x):
        # Output the prefixes to be added to the qkv
        kv = self.up(self.act_fn(self.down(x)))
        q = torch.zeros(size=(kv.shape[0], kv.shape[1], self.dim),device=kv.device)

        return torch.concat((q,kv), dim=-1)


class VanillaPrefix(nn.Module):
    def __init__(self, dim, zero_init=False, mid_dim=256, act_fn='tanh'):
        super().__init__()
        self.dim = dim
        self.kv = nn.Linear(dim, dim * 2)

        if zero_init:
            torch.nn.init.zeros_(self.kv.weight)
            torch.nn.init.zeros_(self.kv.bias)

    def forward(self,x):
        kv = self.kv(x)
        q = torch.zeros(size=(kv.shape[0], kv.shape[1], self.dim),device=kv.device)

        return torch.concat((q,kv), dim=-1)


class AdapterAttention(nn.Module):
    def __init__(self, base_attention, dim, mid_dim=256, p_scale=0.8, vanilla='-'):
        super().__init__()
        self.qkv = deepcopy(base_attention.qkv)
        self.num_heads = deepcopy(base_attention.num_heads)
        self.scale = deepcopy(base_attention.scale)
        self.attn_drop = deepcopy(base_attention.attn_drop)
        self.proj = deepcopy(base_attention.proj)
        self.proj_drop = deepcopy(base_attention.proj_drop)

        if vanilla == 'zero':
            self.adapter = VanillaPrefix(dim, zero_init=True ,mid_dim=mid_dim)
        elif vanilla == 'random':
            self.adapter = VanillaPrefix(dim, zero_init=False ,mid_dim=mid_dim)
        else:
            self.adapter = Adapter(dim, mid_dim=mid_dim)
        self.p_scale = p_scale

    def forward(self, x):
        B, N, C = x.shape

        prefixs = self.adapter(x)
        # print(prefixs)
        # print(prefixs.size(), self.qkv(x).size())

        qkv = (self.qkv(x) + self.p_scale * prefixs).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Prefix(nn.Module):
    def __init__(self, basic_model, num_classes=1000, mid_dim=256, scale=0.8,
        local_parts=[], no_pt = False, add_attention=True, vanilla='-', depth=12):

        super(Prefix, self).__init__()
        self.no_pt = no_pt
        self.mid_dim = mid_dim
        self.scale = scale
        self.vanilla = vanilla
        self.depth = depth

        self.basic_model = deepcopy(basic_model)
        self.embed_dim = embed_dim = self.basic_model.embed_dim

        patch_size = basic_model.patch_embed.patch_size
        depth = len(basic_model.blocks)

        # Change num_class
        # self.basic_model.head = nn.Linear(self.embed_dim, num_classes)

        if add_attention:
            self._add_attention()

        if not self.no_pt:
            self.freeze()
        
        self.local = self.set_local(local_parts)

    def _add_attention(self):
        n = len(self.basic_model.blocks)
        for i, block in enumerate(self.basic_model.blocks):
            if (n - i) <= self.depth:
                base_attention = deepcopy(block.attn)
                block.attn = AdapterAttention(base_attention, self.embed_dim, mid_dim=self.mid_dim, p_scale=self.scale, vanilla=self.vanilla)


    
    def set_local(self, local_part):
        locals = []
        for k, p in self.named_parameters():
            if any(s in k for s in local_part):
                locals.append(k)
        # self.local = locals

        return locals

    def freeze(self):
        pass
    
    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True

    def forward(self,x):
        x = self.basic_model.forward_features(x)
        x = self.basic_model.forward_head(x)
        return x
    
    def state_dict(self):

        if self.local is not None:
            state_dict = {}

            for k, p in self.named_parameters():
                if not k in self.local:
                    state_dict.update({k:p.data})

            return state_dict

        if self.tune_all:
            return super().state_dict()

        state_dict = {}

        for k, p in self.named_parameters():
            if 'global' in k:
                state_dict.update({k:p.data})
            elif self.no_pt and 'basic_model' in k:
                state_dict.update({k:p.data})
            elif self.tune_cls and 'cls_token' in k:
                state_dict.update({k:p.data})
            
        
        return state_dict
    
    def get_local_state_dict(self):

        if self.local is not None:
            state_dict = {}

            for k, p in self.named_parameters():
                if k in self.local:
                    state_dict.update({k:p.data})

            return state_dict


        if self.tune_all:
            return {}
        state_dict = {}

        for k, p in self.named_parameters():
            if 'local' in k or 'head' in k:
                state_dict.update({k:p.data.cpu()})

        return state_dict

    def load_state_dict(self, state_dict, strict=False):
        # self.head.load_state_dict(prompt_state_dict['head'], strict=strict)
        super().load_state_dict(state_dict, strict=False)