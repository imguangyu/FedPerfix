# Some implementation is from https://github.com/sagizty/VPT and https://github.com/KMnP/vpt

import torch
import torch.nn as nn
import copy
import math
from functools import reduce
from operator import mul

from timm.models.vision_transformer import VisionTransformer, PatchEmbed, Block
from copy import deepcopy

def build_prompt(basic_model, args):

    model = Prompt(basic_model, 
        prompt_num = args.local_prompt_num,  num_classes=args.num_classes,
        projection= args.prompt_projection, prompt_drop_rate=args.prompt_drop, no_pt=args.no_pt, local_parts=args.local_parts)
    return model


class Prompt(nn.Module):
    def __init__(self, basic_model, 
        prompt_num=1, num_classes=1000, prompt_drop_rate=0.1,
        local_parts=[]):

        super(Prompt, self).__init__()

        self.basic_model = deepcopy(basic_model)
        self.embed_dim = embed_dim = self.basic_model.embed_dim

        patch_size = basic_model.patch_embed.patch_size
        depth = len(basic_model.blocks)

        self.local_prompt_num = prompt_num

        # Change num_class
        # self.basic_model.head = nn.Linear(self.embed_dim, num_classes)


        prompt_dim = embed_dim
        self.local_prompt_proj = nn.Identity()
        self.global_prompt_proj = nn.Identity()

        val = math.sqrt(6. / float(3 * reduce(mul, patch_size, 1) + prompt_dim))  # noqa


        
        self.local_prompt_embeddings = nn.Parameter(torch.zeros(depth, prompt_num, prompt_dim))
        # xavier_uniform initialization
        nn.init.uniform_(self.local_prompt_embeddings.data, -val, val)

        self.local_prompt_dropout = nn.Dropout(prompt_drop_rate)
        self.global_prompt_dropout = nn.Dropout(prompt_drop_rate)

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

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

        self.local_prompt_embeddings.requires_grad = True
        self.global_prompt_embeddings.requires_grad = True

        if not self.fixed_head:
            for param in self.basic_model.head.parameters():
                param.requires_grad = True
        if self.tune_cls:
            self.basic_model.cls_token.requires_grad = True
        for param in self.local_prompt_proj.parameters():
            param.requires_grad = True
        for param in self.global_prompt_proj.parameters():
            param.requires_grad = True
    
    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True

    def add_prompt(self, i, x):
        removes = 0

        # local 
        local_prompt_embeddings = self.local_prompt_embeddings[i].unsqueeze(0)
        local_prompt_embeddings = self.local_prompt_dropout(self.local_prompt_proj(local_prompt_embeddings))
        # global
        global_prompt_embeddings = self.global_prompt_embeddings[i].unsqueeze(0)
        global_prompt_embeddings = self.global_prompt_dropout(self.global_prompt_proj(global_prompt_embeddings))

        x = torch.cat((x, local_prompt_embeddings.expand(x.shape[0], -1, -1), global_prompt_embeddings.expand(x.shape[0], -1, -1)), dim=1)

        removes += self.local_prompt_num + self.global_prompt_num

        return x, removes

    def forward_features(self, x):
        
        x = self.basic_model.patch_embed(x)
        x = self.basic_model._pos_embed(x)

        # if self.design == "deep-deep":
        for i in range(len(self.basic_model.blocks)):
            x, removes = self.add_prompt(i,x)
            # lastly remove, a genius trick
            num_tokens = x.shape[1]
            x = self.basic_model.blocks[i](x)[:, :num_tokens - removes]

        x = self.basic_model.norm(x)
        return x

    def forward(self,x):
        self.basic_model.eval()
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