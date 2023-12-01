import torch
import torch.nn as nn
import copy
import math
from functools import reduce
from operator import mul

from copy import deepcopy

def build_paitial(basic_model, args):

    model = Partial(basic_model, local_parts=args.local_parts)
    return model


class Partial(nn.Module):
    '''
        A general partial model personalization architecture.
        'basic_model' is the entire model to be personalized.
        'local_parts' is a list of strings, each string is a part of the model to be personalized.

        Please check the implementation of 'set_local' function to see how to set the 'local_parts' parameter.
        Please make sure the 'local_parts' parameter is set correctly.
        Ambiguous names will make all related parameters be personalized. e.g. 'bn' will personalize all parameters of all batch normalization layers.
    '''
    
    def __init__(self, basic_model, local_parts=[]):

        super(Partial, self).__init__()

        self.basic_model = deepcopy(basic_model)
        
        self.local = self.set_local(local_parts)
    
    def set_local(self, local_part):
        locals = []
        for k, p in self.named_parameters():
            if any(s in k for s in local_part):
                locals.append(k)
        # self.local = locals

        return locals

    def forward(self,x):

        x = self.basic_model(x)
        return x
    
    def state_dict(self):

        # Returns the global parameters only
        if self.local is not None:
            state_dict = {}

            for k, p in self.named_parameters():
                if not k in self.local:
                    state_dict.update({k:p.data})

            return state_dict

      
    def get_local_state_dict(self):
        
        # Returns the local parameters only
        if self.local is not None:
            state_dict = {}

            for k, p in self.named_parameters():
                if k in self.local:
                    state_dict.update({k:p.data})

            return state_dict

    def load_state_dict(self, state_dict, strict=False):
        # Load the global parameters only, force strict to False to avoid error
        super().load_state_dict(state_dict, strict=False)