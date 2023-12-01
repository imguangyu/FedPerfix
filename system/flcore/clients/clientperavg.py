import numpy as np
import torch
import time
import copy
import torch.nn as nn
from flcore.optimizers.fedoptimizer import PerAvgOptimizer
from flcore.clients.clientbase import Client
from copy import deepcopy

class clientPerAvg(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        # self.beta = args.beta
        self.beta = self.learning_rate

        self.loss = nn.CrossEntropyLoss()
        self.optimizer = PerAvgOptimizer(self.model.parameters(), lr=self.learning_rate)

    def load_infos(self, info):
        self.model.load_state_dict(info['weights'])
        self.sampled = info['sampled']
        self.test_acc, self.test_num, self.auc = info['test']
        self.train_samples = info['train_samples']
        # self.loss, self.train_num = info['train']
        self.train_time_cost = info['train_time_cost']
    #     self.beta = info['beta']
        self.optimizer.load_state_dict(info['op'])

    def get_infos(self):
        res = super().get_infos()

        # res.update({"beta" : self.beta})
        res.update({"op" : deepcopy(self.optimizer.state_dict())})
        

        return res

    def train(self):
        trainloader = self.load_train_data(self.batch_size*2)
        self.tl = trainloader
        start_time = time.time()

        # self.model.to(self.device)
        self.model.train()

        max_local_steps = self.local_steps
        if self.train_slow:
            max_local_steps = np.random.randint(1, max_local_steps // 2)

        for step in range(max_local_steps):  # local update
            for X, Y in trainloader:
                temp_model = copy.deepcopy(list(self.model.parameters()))

                # step 1
                if type(X) == type([]):
                    x = [None, None]
                    x[0] = X[0][:self.batch_size].to(self.device).float()
                    x[1] = X[1][:self.batch_size].float()
                else:
                    x = X[:self.batch_size].to(self.device).float()
                y = Y[:self.batch_size].type(torch.LongTensor).to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                self.optimizer.zero_grad()
                output = self.model(x)
                loss = self.loss(output, y)
                loss.backward()
                self.optimizer.step()

                # step 2
                if type(X) == type([]):
                    x = [None, None]
                    x[0] = X[0][self.batch_size:].to(self.device).float()
                    x[1] = X[1][self.batch_size:].float().to(self.device)
                else:
                    x = X[self.batch_size:].float().to(self.device)
                y = Y[self.batch_size:].type(torch.LongTensor).to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                self.optimizer.zero_grad()
                output = self.model(x)
                loss = self.loss(output, y)
                loss.backward()

                # restore the model parameters to the one before first update
                for old_param, new_param in zip(self.model.parameters(), temp_model):
                    old_param.data = new_param.data.clone()

                self.optimizer.step(beta=self.beta)

        # self.model.cpu()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    def train_one_step(self):
        # testloader = self.load_test_data(self.batch_size)
        # iter_testloader = iter(testloader)

        trainloader = self.tl if hasattr(self,'tl') and self.tl is not None else self.load_train_data(self.batch_size)
        iter_loader = iter(trainloader)
        # self.model.to(self.device)
        self.model.train()

        (x, y) = next(iter_loader)
        if type(x) == type([]):
            x[0] = x[0].to(self.device).float()
        else:
            x = x.to(self.device).float()
        y = y.type(torch.LongTensor).to(self.device)
        
        if x.shape[0] !=  self.batch_size:
            x = x[:self.batch_size]
            y = y[:self.batch_size]
    
        self.optimizer.zero_grad()
        output = self.model(x)
        loss = self.loss(output, y)
        loss.backward()
        self.optimizer.step()

    # comment for testing on new clients
    def test_metrics(self, temp_model=None):
        temp_model = copy.deepcopy(self.model)
        self.train_one_step()
        return_val = super().test_metrics()
        self.clone_model(temp_model, self.model)
        # self.model = deepcopy(temp_model)
        self.tl = None
        return return_val
