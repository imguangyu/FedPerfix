import copy
import torch
import torch.nn as nn
import numpy as np
import time
from flcore.clients.clientbase import Client
from copy import deepcopy


class clientAPFL(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)

        self.alpha = args.alpha
        self.model_per = copy.deepcopy(self.model)
        self.optimizer_per = torch.optim.SGD(self.model_per.parameters(), lr=self.learning_rate)

    def load_infos(self, info):
        self.model.load_state_dict(info['weights'])
        m_per = info['local']
        self.model_per.load_state_dict(m_per)
        # self.model_local.load_state_dict(m_local)
        self.sampled = info['sampled']
        self.test_acc, self.test_num, self.auc = info['test']
        self.train_samples = info['train_samples']
        # self.loss, self.train_num = info['train']
        self.train_time_cost = info['train_time_cost']
        self.alpha = info['alpha']

    def get_infos(self):
        res = super().get_infos()

        res.update({"local" : deepcopy(self.model_per.cpu().state_dict())})
        res.update({"alpha" : self.alpha})

        return res

    def train(self):
        trainloader = self.load_train_data()
        
        start_time = time.time()

        # self.model.to(self.device)
        self.model.train()

        max_local_steps = self.local_steps
        if self.train_slow:
            max_local_steps = np.random.randint(1, max_local_steps // 2)

        for step in range(max_local_steps):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device).float()
                else:
                    x = x.to(self.device).float()
                y = y.type(torch.LongTensor).to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                self.optimizer.zero_grad()
                output = self.model(x)
                loss = self.loss(output, y)
                loss.backward()
                self.optimizer.step()

                self.optimizer_per.zero_grad()
                output_per = self.model_per(x)
                loss_per = self.loss(output_per, y)
                loss_per.backward()
                self.optimizer_per.step()

                self.alpha_update()

        for (lk,lp), (k,p) in zip(self.model_per.named_parameters(), self.model.named_parameters()):
            assert lk == k
            # if lk in self.model_per.local:
            lp.data = (1 - self.alpha) * p + self.alpha * lp

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    # https://github.com/MLOPTPSU/FedTorch/blob/b58da7408d783fd426872b63fbe0c0352c7fa8e4/fedtorch/comms/utils/flow_utils.py#L240
    def alpha_update(self):
        grad_alpha = 0
        for l_params, p_params in zip(self.model.parameters(), self.model_per.parameters()):
            dif = p_params.data - l_params.data
            grad = self.alpha * p_params.grad.data + (1-self.alpha) * l_params.grad.data
            grad_alpha += dif.view(-1).T.dot(grad.view(-1))
        
        grad_alpha += 0.02 * self.alpha
        self.alpha = self.alpha - self.learning_rate * grad_alpha
        self.alpha = np.clip(self.alpha.item(), 0.0, 1.0)