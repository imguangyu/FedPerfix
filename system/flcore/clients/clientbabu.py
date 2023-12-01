import copy
import torch
import torch.nn as nn
import numpy as np
import time
from flcore.clients.clientbase import Client


class clientBABU(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)

        self.fine_tuning_steps = args.fine_tuning_steps

        # for param in self.model.head.parameters():
        #     param.requires_grad = False


    def train(self):
        trainloader = self.load_train_data()

        self.tl = trainloader
        
        start_time = time.time()

        # self.model.to(self.device)
        self.model.train()

        max_local_steps = self.local_steps
        if self.train_slow:
            max_local_steps = np.random.randint(1, max_local_steps // 2)
        
        if hasattr(self.model.basic_model,"head"):
            for param in self.model.basic_model.head.parameters():
                param.requires_grad = False
        else:
            for param in self.model.basic_model.fc.parameters():
                param.requires_grad = False

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

                if self.debug and i>1:
                    break

        # self.model.cpu()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    # def set_parameters(self, base):
    #     for new_param, old_param in zip(base.parameters(), self.model.base.parameters()):
    #         old_param.data = new_param.data.clone()
    
    def train_one_step(self):
        # testloader = self.load_test_data(self.batch_size)
        # iter_testloader = iter(testloader)

        trainloader = self.tl if hasattr(self,'tl') and self.tl is not None else self.load_train_data(self.batch_size)
        iter_loader = iter(trainloader)
        self.model.to(self.device)
        self.model.train()

        for i in range(self.fine_tuning_steps):
            (x, y) = next(iter_loader)
            if type(x) == type([]):
                x[0] = x[0].to(self.device).float()
            else:
                x = x.to(self.device).float()
            y = y.type(torch.LongTensor).to(self.device)
            self.optimizer.zero_grad()
            output = self.model(x)
            loss = self.loss(output, y)
            loss.backward()
            self.optimizer.step()

    # comment for testing on new clients
    def test_metrics(self, temp_model=None):
        temp_model = copy.deepcopy(self.model)
        if hasattr(self.model.basic_model,"head"):
            for param in self.model.basic_model.head.parameters():
                param.requires_grad = True
        else:
            for param in self.model.basic_model.fc.parameters():
                param.requires_grad = True

        self.train_one_step()
        return_val = super().test_metrics()
        self.clone_model(temp_model, self.model)
        self.tl = None
        return return_val

