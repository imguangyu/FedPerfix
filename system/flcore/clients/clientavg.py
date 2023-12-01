import torch
import torch.nn as nn
import numpy as np
import time
from flcore.clients.clientbase import Client
from utils.privacy import *


class clientAVG(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)

    def train(self):
        trainloader = self.load_train_data()
        # self.model.to(self.device)
        self.model.train()

        # differential privacy
        if self.privacy:
            self.model, self.optimizer, trainloader, privacy_engine = \
                initialize_dp(self.model, self.optimizer, trainloader, self.dp_sigma)
        
        start_time = time.time()

        max_local_steps = self.local_steps
        if self.train_slow:
            max_local_steps = np.random.randint(1, max_local_steps // 2)

        # if hasattr(self.model, 'two_stage'):
        #     if self.model.two_stage:
        #         self.model.freeze()
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

                # if self.debug and i>1:
                #     break
                
        # if hasattr(self.model, 'two_stage'):
        #     if self.model.two_stage:
        #         self.model.unfreeze()
        #         for step in range(max_local_steps):
        #             for i, (x, y) in enumerate(trainloader):
        #                 if type(x) == type([]):
        #                     x[0] = x[0].to(self.device).float()
        #                 else:
        #                     x = x.to(self.device).float()
        #                 y = y.type(torch.LongTensor).to(self.device)
        #                 if self.train_slow:
        #                     time.sleep(0.1 * np.abs(np.random.rand()))
        #                 self.optimizer.zero_grad()
        #                 output = self.model(x)
        #                 loss = self.loss(output, y)
        #                 loss.backward()
        #                 self.optimizer.step()


        # self.model.cpu()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

        if self.privacy:
            eps, DELTA = get_dp_params(privacy_engine)
            print(f"Client {self.id}", f"epsilon = {eps:.2f}, sigma = {DELTA}")