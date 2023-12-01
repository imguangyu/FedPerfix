import torch
import os
import numpy as np
import h5py
import copy
import time
import random
from copy import deepcopy
import wandb

from utils.data_utils import read_client_data


class Server(object):
    def __init__(self, args, times):
        # Set up the main attributes
        self.device = args.device
        self.dataset = args.dataset
        self.global_rounds = args.global_rounds
        self.local_steps = args.local_steps
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.global_model = copy.deepcopy(args.model)
        self.num_clients = args.num_clients
        self.join_ratio = args.join_ratio
        self.random_join_ratio = args.random_join_ratio
        self.join_clients = int(self.num_clients * self.join_ratio)
        self.algorithm = args.algorithm
        self.time_select = args.time_select
        self.goal = args.goal
        self.time_threthold = args.time_threthold
        self.save_folder_name = args.save_folder_name
        self.top_cnt = 100

        self.clients = []
        self.selected_clients = []
        self.train_slow_clients = []
        self.send_slow_clients = []

        self.uploaded_weights = []
        self.uploaded_ids = []
        self.uploaded_models = []

        self.rs_test_acc = []
        self.rs_test_auc = []
        self.rs_train_loss = []

        self.times = times
        self.eval_gap = args.eval_gap
        self.client_drop_rate = args.client_drop_rate
        self.train_slow_rate = args.train_slow_rate
        self.send_slow_rate = args.send_slow_rate
        self.debug = args.debug

        self.Budget = []

    def set_clients(self, args, clientObj):
        for i, train_slow, send_slow in zip(range(self.num_clients), self.train_slow_clients, self.send_slow_clients):
            train_data = read_client_data(self.dataset, i, is_train=True)
            test_data = read_client_data(self.dataset, i, is_train=False)
            client = clientObj(args, 
                            id=i, 
                            train_samples=len(train_data), 
                            test_samples=len(test_data), 
                            train_slow=train_slow, 
                            send_slow=send_slow)
            self.clients.append(client)

    # def set_clients(self, args, clientObj):
    #     for i, train_slow, send_slow in zip(range(self.num_clients), self.train_slow_clients, self.send_slow_clients):
    #         # train_data = read_client_data(self.dataset, i, is_train=True)
    #         # test_data = read_client_data(self.dataset, i, is_train=False)
    #         client = clientObj(args, i, 1, 1, train_slow=None, send_slow=None)
    #         self.clients.append(client)


    # random select slow clients
    def select_slow_clients(self, slow_rate):
        slow_clients = [False for i in range(self.num_clients)]
        idx = [i for i in range(self.num_clients)]
        idx_ = np.random.choice(idx, int(slow_rate * self.num_clients))
        for i in idx_:
            slow_clients[i] = True

        return slow_clients

    def set_slow_clients(self):
        self.train_slow_clients = self.select_slow_clients(
            self.train_slow_rate)
        self.send_slow_clients = self.select_slow_clients(
            self.send_slow_rate)

    def select_clients(self):
        if self.random_join_ratio:
            join_clients = np.random.choice(range(self.join_clients, self.num_clients+1), 1, replace=False)[0]
        else:
            join_clients = self.join_clients
        selected_clients = list(np.random.choice(self.clients, join_clients, replace=False))

        return selected_clients

    def send_models(self):
        assert (len(self.clients) > 0)

        infos = []
        for client in self.clients:
            start_time = time.time()
            # client.set_parameters(self.global_model)
            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

            # info = {
            #             'weights': self.global_model.cpu().state_dict(), 
            #             'id': client.id, 
            #             "optimizer" : client.optimizer.state_dict(), 
            #             "train_samples": client.train_samples,
            #             'local': copy.deepcopy(client.model.cpu().get_local_state_dict()),
            #             'test': deepcopy((self.test_acc, self.test_num, self.auc))
            #         }
            info = deepcopy(client.get_infos())
            info.update({'weights': self.global_model.cpu().state_dict()})
            infos.append(info)

        return infos

    def receive_models(self, recieved_clients):
        
        recieved_clients.sort(key=lambda tup: tup['id'])
        
        for c in recieved_clients:
            self.clients[c['id']].load_infos(c)

        selected_clients = []
        selected_acc = []
        for c in self.clients:
            if c.sampled:
                selected_clients.append(c)
                selected_acc.append(c.test_acc / c.test_num)
        selected_clients.sort(key=lambda tup: tup.id)
        self.selected_clients = selected_clients
        print(sum(selected_acc) / len(selected_acc)) 
        assert (len(self.selected_clients) > 0)

        # active_clients = random.sample(
        #     self.selected_clients, int((1-self.client_drop_rate) * self.join_clients))
        active_clients = selected_clients

        self.uploaded_weights = []
        self.uploaded_models = []
        tot_samples = 0
        for client in active_clients:
            client_time_cost = client.train_time_cost['total_cost'] / client.train_time_cost['num_rounds'] + \
                    client.send_time_cost['total_cost'] / client.send_time_cost['num_rounds']
            # if client_time_cost <= self.time_threthold:
            tot_samples += client.train_samples 
            self.uploaded_weights.append(client.train_samples)
            self.uploaded_models.append(client.model)
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples

    def aggregate_parameters(self):
        assert (len(self.uploaded_models) > 0)

        self.global_model = copy.deepcopy(self.uploaded_models[0])
        for k, param in self.global_model.named_parameters():
            if not k in self.global_model.local:
                param.data.zero_()
            
        for w, client_model in zip(self.uploaded_weights, self.uploaded_models):
            self.add_parameters(w, client_model)

    def add_parameters(self, w, client_model):
        for (k1, server_param), (k2, client_param) in zip(self.global_model.named_parameters(), client_model.named_parameters()):
            assert k1 == k2
            if not k1 in self.global_model.local:
                server_param.data += client_param.data.clone() * w

    def save_global_model(self):
        # model_path = os.path.join("models", self.dataset)
        try: 
            model_path = wandb.run.dir
        except:
            model_path = os.path.join("models", self.dataset)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        torch.save(self.global_model, model_path)
    
    def save_local_models(self, infos, round):
        new_infos = copy.deepcopy(infos)
        # for info in new_infos:
        #     try:
        #         info.pop('weights')
        #         info.pop('local')
        #         info.pop('optimizer')
        #     except:
        #         pass

        try: 
            model_path = os.path.join("../logs", wandb.run.name)
        except:
            model_path = os.path.join("models", self.dataset)
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        torch.save(new_infos, os.path.join(model_path, "%d.pt" % round))

    def load_model(self):
        model_path = os.path.join("models", self.dataset)
        model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        assert (os.path.exists(model_path))
        self.global_model = torch.load(model_path)

    def model_exists(self):
        model_path = os.path.join("models", self.dataset)
        model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        return os.path.exists(model_path)
        
    def save_results(self):
        algo = self.dataset + "_" + self.algorithm
        result_path = "../results/"
        if not os.path.exists(result_path):
            os.makedirs(result_path)

        if (len(self.rs_test_acc)):
            algo = algo + "_" + self.goal + "_" + str(self.times)
            file_path = result_path + "{}.h5".format(algo)
            print("File path: " + file_path)

            with h5py.File(file_path, 'w') as hf:
                hf.create_dataset('rs_test_acc', data=self.rs_test_acc)
                hf.create_dataset('rs_test_auc', data=self.rs_test_auc)
                hf.create_dataset('rs_train_loss', data=self.rs_train_loss)

    def save_item(self, item, item_name):
        if not os.path.exists(self.save_folder_name):
            os.makedirs(self.save_folder_name)
        torch.save(item, os.path.join(self.save_folder_name, "server_" + item_name + ".pt"))

    def load_item(self, item_name):
        return torch.load(os.path.join(self.save_folder_name, "server_" + item_name + ".pt"))

    def test_metrics(self):
        num_samples = []
        tot_correct = []
        tot_auc = []
        for c in self.clients:
            ct, ns, auc = c.test_acc, c.test_num, c.auc 
            tot_correct.append(ct*1.0)
            tot_auc.append(auc*ns)
            num_samples.append(ns)

        ids = [c.id for c in self.clients]

        return ids, num_samples, tot_correct, tot_auc

    def train_metrics(self):
        num_samples = []
        losses = []
        for c in self.clients:
            cl, ns = c.train_metrics()
            num_samples.append(ns)
            losses.append(cl*1.0)

        ids = [c.id for c in self.clients]

        return ids, num_samples, losses

    # evaluate selected clients
    def evaluate(self, acc=None, loss=None):
        stats = self.test_metrics()
        # stats_train = self.train_metrics()

        test_acc = sum(stats[2])*1.0 / sum(stats[1]) * 100 if sum(stats[1]) != 0 else 0.0
        test_auc = sum(stats[3])*1.0 / sum(stats[1]) if sum(stats[1]) != 0 else 0.0
        # train_loss = sum(stats_train[2])*1.0 / sum(stats_train[1]) if sum(stats[1]) != 0 else 0.0
        accs = [a * 100 / n if n!=0 else 0 for a, n in zip(stats[2], stats[1])]
        aucs = [a / n if n!=0 else 0 for a, n in zip(stats[3], stats[1])]
        
        if acc == None:
            self.rs_test_acc.append(test_acc)
        else:
            acc.append(test_acc)
        
        # if loss == None:
        #     self.rs_train_loss.append(train_loss)
        # else:
        #     loss.append(train_loss)

        # print("Averaged Train Loss: {:.4f}".format(train_loss))


        # Global
        print("Global Test Accurancy: {:.4f}".format(test_acc))
        print("Global Test AUC: {:.4f}".format(test_auc))
        # Personalized
        print("Averaged Test Accurancy: {:.4f}".format(np.mean(accs)))
        print("Std Test Accurancy: {:.4f}".format(np.std(accs)))
        print("Averaged Test AUC: {:.4f}".format(np.mean(aucs)))
        print("Std Test AUC: {:.4f}".format(np.std(accs)))

        if not self.debug:
            res = {}
            res.update({"Global/Acc": test_acc})
            res.update({"Global/Auc": test_auc})
            res.update({"Averaged/Acc": np.mean(accs)})
            res.update({"Averaged/Acc_std": np.std(accs)})
            res.update({"Averaged/AUC": np.mean(aucs)})
            res.update({"Averaged/AUC_std": np.std(aucs)})
            wandb.log(res)

    def print_(self, test_acc, test_auc, train_loss):
        print("Average Test Accurancy: {:.4f}".format(test_acc))
        print("Average Test AUC: {:.4f}".format(test_auc))
        print("Average Train Loss: {:.4f}".format(train_loss))

    def check_done(self, acc_lss, top_cnt=None, div_value=None):
        for acc_ls in acc_lss:
            if top_cnt != None and div_value != None:
                find_top = len(acc_ls) - torch.topk(torch.tensor(acc_ls), 1).indices[0] > top_cnt
                find_div = len(acc_ls) > 1 and np.std(acc_ls[-top_cnt:]) < div_value
                if find_top and find_div:
                    pass
                else:
                    return False
            elif top_cnt != None:
                find_top = len(acc_ls) - torch.topk(torch.tensor(acc_ls), 1).indices[0] > top_cnt
                if find_top:
                    pass
                else:
                    return False
            elif div_value != None:
                find_div = len(acc_ls) > 1 and np.std(acc_ls[-top_cnt:]) < div_value
                if find_div:
                    pass
                else:
                    return False
            else:
                raise NotImplementedError
        return True
