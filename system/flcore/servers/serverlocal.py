from flcore.clients.clientavg import clientAVG
from flcore.servers.serverbase import Server
from threading import Thread
import time
import copy


class Local(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(args, clientAVG)
        self.client_obj = clientAVG

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")
        
        # self.load_model()

    def send_models(self):
        assert (len(self.clients) > 0)

        infos = []
        for client in self.clients:
            start_time = time.time()
            # client.set_parameters(self.global_model)
            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

            info = copy.deepcopy(client.get_infos())
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


    def train(self):
        for i in range(self.global_rounds+1):
            self.selected_clients = self.select_clients()

            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                self.evaluate()

            self.selected_clients = self.select_clients()
            for client in self.selected_clients:
                client.train()

            # threads = [Thread(target=client.train)
            #            for client in self.selected_clients]
            # [t.start() for t in threads]
            # [t.join() for t in threads]


        print("\nBest global accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        print(max(self.rs_test_acc))

        self.save_results()
        self.save_global_model()
