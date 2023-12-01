import random
from flcore.clients.clientbabu import clientBABU
from flcore.servers.serverbase import Server
from threading import Thread


class FedBABU(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(args, clientBABU)
        self.client_obj = clientBABU

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")
        self.Budget = []
        # self.load_model()


    def train(self):
        for i in range(self.global_rounds+1):
            self.selected_clients = self.select_clients()
            self.send_models()

            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                self.evaluate()

            for client in self.selected_clients:
                client.train()

            # threads = [Thread(target=client.train)
            #            for client in self.selected_clients]
            # [t.start() for t in threads]
            # [t.join() for t in threads]

            self.receive_models()
            self.aggregate_parameters()

        print("\nBest global accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        print(max(self.rs_test_acc))

        for client in self.clients:
            client.fine_tune()
        print("\n-------------Evaluate fine-tuned model-------------")
        self.evaluate()

        self.save_results()
        self.save_global_model()


    # def receive_models(self, recieved_clients):
    #     recieved_clients.sort(key=lambda tup: tup['id'])
        
    #     for c in recieved_clients:
    #         self.clients[c['id']].load_infos(c)

    #     selected_clients = []
    #     for c in self.clients:
    #         if c.sampled:
    #             selected_clients.append(c)
    #     selected_clients.sort(key=lambda tup: tup.id)
    #     self.selected_clients = selected_clients


    #     assert (len(self.selected_clients) > 0)

    #     active_clients = random.sample(
    #         self.selected_clients, int((1-self.client_drop_rate) * self.join_clients))

    #     self.uploaded_weights = []
    #     self.uploaded_models = []
    #     tot_samples = 0
    #     for client in active_clients:
    #         client_time_cost = client.train_time_cost['total_cost'] / client.train_time_cost['num_rounds'] + \
    #                 client.send_time_cost['total_cost'] / client.send_time_cost['num_rounds']
    #         if client_time_cost <= self.time_threthold:
    #             tot_samples += client.train_samples
    #             self.uploaded_weights.append(client.train_samples)
    #             self.uploaded_models.append(client.model.base)
    #     for i, w in enumerate(self.uploaded_weights):
    #         self.uploaded_weights[i] = w / tot_samples
