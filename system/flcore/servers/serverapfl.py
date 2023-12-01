from flcore.clients.clientapfl import clientAPFL
from flcore.servers.serverbase import Server
from threading import Thread
from copy import deepcopy
import time


class APFL(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(args, clientAPFL)
        self.client_obj = clientAPFL

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
    # def send_models(self):
    #     assert (len(self.clients) > 0)

    #     infos = []
    #     for client in self.clients:
    #         start_time = time.time()
    #         # client.set_parameters(self.global_model)
    #         client.send_time_cost['num_rounds'] += 1
    #         client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

    #         info = {
    #                 'weights': self.global_model.cpu().state_dict(), 
    #                 'id': client.id, 
    #                 "optimizer" : client.optimizer.state_dict(),
    #                 "train_samples": client.train_samples,
    #                 "local" : deepcopy(client.model_per.cpu().state_dict()),
    #                 'acc': client.test_acc,
    #                 "alpha": client.alpha
    #                 }
    #         infos.append(info)

    #     return infos

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

        self.save_results()
        self.save_global_model()
