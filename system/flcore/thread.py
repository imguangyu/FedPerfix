from copy import deepcopy

class Thread:

    def __init__(self, args, client_infos) -> None:
        self.device = client_infos['device']
        client_obj = client_infos['client_obj']
        self.client = client_obj(args, -1, 1, 1, train_slow=None, send_slow=None)
        self.client_map = client_infos['client_map']
        self.round = 0
        self.args = args

    def run(self, infos):
        client_results = []
        infos.sort(key=lambda tup: tup['id'])
        for i, client_idx in enumerate(self.client_map[self.round]):
            self.client.load_infos(infos[client_idx])
            self.client.model.to(self.device)
            if self.args.algorithm == 'APFL':
                self.client.model_per.to(self.device)
            if self.args.algorithm == 'Super':
                self.client.model_local.to(self.device)
                self.client.model_per.to(self.device)
                # self.client.model_local.to(self.device)
                self.client.start_mix = self.round >= self.args.mix_t

            self.client.id = client_idx
            self.client.device = self.device

            self.client.sampled = False
            if i < len(self.client_map[self.round]) * self.args.join_ratio:
                self.client.sampled = True
                self.client.train() 

            self.client.test_acc, self.client.test_num, self.client.auc = self.client.test_metrics()
            # self.client.loss, self.client.train_num = self.client.train_metrics()
            # self.client.round += 1
            
            result = deepcopy(self.client.get_infos())

            # if hasattr(self.client, 'pred'):
            #     result.update({'local': deepcopy(self.client.pred.cpu().state_dict())})
            # if hasattr(self.client.model, 'get_local_state_dict')  and self.args.algorithm != 'APFL' :
            #     result.update({'local': deepcopy(self.client.model.cpu().get_local_state_dict())})
            # if hasattr(self.client, 'model_per'):
            #     # result.update({"local" : (deepcopy(self.client.model_per.cpu().state_dict()), deepcopy(self.client.model_local.cpu().state_dict()))})
            #     if hasattr(self.client.model_per, 'get_local_state_dict') and self.args.algorithm != 'APFL':
            #         result.update({"local" : deepcopy(self.client.model_per.cpu().get_local_state_dict())})
            #     else:
            #         result.update({"local" : deepcopy(self.client.model_per.cpu().state_dict())})
            #         result.update({"alpha" : self.client.alpha})



            client_results.append(deepcopy(result))

        self.round += 1

        return client_results










    