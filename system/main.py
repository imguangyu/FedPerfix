#!/usr/bin/env python
import copy
from multiprocessing import Queue
import torch
import argparse
import os
import time
import warnings
import numpy as np
import torchvision
import logging
import timm
import random
from collections import defaultdict
from utils.custom_multiprocess import MyPool
from copy import deepcopy
import wandb
from torch.multiprocessing import current_process

from flcore.servers.serveravg import FedAvg
from flcore.servers.serverperavg import PerAvg
from flcore.servers.serverlocal import Local
from flcore.servers.serverapfl import APFL

from flcore.servers.serverbabu import FedBABU

from flcore.thread import Thread

from flcore.trainmodel.prompt import build_prompt
from flcore.trainmodel.prefix import build_prefix
from flcore.trainmodel.partial import build_paitial
from flcore.trainmodel.adapter import build_adapter


from utils.result_utils import average_data
from utils.mem_utils import MemReporter

logger = logging.getLogger()
logger.setLevel(logging.ERROR)

warnings.simplefilter("ignore")
# torch.manual_seed(0)

# hyper-params for Text tasks
vocab_size = 98635
max_len=200
hidden_dim=32


def saved_config(args):
    if args.method == 'none':
        logger.info('Using custom setting.')
    else:
        logger.info('Using saved {} setting.'.format(args.method))   
        args.local_learning_rate = 0.01
        if args.method == 'FedAVG':
            args.model = 'vit_small_patch16_224'
            args.algorithm = 'FedAvg'
            args.local_parts = []
            args.no_pt = True
        elif args.method == 'Local':
            args.model = 'vit_small_patch16_224'
            args.algorithm = 'Local'
            args.local_parts = []
            args.no_pt = True
        elif args.method == 'APFL':
            args.model = 'vit_small_patch16_224'
            args.algorithm = 'APFL'
            args.alpha = 0.25
            args.local_parts = []
            args.no_pt = True
        elif args.method == 'PerAvg': 
            args.model = 'vit_small_patch16_224'
            args.algorithm = 'PerAvg'
            args.local_parts = []
            args.local_learning_rate = 0.001
            args.no_pt = True
        elif args.method == 'FedBN':
            args.model = 'vit_small_patch16_224'
            args.algorithm = 'FedAvg'
            args.local_parts = ['norm']
            args.no_pt = True
        elif args.method == 'FedBABU':
            args.model = 'vit_small_patch16_224'
            args.algorithm = 'FedBABU'
            args.local_parts = ['head']
            args.no_pt = True
        elif args.method == 'FedRep':
            args.model = 'vit_small_patch16_224'
            args.algorithm = 'FedAvg'
            args.local_parts = ['head']
            args.no_pt = True
        elif args.method == 'VanillaAttention':
            args.model = 'vit_small_patch16_224'
            args.algorithm = 'FedAvg'
            args.local_parts = ['attn', 'head']
            args.no_pt = True
        elif args.method == 'FedPerfix':
            args.model = 'prefix'
            args.basic_model = 'vit_small_patch16_224'
            args.algorithm = 'FedAvg'
            args.no_pt = True
            args.local_parts = ['blocks.11','adapter', 'head'] \
                if args.dataset == 'cifar100' else ['adapter', 'head']
            args.mid_dim = 256
            args.scale = 1.5

        # NOTE: Please specify the local parts and algorithm for ResNet50 following the above setting
        elif args.method == 'ResNet50':
            args.model = 'resnet50'
            args.no_pt = True
    

def setup(args):

    time_list = []
    reporter = MemReporter()
    model_str = args.model
    args.model_str = model_str
    i = args.times


    # for i in range(args.prev, args.times):
    #     print(f"\n============= Running time: {i}th =============")
    print("Creating server and clients ...")
    # start = time.time()

    # Generate args.model
    if 'vit' in model_str:
        if not args.no_pt:
            basic_model = timm.create_model(model_str, num_classes=args.num_classes, pretrained=True).to(args.device)
            args.model = build_paitial(basic_model, args)
        else:
            basic_model = timm.create_model(model_str, num_classes=args.num_classes, pretrained=False).to(args.device)
            args.model = build_paitial(basic_model, args)
    elif model_str == 'vpt':
        if not args.no_pt:
            basic_model = timm.create_model(args.basic_model, num_classes=args.num_classes, pretrained=True).to(args.device)
            args.model =  build_prompt(basic_model, args)
        else:
            basic_model = timm.create_model(args.basic_model, num_classes=args.num_classes, pretrained=False).to(args.device)
            args.model =  build_prompt(basic_model, args)
    elif model_str == 'prefix':
        args.model_str = args.basic_model
        if not args.no_pt:
            raise NotImplementedError
        else:
            basic_model = timm.create_model(args.basic_model, num_classes=args.num_classes, pretrained=False).to(args.device)
            args.model =  build_prefix(basic_model, args)
    elif model_str == 'adapter':
        args.model_str = 'adapter_' + args.basic_model
        if not args.no_pt:
            raise NotImplementedError
        else:
            basic_model = timm.create_model(args.basic_model, num_classes=args.num_classes, pretrained=False).to(args.device)
            args.model =  build_adapter(basic_model, args)
    elif model_str == 'resnet50':
        if not args.no_pt:
            basic_model = timm.create_model(model_str, num_classes=args.num_classes, pretrained=True).to(args.device)
            # basic_model.head = basic_model.fc
            args.model = build_paitial(basic_model, args)
        else:
            basic_model = timm.create_model(model_str, num_classes=args.num_classes, pretrained=False).to(args.device)
            # basic_model.head = basic_model.fc
            args.model = build_paitial(basic_model, args)


    else:
        raise NotImplementedError

    # select algorithm
    if args.algorithm == "FedAvg":
        server = FedAvg(args, i)

    elif args.algorithm == "Local":
        server = Local(args, i)

    elif args.algorithm == "PerAvg":
        server = PerAvg(args, i)

    elif args.algorithm == "APFL":
        server = APFL(args, i)

    elif args.algorithm == "FedBABU":
        server = FedBABU(args, i)

    else:
        raise NotImplementedError

    return server


# Setup Functions
def set_random_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    ## NOTE: If you want every run to be exactly the same each time
    ##       uncomment the following lines
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def init_process(q, Thread):
    set_random_seed()
    global thread
    ci = q.get()
    # client = deepcopy(args.model)
    thread = Thread(ci[0], ci[1])
    print("Thread %d initialized!" % current_process()._identity[0])
    # save_path = ci[2]
    # if not ci[1].debug:
    #     wandb.init(group=save_path)
    # client.server = 

def run_clients(received_info):
    # try:
        # received_info, save_path = info
        # client, received_info = client_args
        # glo.set_value('writer', SummaryWriter(log_dir=save_path))
        return thread.run(received_info)
    # except KeyboardInterrupt:
    #     logging.info('exiting')
    #     return None

def allocate_clients_to_threads_personalized(args):
    mapping_dict = defaultdict(list)
    for round in range(args.global_rounds):
        # if args.client_sample<1.0:
            # num_clients = int(args.client_number*args.client_sample)
        num_clients = args.num_clients
        all_client_list = range(num_clients)
        sampled_client_list = random.sample(range(args.num_clients), int(args.num_clients * args.join_ratio))
        num_sampled = len(sampled_client_list)
        
        if num_sampled % args.num_threads==0 and num_sampled>0:
            clients_per_thread = int(num_sampled / args.num_threads)
            for c, t in enumerate(range(0, num_sampled, clients_per_thread)):
                idxs = [sampled_client_list[x] for x in range(t, t+clients_per_thread)]
                mapping_dict[c].append(idxs)


            remaining_client_list = [x for x in all_client_list if x not in sampled_client_list]
            if len(remaining_client_list) > 0:
                num_clients = len(remaining_client_list)
                clients_per_thread = int(num_clients / args.num_threads)
                for c, t in enumerate(range(0, num_clients, clients_per_thread)):
                    idxs = [remaining_client_list[x] for x in range(t, t+clients_per_thread)]
                    mapping_dict[c][round] += idxs
        else:
            logging.info("############ WARNING: Sampled client number not divisible by number of threads ##############")
            break
    return mapping_dict

if __name__ == "__main__":
    set_random_seed()
    total_start = time.time()
    torch.multiprocessing.set_start_method('spawn')

    parser = argparse.ArgumentParser()
    # general
    parser.add_argument('-go', "--goal", type=str, default="test", 
                        help="The goal for this experiment")
    parser.add_argument('-dev', "--device", type=str, default="cpu",
                        choices=["cpu", "cuda"])
    parser.add_argument('-did', "--device_id", type=str, default="0,1,2,3")
    parser.add_argument('-data', "--dataset", type=str, default="cifar100")
    parser.add_argument('-nb', "--num_classes", type=int, default=100)
    parser.add_argument('-m', "--model", type=str, default="vit_base_patch16_224_in21k")
    parser.add_argument('-p', "--head", type=str, default="cnn")
    parser.add_argument('-lbs', "--batch_size", type=int, default=64)
    parser.add_argument('-lr', "--local_learning_rate", type=float, default=0.005,
                        help="Local learning rate")
    parser.add_argument('-gr', "--global_rounds", type=int, default=50)
    parser.add_argument('-ls', "--local_steps", type=int, default=10)
    parser.add_argument('-algo', "--algorithm", type=str, default="FedAvg")
    parser.add_argument('-jr', "--join_ratio", type=float, default=0.125,
                        help="Ratio of clients per round")
    parser.add_argument('-rjr', "--random_join_ratio", type=bool, default=False,
                        help="Random ratio of clients per round")
    parser.add_argument('-nc', "--num_clients", type=int, default=64,
                        help="Total number of clients")
    parser.add_argument('-nt', "--num_threads", type=int, default=4,
                        help="Total number of threads")
    parser.add_argument('-pv', "--prev", type=int, default=0,
                        help="Previous Running times")
    parser.add_argument('-t', "--times", type=int, default=1,
                        help="Running times")
    parser.add_argument('-eg', "--eval_gap", type=int, default=1,
                        help="Rounds gap for evaluation")
    parser.add_argument('-dp', "--privacy", type=bool, default=False,
                        help="differential privacy")
    parser.add_argument('-dps', "--dp_sigma", type=float, default=0.0)
    parser.add_argument('-sfn', "--save_folder_name", type=str, default='models')

    parser.add_argument('-mtd', "--method", type=str, default="FedAVG")


    # practical
    parser.add_argument('-cdr', "--client_drop_rate", type=float, default=0.0,
                        help="Rate for clients that train but drop out")
    parser.add_argument('-tsr', "--train_slow_rate", type=float, default=0.0,
                        help="The rate for slow clients when training locally")
    parser.add_argument('-ssr', "--send_slow_rate", type=float, default=0.0,
                        help="The rate for slow clients when sending global model")
    parser.add_argument('-ts', "--time_select", type=bool, default=False,
                        help="Whether to group and select clients at each round according to time cost")
    parser.add_argument('-tth', "--time_threthold", type=float, default=10000,
                        help="The threthold for droping slow clients")
    # pFedMe / PerAvg / FedProx / FedAMP / FedPHP
    parser.add_argument('-bt', "--beta", type=float, default=0.0,
                        help="Average moving parameter for pFedMe, Second learning rate of Per-FedAvg, \
                        or L1 regularization weight of FedTransfer")
    parser.add_argument('-lam', "--lamda", type=float, default=1.0,
                        help="Regularization weight for pFedMe and FedAMP")
    parser.add_argument('-mu', "--mu", type=float, default=0,
                        help="Proximal rate for FedProx")
    parser.add_argument('-K', "--K", type=int, default=5,
                        help="Number of personalized training steps for pFedMe")
    parser.add_argument('-lrp', "--p_learning_rate", type=float, default=0.01,
                        help="personalized learning rate to caculate theta aproximately using K steps")
    # FedFomo
    parser.add_argument('-M', "--M", type=int, default=5,
                        help="Server only sends M client models to one client at each round")
    # FedMTL
    parser.add_argument('-itk', "--itk", type=int, default=4000,
                        help="The iterations for solving quadratic subproblems")
    # FedAMP
    parser.add_argument('-alk', "--alphaK", type=float, default=1.0, 
                        help="lambda/sqrt(GLOABL-ITRATION) according to the paper")
    parser.add_argument('-sg', "--sigma", type=float, default=1.0)
    # APFL
    parser.add_argument('-al', "--alpha", type=float, default=1.0)
    # Ditto / FedRep
    parser.add_argument('-pls', "--plocal_steps", type=int, default=1)
    # MOON
    parser.add_argument('-ta', "--tau", type=float, default=1.0)
    # FedBABU
    parser.add_argument('-fts', "--fine_tuning_steps", type=int, default=1)
    # APPLE
    parser.add_argument('-dlr', "--dr_learning_rate", type=float, default=0.0)
    parser.add_argument('-L', "--L", type=float, default=1.0)

    parser.add_argument('--debug', action='store_true', default=False,
                    help='reduce the sampler number for debugging')  
    # Prompt
    parser.add_argument('-vt', "--basic_model", type=str, default="vit_small_patch16_224_in21k")
    parser.add_argument('--local_prompt_num', type=int, default=10, metavar='N',
                        help='local prompt number for promper') 
    parser.add_argument('--global_prompt_num', type=int, default=10, metavar='N',
                        help='global prompt number for promper') 
    parser.add_argument('--tie', type=int, default=6, metavar='N',
                        help='tie for local and global') 
    parser.add_argument('--cluster', action='store_true', default=False,
                    help='enable client clustering')  
    parser.add_argument('--fixed_head', action='store_true', default=False,
                    help='enable fixed_head')  
    parser.add_argument('--tune_cls', action='store_true', default=False,
                    help='enable tune_cls')  
    parser.add_argument('--cluster_thres', type=float, default=0.5, metavar='N',
                        help='threshold for local aggregation') 

    parser.add_argument('--no_pt', action='store_true', default=False,
                help='no pretrain')  

    parser.add_argument('--tune_all', action='store_true', default=False,
                help='tune all parameters for prompts')  
    parser.add_argument("--local_parts", type=str, default="[]")

    # Prefix
    parser.add_argument('--mid_dim', type=int, default=512, metavar='D',
                    help='mid_dim for prefix adapter') 
    parser.add_argument('--scale', type=float, default=0.8, metavar='s',
                    help='scale for prefix adapter')    
    parser.add_argument('--prefix_depth', type=int, default=12, metavar='D',
                    help='depth for prefix adapter')  
    parser.add_argument('--vanilla', type=str, default='-',
                help='use vanilla prefix tuning')      
    parser.add_argument('--two_stage', action='store_true', default=False,
                help='use two_stage prefix tuning') 
    
    # Super
    parser.add_argument('-nu', "--nu", type=float, default=2,
                        help="NU rate for Super")
    parser.add_argument('-eval_lam', "--eval_lam", type=float, default=0.25,
                        help="NU rate for Super")
    parser.add_argument('-mix_t', "--mix_t", type=int, default=20,
                        help="start rounds for Super")
    

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id

    if args.device == "cuda" and not torch.cuda.is_available():
        print("\ncuda is not avaiable.\n")
        args.device = "cpu"

    args.local_parts = eval(args.local_parts)

    print("=" * 50)

    saved_config(args)
    print("Algorithm: {}".format(args.algorithm))
    print("Local batch size: {}".format(args.batch_size))
    print("Local steps: {}".format(args.local_steps))
    print("Local learing rate: {}".format(args.local_learning_rate))
    print("Total number of clients: {}".format(args.num_clients))
    print("Clients join in each round: {}".format(args.join_ratio))
    print("Client drop rate: {}".format(args.client_drop_rate))
    print("Time select: {}".format(args.time_select))
    print("Time threthold: {}".format(args.time_threthold))
    print("Global rounds: {}".format(args.global_rounds))
    print("Running times: {}".format(args.times))
    print("Dataset: {}".format(args.dataset))
    print("Local model: {}".format(args.model))
    print("Using device: {}".format(args.device))
    print("Local Parts: {}".format(args.local_parts))

    if args.device == "cuda":
        print("Cuda device id: {}".format(os.environ["CUDA_VISIBLE_DEVICES"]))
    print("=" * 50)

    server = setup(args)
    

    save_path = '../logs/{}_lr{:.0e}_e{}_c{}_{}_{}'.format(
                        args.method, args.local_learning_rate, args.local_steps, args.num_clients, args.dataset, time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()))


    if not args.debug:
        wandb.init(config=args, project="Your Project Name", entity="Your Entity Name")
        wandb.run.name = save_path.split(os.path.sep)[-1]

    
    mapping_dict = allocate_clients_to_threads_personalized(args)
    client_dicts = [{'device': "cuda:%d" % (i % torch.cuda.device_count()) if torch.cuda.device_count()>0 else "cpu", 'client_map':mapping_dict[i], 'client_obj': server.client_obj} for i in range(args.num_threads)]


    client_info = Queue()
        # clients = {}
    for i in range(args.num_threads):
        client_info.put((args, client_dicts[i]))

    
    pool = MyPool(args.num_threads, init_process, (client_info, Thread))


    # if not args.debug:
    # time.sleep(10 * (args.num_clients * args.join_ratio / args.num_threads))

    if args.algorithm in ['Local']:
        for i in range(args.global_rounds):
            s_t = time.time()
            # pool.join()
            # s.selected_clients = self.select_clients()
            client_infos = server.send_models()
            # server.save_local_models(client_infos, i)

            if i % server.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                server.evaluate()

            # for client in server.selected_clients:
            #     client.train()
            thread_outputs = pool.map(run_clients, [client_infos for _ in range(args.num_threads)])
            thread_outputs = [c for sublist in thread_outputs for c in sublist]

            server.receive_models(thread_outputs)
            # server.aggregate_parameters()

            server.Budget.append(time.time() - s_t)
            print("\nEvaluate global model")
            # server.evaluate()
            print('-'*25, 'time cost', '-'*25, server.Budget[-1])

        # print("\nBest global accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        # print(max(server.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(server.Budget[1:])/len(server.Budget[1:]))

        server.save_results()
        server.save_global_model()
        # server.save_local_models()
        pool.close()
        pool.join()

    else:

        for i in range(args.global_rounds):
            s_t = time.time()
            # pool.join()
            # s.selected_clients = self.select_clients()
            client_infos = server.send_models()
            # server.save_local_models(client_infos, i)


            if i % server.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                server.evaluate()

            # for client in server.selected_clients:
            #     client.train()
            thread_outputs = pool.map(run_clients, [client_infos for _ in range(args.num_threads)])
            thread_outputs = [c for sublist in thread_outputs for c in sublist]

            if i==49:
                server.save_local_models(thread_outputs, i)

            server.receive_models(thread_outputs)
            server.aggregate_parameters()

            server.Budget.append(time.time() - s_t)
            print("\nEvaluate global model")
            # server.evaluate()
            print('-'*25, 'time cost', '-'*25, server.Budget[-1])

        # print("\nBest global accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        # print(max(server.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(server.Budget[1:])/len(server.Budget[1:]))

        server.save_results()
        server.save_global_model()
        # server.save_local_models()
        pool.close()
        pool.join()


