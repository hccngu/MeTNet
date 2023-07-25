from cmath import nan
from copy import deepcopy
import os
import sklearn.metrics
import numpy as np
import pandas as pd
import sys
import time
from collections import OrderedDict
from . import word_encoder
from . import data_loader
import torch
from torch import autograd, optim, nn, threshold
from torch.autograd import Variable
from torch.nn import functional as F
# from pytorch_pretrained_bert import BertAdam
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.nn.parallel import DistributedDataParallel as DDP

from .viterbi import ViterbiDecoder
from utils.tripletloss import TripletLoss


class FewShotNERFramework_MTNet:

    def __init__(self, train_data_loader, val_data_loader, test_data_loader, args, tokenizer, use_sampled_data=False):
        '''
        train_data_loader: DataLoader for training.
        val_data_loader: DataLoader for validating.
        test_data_loader: DataLoader for testing.
        '''
        self.train_data_loader = train_data_loader
        self.val_data_loader = val_data_loader
        self.test_data_loader = test_data_loader
        self.args = args
        self.tokenizer = tokenizer
        self.use_sampled_data = use_sampled_data

    def __load_model__(self, ckpt):
        '''
        ckpt: Path of the checkpoint
        return: Checkpoint dict
        '''
        if os.path.isfile(ckpt):
            checkpoint = torch.load(ckpt)
            print("Successfully loaded checkpoint '%s'" % ckpt)
            return checkpoint
        else:
            raise Exception("No checkpoint found at '%s'" % ckpt)
    
    def item(self, x):
        '''
        PyTorch before and after 0.4
        '''
        torch_version = torch.__version__.split('.')
        if int(torch_version[0]) == 0 and int(torch_version[1]) < 4:
            return x[0]
        else:
            return x.item()

    def __generate_label_data__(self, query):
        label_tokens_index = []
        label_tokens_mask = []
        label_text_mask = []
        if self.args.have_otherO is True:
            for label_dic in query['label2tag']:
                for label_id in label_dic:
                    if label_id == 0:
                        label_tokens = ['other']
                    else:
                        label_tokens = label_dic[label_id].split('-')
                    label_tokens = ['[CLS]'] + label_tokens + ['[SEP]']
                    indexed_label_tokens = self.tokenizer.convert_tokens_to_ids(label_tokens)
                    # padding
                    while len(indexed_label_tokens) < 10:
                        indexed_label_tokens.append(0)
                    label_tokens_index.append(indexed_label_tokens)
                    # mask
                    mask = np.zeros((10), dtype=np.int32)
                    mask[:len(label_tokens)] = 1
                    label_tokens_mask.append(mask)
                    # text mask, also mask [CLS] and [SEP]
                    text_mask = np.zeros((10), dtype=np.int32)
                    text_mask[1:len(label_tokens)-1] = 1
                    label_text_mask.append(text_mask)
        else:
            for label_dic in query['label2tag']:
                for label_id in label_dic:
                    if label_id != 0:
                        label_tokens = label_dic[label_id].split('-')
                        label_tokens = ['[CLS]'] + label_tokens + ['[SEP]']
                        indexed_label_tokens = self.tokenizer.convert_tokens_to_ids(label_tokens)
                        # padding
                        while len(indexed_label_tokens) < 10:
                            indexed_label_tokens.append(0)
                        label_tokens_index.append(indexed_label_tokens)
                        # mask
                        mask = np.zeros((10), dtype=np.int32)
                        mask[:len(label_tokens)] = 1
                        label_tokens_mask.append(mask)
                        # text mask, also mask [CLS] and [SEP]
                        text_mask = np.zeros((10), dtype=np.int32)
                        text_mask[1:len(label_tokens)-1] = 1
                        label_text_mask.append(text_mask)

        label_tokens_index = torch.Tensor(label_tokens_index).long().cuda()
        label_tokens_mask = torch.Tensor(label_tokens_mask).long().cuda()
        label_text_mask = torch.Tensor(label_text_mask).long().cuda()

        label_data = {}
        label_data['word'] = label_tokens_index
        label_data['mask'] = label_tokens_mask
        label_data['text_mask'] = label_text_mask
        return label_data

    def __zero_grad__(self, params):
        for p in params:
            if p.grad is not None:
                p.grad.zero_()

    def __get_sample_pairs__(self, data):
        data_1 = {}
        data_2 = {}
        data_1['word_emb'] = data['word_emb'][[l in [*range(1, self.args.N+1)] for l in data['label']]]
        data_1['label'] = data['label'][[l in [*range(1, self.args.N+1)] for l in data['label']]]
        data_2['word_emb'] = data['word_emb'][[l in [*range(0, self.args.N+1)] for l in data['label']]]
        data_2['label'] = data['label'][[l in [*range(0, self.args.N+1)] for l in data['label']]]

        return data_1, data_2

    def __generate_pair_label__(self, label1, label2):
        pair_label = []
        for l1 in label1:
            for l2 in label2:
                if l1 == l2:
                    pair_label.append(1.0)
                else:
                    pair_label.append(0.0)
        return torch.Tensor(pair_label).cuda()

    def __generate_query_pair_label__(self, query_dis, query_label):
        query_pair_label = []
        after_query_dis = []
        for i, l in enumerate(query_label):
            tmp = torch.zeros([1, self.args.N])
            if l == -1:
                continue
            elif l == 0:
                query_pair_label.append(tmp)
                after_query_dis.append(query_dis[i])
            else:
                tmp[0, l-1] = 1
                query_pair_label.append(tmp)
                after_query_dis.append(query_dis[i])
        query_pair_label = torch.cat(query_pair_label, dim=0).view(-1).cuda()
        after_query_dis = torch.stack(after_query_dis).view(-1).cuda()

        return after_query_dis, query_pair_label
    
    def __get_proto__(self, label_data_emb, label_data_text_mask, support_emb, support, model):
        if self.args.label_name_mode == 'mean':
            temp_word_list = []
            for i, word_emb_list in enumerate(support_emb):
                temp_word_list.append(word_emb_list[support['text_mask'][i]==1])
            temp_label_list = []
            temp_word_list = torch.cat(temp_word_list)  # [x, 768]
            temp_label_list = torch.cat(support['label'], dim=0)  # [x,]
            assert temp_word_list.shape[0] == temp_label_list.shape[0]
            Proto = []
            if self.args.have_otherO is True:
                for i in range(self.args.N+1):
                    Proto.append(torch.mean(temp_word_list[temp_label_list==i], dim=0).view(1,-1))
            else:
                for i in range(self.args.N):
                    Proto.append(torch.mean(temp_word_list[temp_label_list==i+1], dim=0).view(1,-1))
            Proto = torch.cat(Proto)
        elif self.args.label_name_mode == 'LnAsQ':
            # get Q = mean(init label name)
            Q = []
            K = {}
            assert label_data_emb.shape[0] == label_data_text_mask.shape[0]
            for i, l_ebd in enumerate(label_data_emb):  # [10, 768]
                p = l_ebd[label_data_text_mask[i]==1]
                # K[i] = p
                p = p.mean(dim=0)
                Q.append(p.view(1,-1))
            Q = torch.cat(Q,0)

            # get K or V = cat(label name and word in class i)
            temp_word_list = []
            for i, word_emb_list in enumerate(support_emb):
                temp_word_list.append(word_emb_list[support['text_mask'][i]==1])
            temp_label_list = []
            temp_word_list = torch.cat(temp_word_list)  # [x, 768]
            temp_label_list = torch.cat(support['label'], dim=0)  # [x,]
            assert temp_word_list.shape[0] == temp_label_list.shape[0]
            if self.args.have_otherO is True:
                for i in range(self.args.N+1):
                    K[i] = temp_word_list[temp_label_list==i]
            else:
                for i in range(self.args.N):
                    K[i] = temp_word_list[temp_label_list==i+1]
            
            # Attention
            Proto = []
            for i, q in enumerate(Q):
                temp = torch.mm(model.att(q.view(1, -1)), K[i].t())
                att_weights = F.softmax(F.layer_norm(temp, normalized_shape=(temp.shape[0],temp.shape[1])), dim=1)
                # print("att_weights:", att_weights)
                proto = torch.mm(att_weights, K[i])  # [1, 768]
                Proto.append(proto)
            Proto = torch.cat(Proto)
        elif self.args.label_name_mode == 'LnAsQKV':
            # get Q = mean(init label name)
            Q = []
            K = {}
            assert label_data_emb.shape[0] == label_data_text_mask.shape[0]
            for i, l_ebd in enumerate(label_data_emb):  # [10, 768]
                p = l_ebd[label_data_text_mask[i]==1]
                K[i] = p
                p = p.mean(dim=0)
                Q.append(p.view(1,-1))
            Q = torch.cat(Q,0)

            # get K or V = cat(label name and word in class i)
            temp_word_list = []
            for i, word_emb_list in enumerate(support_emb):
                temp_word_list.append(word_emb_list[support['text_mask'][i]==1])
            temp_label_list = []
            temp_word_list = torch.cat(temp_word_list)  # [x, 768]
            temp_label_list = torch.cat(support['label'], dim=0)  # [x,]
            assert temp_word_list.shape[0] == temp_label_list.shape[0]

            if self.args.have_otherO is True:
                for i in range(self.args.N+1):
                    K[i] = torch.cat((K[i],temp_word_list[temp_label_list==i]),dim=0)
            else:
                for i in range(self.args.N):
                    K[i] = torch.cat((K[i],temp_word_list[temp_label_list==i+1]),dim=0)
            
            # Attention
            Proto = []
            for i, q in enumerate(Q):
                temp = torch.mm(model.att(q.view(1, -1)), K[i].t())
                att_weights = F.softmax(F.layer_norm(temp, normalized_shape=(temp.shape[0],temp.shape[1])), dim=1)
                # print("att_weights:", att_weights)
                proto = torch.mm(att_weights, K[i])  # [1, 768]
                Proto.append(proto)
            Proto = torch.cat(Proto)
        else:
            raise NotImplementedError

        return Proto

    def __pos_dist__(self, instances, class_proto):  # ins:[N*K, 256], cla:[N, 256]
        return torch.pow(torch.pow(class_proto.unsqueeze(0) - instances.unsqueeze(1), 2).sum(-1), 0.5)

    def train(self,
              model,
              model_name,
              learning_rate=1e-4,
              train_iter=30000,
              val_iter=1000,
              val_step=2000,
              load_ckpt=None,
              save_ckpt=None,
              warmup_step=300,
              grad_iter=1,
              fp16=False,
              use_sgd_for_bert=False):
        '''
        model: a FewShotREModel instance
        model_name: Name of the model
        B: Batch size
        N: Num of classes for each batch
        K: Num of instances for each class in the support set
        Q: Num of instances for each class in the query set
        ckpt_dir: Directory of checkpoints
        learning_rate: Initial learning rate
        train_iter: Num of iterations of training
        val_iter: Num of iterations of validating
        val_step: Validate every val_step steps
        '''
        print("Start training...")
    
        # Init optimizer
        print('Use bert optim!')

        # set Bert learning rate
        parameters_to_optimize = list(model.word_encoder.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        parameters_to_optimize = [
            {'params': [p for n, p in parameters_to_optimize 
                if not any(nd in n for nd in no_decay)], 'weight_decay': self.args.bert_wd},
            {'params': [p for n, p in parameters_to_optimize
                if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        if use_sgd_for_bert:
            bert_optimizer = torch.optim.SGD(parameters_to_optimize, lr=self.args.bert_lr)
        else:
            bert_optimizer = AdamW(parameters_to_optimize, lr=self.args.bert_lr, correct_bias=False)
        bert_scheduler = get_linear_schedule_with_warmup(bert_optimizer, num_warmup_steps=warmup_step, num_training_steps=train_iter) 

        # set learning rate of model without Bert
        parameters_to_optimize = list(model.named_parameters())
        without = ['word_encoder']
        no_decay = ['bias']
        parameters_to_optimize = [
            {'params': [p for n, p in parameters_to_optimize 
                if not (any(nd in n for nd in no_decay) or any(wo in n for wo in without))], 'weight_decay': self.args.wobert_wd},
            {'params': [p for n, p in parameters_to_optimize
                if any(nd in n for nd in no_decay) and not any(wo in n for wo in without)], 'weight_decay': 0.0}
        ]
        wobert_optimizer = AdamW(parameters_to_optimize, lr=self.args.meta_lr, correct_bias=False)
        wobert_scheduler = get_linear_schedule_with_warmup(wobert_optimizer, num_warmup_steps=warmup_step, num_training_steps=train_iter) 

        # load model
        if load_ckpt:
            state_dict = self.__load_model__(load_ckpt)['state_dict']
            own_state = model.state_dict()
            for name, param in state_dict.items():
                if name not in own_state:
                    print('ignore {}'.format(name))
                    continue
                print('load {} from {}'.format(name, load_ckpt))
                own_state[name].copy_(param)

        if fp16:
            from apex import amp
            model, optimizer = amp.initialize(model, optimizer, opt_level='O1')

        model.train()
        loss_func = TripletLoss(args=self.args)

        # Training
        best_f1 = 0.0
        iter_loss = 0.0
        iter_sample = 0
        pred_cnt = 0
        label_cnt = 0
        correct_cnt = 0

        it = 0
        while it + 1 < train_iter:
            for _, (support, query) in enumerate(self.train_data_loader):
                '''
                support/query:
                {
                    'word': 2维tensor矩阵[~n*k, max_length], 里面都是单词在此表中的索引号[[101, 1996,...,0],[...]],
                    'mask': 同上, 补PAD的地方为0, 其他地方为1[[1, 1,..., 0],[...]],
                    'label': 列表[tensor([0, 1,..., 0, 0]), tensor([0, 0,..., 0]),...]set(-1, 0, 1, 2),
                    'sentence_num': [5, 5, 5, 5](长度为batch_size大小, 每个位置表示单个batch中的句子数目),
                    'text_mask': 与mask类似, 就是补CLS和SEP的位置也都为0了,
                    query独有:
                    'label2tag':  # 对应一个batch里的4个部分
                    [
                        {   0: 'O', 
                            1: 'product-software', 
                            2: 'location-island', 
                            3: 'person-director', 
                            4: 'event-protest', 
                            5: 'other-disease'
                        }, 
                        {
                            0: 'O',
                            1: 'location-GPE',
                            2: 'location-road/railway/highway/transit', 
                            3: 'person-director', 
                            4: 'other-biologything', 
                            5: 'building-airport'
                        }, 
                        {
                            0: 'O', 
                            1: 'event-attack/battle/war/militaryconflict', 
                            2: 'product-software', 
                            3: 'other-award', 
                            4: 'building-restaurant', 
                            5: 'person-politician'
                        }, 
                        {
                            0: 'O', 
                            1: 'person-artist/author', 
                            2: 'building-hotel', 
                            3: 'other-award', 
                            4: 'location-mountain', 
                            5: 'other-god'
                        }
                    ]
                }
                '''
                margin = model.param
                alpha = model.alpha
                if torch.cuda.is_available():
                    for k in support:
                        if k != 'label' and k != 'sentence_num':
                            support[k] = support[k].cuda()
                            query[k] = query[k].cuda()
                    query_label = torch.cat(query['label'], 0)
                    query_label = query_label.cuda()

                # get proto init rep
                label_data = self.__generate_label_data__(query)
                label_data_emb = model.word_encoder(label_data['word'], label_data['mask'])  # [num_label_sent, 10, 768]
                support_emb = model.word_encoder(support['word'], support['mask'])
                Proto = self.__get_proto__(label_data_emb, label_data['text_mask'], support_emb, support, model)  #[N, 768]
                
                # support,proto -> MLP -> new emb
                support_label = torch.cat(support['label'], dim=0)
                support_emb = support_emb[support['text_mask']==1]
                support_afterMLP_emb = model(support_emb)
                proto_afterMLP_emb = model(Proto)
                if self.args.use_proto_as_neg is True:
                    if self.args.have_otherO is True:
                        support_label = torch.cat((support_label, torch.tensor([0 for _ in range(self.args.N)],dtype=torch.int64)))
                    else:
                        support_label = torch.cat((support_label, torch.tensor([0 for _ in range(1,self.args.N)],dtype=torch.int64)))
                    support_afterMLP_emb = torch.cat((support_afterMLP_emb, proto_afterMLP_emb+1e-8),dim=0)
                    support_dis = []
                    if self.args.have_otherO is True:
                        for i in range(self.args.N+1):
                            support_dis_one_line = self.__pos_dist__(proto_afterMLP_emb[i].view(1, -1), support_afterMLP_emb)
                            temp_lst = [0 for _ in range(support_dis_one_line.shape[1])]
                            temp_lst[-(self.args.N+1-i)] = 1
                            temp_lst = np.array(temp_lst)
                            support_dis_one_line = support_dis_one_line.view(-1)[temp_lst == 0].view(1, -1)
                            support_dis.append(support_dis_one_line)
                    else:
                        for i in range(self.args.N):
                            support_dis_one_line = self.__pos_dist__(proto_afterMLP_emb[i].view(1, -1), support_afterMLP_emb)
                            temp_lst = [0 for _ in range(support_dis_one_line.shape[1])]
                            temp_lst[-(self.args.N-i)] = 1
                            temp_lst = np.array(temp_lst)
                            support_dis_one_line = support_dis_one_line.view(-1)[temp_lst == 0].view(1, -1)
                            support_dis.append(support_dis_one_line)
                    support_dis = torch.cat(support_dis, dim=0).view(-1)  # [N+1, N*K]
                else:
                    support_dis = self.__pos_dist__(proto_afterMLP_emb, support_afterMLP_emb).view(-1)  # [N, N*K]
                # if self.args.use_diff_threshold == False:
                # print("support_dis_before_norm:", torch.max(support_dis).item(), torch.min(support_dis).item(), torch.mean(support_dis).item())
                support_dis = F.layer_norm(support_dis, normalized_shape=[support_dis.shape[0]], bias=torch.full((support_dis.shape[0],), self.args.ln_bias).cuda())
                # print("support_dis_after_norm:", torch.max(support_dis).item(), torch.min(support_dis).item(), torch.mean(support_dis).item())
                if self.args.have_otherO is True:
                    support_dis = support_dis.view(self.args.N+1, -1)
                else:
                    support_dis = support_dis.view(self.args.N, -1)
                # if self.args.use_diff_threshold == True:
                #     margin = torch.mean(support_dis)
                support_loss = loss_func(support_dis, support_label, margin, alpha)
                
                self.__zero_grad__(model.fc.parameters())  
                grads_fc = autograd.grad(support_loss, model.fc.parameters(), allow_unused=True, retain_graph=True)
                fast_weights_fc, orderd_params_fc = model.cloned_fc_dict(), OrderedDict()
                for (key, val), grad in zip(model.fc.named_parameters(), grads_fc):
                    fast_weights_fc[key] = orderd_params_fc[key] = val - self.args.task_lr * grad  # grad中weight数量级是1e-4，bias是1e-11，有点太小了？
                
                fast_weights = {}
                fast_weights['fc'] = fast_weights_fc

                train_support_loss = []
                for _ in range(self.args.train_support_iter - 1):
                    support_afterMLP_emb = model(support_emb, fast_weights)
                    proto_afterMLP_emb = model(Proto, fast_weights)
                    if self.args.use_proto_as_neg == True:
                        # support_label = torch.cat((support_label, torch.tensor([0 for _ in range(1,self.args.N)],dtype=torch.int64)))
                        support_afterMLP_emb = torch.cat((support_afterMLP_emb, proto_afterMLP_emb+1e-8),dim=0)
                        support_dis = []
                        if self.args.have_otherO is True:
                            for i in range(self.args.N+1):
                                support_dis_one_line = self.__pos_dist__(proto_afterMLP_emb[i].view(1, -1), support_afterMLP_emb)
                                temp_lst = [0 for _ in range(support_dis_one_line.shape[1])]
                                temp_lst[-(self.args.N+1-i)] = 1
                                temp_lst = np.array(temp_lst)
                                support_dis_one_line = support_dis_one_line.view(-1)[temp_lst == 0].view(1, -1)
                                support_dis.append(support_dis_one_line)
                        else:
                            for i in range(self.args.N):
                                support_dis_one_line = self.__pos_dist__(proto_afterMLP_emb[i].view(1, -1), support_afterMLP_emb)
                                temp_lst = [0 for _ in range(support_dis_one_line.shape[1])]
                                temp_lst[-(self.args.N-i)] = 1
                                temp_lst = np.array(temp_lst)
                                support_dis_one_line = support_dis_one_line.view(-1)[temp_lst == 0].view(1, -1)
                                support_dis.append(support_dis_one_line)
                        support_dis = torch.cat(support_dis, dim=0).view(-1)
                    else:
                        support_dis = self.__pos_dist__(proto_afterMLP_emb, support_afterMLP_emb).view(-1)
                    # if self.args.use_diff_threshold == False:  
                    # print("support_dis_before_norm:", torch.max(support_dis).item(), torch.min(support_dis).item(), torch.mean(support_dis).item())
                    support_dis = F.layer_norm(support_dis, normalized_shape=[support_dis.shape[0]], bias=torch.full((support_dis.shape[0],), self.args.ln_bias).cuda())
                    # print("support_dis_after_norm:", torch.max(support_dis).item(), torch.min(support_dis).item(), torch.mean(support_dis).item())
                    if self.args.have_otherO is True:
                        support_dis = support_dis.view(self.args.N+1, -1)
                    else:
                        support_dis = support_dis.view(self.args.N, -1)
                    # if self.args.use_diff_threshold == True:
                    #     margin = torch.mean(support_dis)
                    support_loss = loss_func(support_dis, support_label, margin, alpha)
                    train_support_loss.append(support_loss.item())
                    # print_info = 'train_support, ' + str(support_loss.item())
                    # print('\033[0;31;40m{}\033[0m'.format(print_info))
                    self.__zero_grad__(orderd_params_fc.values())

                    grads_fc = torch.autograd.grad(support_loss, orderd_params_fc.values(), allow_unused=True, retain_graph=True)
                    for (key, val), grad in zip(orderd_params_fc.items(), grads_fc):
                        if grad is not None:
                            fast_weights['fc'][key] = orderd_params_fc[key] = val - self.args.task_lr * grad
                
                # query, proto -> MLP -> new emb
                query_emb = model.word_encoder(query['word'], query['mask'])
                query_emb = query_emb[query['text_mask']==1]
                query_afterMLP_emb = model(query_emb, fast_weights)
                proto_afterMLP_emb = model(Proto, fast_weights)
                if self.args.use_proto_as_neg == True:
                    if self.args.have_otherO is True:
                        query_label = torch.cat((query_label, torch.tensor([0 for _ in range(self.args.N)],dtype=torch.int64).cuda()))
                    else:
                        query_label = torch.cat((query_label, torch.tensor([0 for _ in range(1,self.args.N)],dtype=torch.int64).cuda()))
                    query_afterMLP_emb = torch.cat((query_afterMLP_emb,proto_afterMLP_emb+1e-8),dim=0)
                    query_dis = []
                    if self.args.have_otherO is True:
                        for i in range(self.args.N+1):
                            query_dis_one_line = self.__pos_dist__(proto_afterMLP_emb[i].view(1, -1), query_afterMLP_emb)
                            temp_lst = [0 for _ in range(query_dis_one_line.shape[1])]
                            temp_lst[-(self.args.N+1-i)] = 1
                            temp_lst = np.array(temp_lst)
                            query_dis_one_line = query_dis_one_line.view(-1)[temp_lst == 0].view(1, -1)
                            query_dis.append(query_dis_one_line)
                    else:
                        for i in range(self.args.N):
                            query_dis_one_line = self.__pos_dist__(proto_afterMLP_emb[i].view(1, -1), query_afterMLP_emb)
                            temp_lst = [0 for _ in range(query_dis_one_line.shape[1])]
                            temp_lst[-(self.args.N-i)] = 1
                            temp_lst = np.array(temp_lst)
                            query_dis_one_line = query_dis_one_line.view(-1)[temp_lst == 0].view(1, -1)
                            query_dis.append(query_dis_one_line)
                    query_dis = torch.cat(query_dis, dim=0).view(-1)
                else:
                    query_dis = self.__pos_dist__(proto_afterMLP_emb, query_afterMLP_emb).view(-1)  # [N, N*K]
                # if self.args.use_diff_threshold == False:
                # print("query_dis_before_norm:", torch.max(query_dis).item(), torch.min(query_dis).item(), torch.mean(query_dis).item())
                query_dis = F.layer_norm(query_dis, normalized_shape=[query_dis.shape[0]], bias=torch.full((query_dis.shape[0],), self.args.ln_bias).cuda())
                # print("query_dis_before_norm:", torch.max(query_dis).item(), torch.min(query_dis).item(), torch.mean(query_dis).item())
                if self.args.have_otherO is True:
                    query_dis = query_dis.view(self.args.N+1, -1)
                else:
                    query_dis = query_dis.view(self.args.N, -1)
                # if self.args.use_diff_threshold == True:
                #     margin = torch.mean(query_dis)
                query_loss = loss_func(query_dis, query_label, margin, alpha)

                # update param
                bert_optimizer.zero_grad()
                wobert_optimizer.zero_grad()
                query_loss.backward()
                bert_optimizer.step()
                wobert_optimizer.step()
                bert_scheduler.step()
                wobert_scheduler.step()

                # make prediction
                if self.args.use_proto_as_neg == True:
                    if self.args.have_otherO is True:
                        query_dis = query_dis[:, :-(self.args.N)]
                        query_label = query_label[:-(self.args.N)]
                    else:
                        query_dis = query_dis[:, :-(self.args.N-1)]
                        query_label = query_label[:-(self.args.N-1)]

                if self.args.use_diff_threshold == True:
                    threshold = []
                    if self.args.have_otherO is True:
                        for i in range(0, self.args.N+1):
                            if self.args.threshold_mode == 'mean':
                                threshold.append(torch.mean(query_dis[i][query_label==i]).item())
                            elif self.args.threshold_mode == 'max':
                                threshold.append(torch.max(query_dis[i][query_label==i]).item())
                            else:
                                raise NotImplementedError
                    else:
                        for i in range(1, self.args.N+1):
                            if self.args.threshold_mode == 'mean':
                                threshold.append(torch.mean(query_dis[i-1][query_label==i]).item())
                            elif self.args.threshold_mode == 'max':
                                threshold.append(torch.max(query_dis[i-1][query_label==i]).item())
                            else:
                                raise NotImplementedError

                    print("threshold:", threshold)
                    query_dis = query_dis.t()  # [x, N]
                    query_pred = []
                    
                    if self.args.have_otherO is True:
                        for tmp in query_dis:
                            query_pred.append(torch.min(tmp, dim=0)[1].item())
                    else:
                        for tmp in query_dis:
                            flag = []
                            for tm, th in zip(tmp, threshold):
                                if tm < th:
                                    flag.append(tm.view(1,-1))
                                else:
                                    flag.append(torch.tensor([[9999.0]]).cuda())
                            flag = torch.cat(flag).view(-1)
                            if torch.min(flag) != 9999.0:
                                query_pred.append(torch.min(tmp, dim=0)[1].item()+1)
                            else:
                                query_pred.append(0)
                    query_pred = torch.Tensor(query_pred).cuda()
                else:
                    query_dis = query_dis.t()  # [x, N]
                    query_pred = []
                    
                    if self.args.have_otherO is True:
                        for tmp in query_dis:
                            query_pred.append(torch.min(tmp, dim=0)[1].item())
                    else:
                        if self.args.multi_margin is True:
                            for tmp in query_dis:
                                flag = []
                                for tm, th in zip(tmp, margin):
                                    if tm < th:
                                        flag.append(tm.view(1,-1))
                                    else:
                                        flag.append(torch.tensor([[9999.0]]).cuda())
                                flag = torch.cat(flag).view(-1)
                                if torch.min(flag) != 9999.0:
                                    query_pred.append(torch.min(tmp, dim=0)[1].item()+1)
                                else:
                                    query_pred.append(0)
                        else:
                            for tmp in query_dis:
                                if any(t < margin for t in tmp):
                                    query_pred.append(torch.min(tmp, dim=0)[1].item()+1)
                                else:
                                    query_pred.append(0)
                    query_pred = torch.Tensor(query_pred).cuda()
                
                assert query_pred.shape[0] == query_label.shape[0]

                tmp_pred_cnt, tmp_label_cnt, correct = model.metrics_by_entity(query_pred, query_label)

                iter_loss += self.item(query_loss.data)
                pred_cnt += tmp_pred_cnt
                label_cnt += tmp_label_cnt
                correct_cnt += correct
                iter_sample += 1

                if (it + 1) % 100 == 0 or (it + 1) % val_step == 0:
                    precision = correct_cnt / (pred_cnt + 1e-10)
                    recall = correct_cnt / (label_cnt + 1e-10)
                    f1 = 2 * precision * recall / (precision + recall + 1e-10)
                    print('step: {0:4} | loss: {1:2.6f} | [ENTITY] precision: {2:3.4f}, recall: {3:3.4f}, f1: {4:3.4f}'\
                        .format(it + 1, iter_loss/ iter_sample, precision, recall, f1) + '\r')
                    print('margin:', margin)
                    # print('alpha:', alpha.item())
                    # a = deepcopy(alpha)
                    # print('alpha_after_sigmoid:', torch.sigmoid(a).item())

                    iter_loss = 0.
                    iter_sample = 0.
                    pred_cnt = 0
                    label_cnt = 0
                    correct_cnt = 0

                if (it + 1) % val_step == 0:
                    # torch.save({'state_dict': model.state_dict()}, 'current_siamese.ckpt')
                    _, _, f1, _, _, _, _ = self.eval(model, val_iter)
                    model.train()
                    if f1 > best_f1:
                        print('Best checkpoint')
                        torch.save({'state_dict': model.state_dict()}, save_ckpt)
                        best_f1 = f1
                    iter_loss = 0.
                    iter_sample = 0.
                    pred_cnt = 0
                    label_cnt = 0
                    correct_cnt = 0

                if (it + 1)  == train_iter:
                    break
                it += 1
                
        print("\n####################\n")
        print("Finish training " + model_name)
    
    def __save_test_inference__(self, logits, pred, query):

        # query 去掉-1
        new_query_label = []
        new_query_word = []
        new_query_textmask = []
        for lq, wq, tq in zip(query['label'], query['word'], query['text_mask']):
            pass

                
        # logits = F.softmax(logits, dim=-1)
        # 将word转为真实单词
        sentence_list = []  # 二维列表
        for words, mask in zip(query['word'], query['text_mask']):
            real_words = []
            for word in words[mask==1]:
                real_words.append(self.tokenizer.decode(word).replace(" ", ""))
            sentence_list.append(real_words)

        # 将label和pred转为label word
        sentence_num = []
        sentence_num.append(0)
        tmp = 0
        for i in query['sentence_num']:
            tmp += i
            sentence_num.append(tmp)
        real_label_list = []  # 二维列表
        pred_label_list = []  # 二维列表
        label_name_list = []  # 二维列表
        # pred和logits切成二维矩阵
        pred_list = []
        # logits_list = []
        sentence_len = []
        sentence_len.append(0)
        tmp = 0
        for _labels in query['label']:
            tmp += _labels.shape[0]
            sentence_len.append(tmp)
        for i in range(len(sentence_len)-1):
            tmp2 = pred[sentence_len[i]: sentence_len[i+1]]
            # tmp3 = logits[sentence_len[i]: sentence_len[i+1]]
            pred_list.append(tmp2.cpu())
            # logits_list.append(tmp3.cpu().detach().numpy().tolist())

        for i in range(len(sentence_num)-1):
            for j in range(sentence_num[i], sentence_num[i+1]):
                tmp_label_list = []
                tmp_pred_list = []
                tmp_label_name_list = []
                assert query['label'][j].shape[0] == pred_list[j].shape[0]
                for l, p in zip(query['label'][j], pred_list[j]):
                    if l == -1:
                        tmp_label_list.append(str(-1))
                    else:
                        tmp_label_list.append(query['label2tag'][i][l.item()])
                    tmp_pred_list.append(query['label2tag'][i][p.item()])
                    tmp_label_name_list.append(str(query['label2tag'][i]))
                real_label_list.append(tmp_label_list)
                pred_label_list.append(tmp_pred_list)
                label_name_list.append(tmp_label_name_list)  # 每个元任务的label_list
        
        return sentence_list, real_label_list, pred_label_list, label_name_list

    def eval(self,
            model,
            eval_iter,
            ckpt=None): 
        '''
        model: a FewShotREModel instance
        B: Batch size
        N: Num of classes for each batch
        K: Num of instances for each class in the support set
        Q: Num of instances for each class in the query set
        eval_iter: Num of iterations
        ckpt: Checkpoint path. Set as None if using current model parameters.
        return: Accuracy
        '''
        print("evaluating...")
        
        model.eval()
        loss_func = TripletLoss(args=self.args)
        
        if ckpt is None:
            print("Use val dataset")
            eval_dataset = self.val_data_loader
            # print("Use test dataset")
            # eval_dataset = self.test_data_loader
        else:
            print("Use test dataset")
            if ckpt != 'none':
                state_dict = self.__load_model__(ckpt)['state_dict']
                own_state = model.state_dict()
                for name, param in state_dict.items():
                    if name not in own_state:
                        continue
                    own_state[name].copy_(param)
            eval_dataset = self.test_data_loader
        
        if self.args.only_use_test:
            eval_dataset = self.test_data_loader

        margin = model.param
        alpha = model.alpha
        print("margin:", margin)
        print("alpha:", alpha)

        if self.args.margin != -1:
            margin = self.args.margin
            print('set margin:', margin)

        pred_cnt = 0 # pred entity cnt
        label_cnt = 0 # true label entity cnt
        correct_cnt = 0 # correct predicted entity cnt

        fp_cnt = 0 # misclassify O as I-
        fn_cnt = 0 # misclassify I- as O
        total_token_cnt = 0 # total token cnt
        within_cnt = 0 # span correct but of wrong fine-grained type 
        outer_cnt = 0 # span correct but of wrong coarse-grained type
        total_span_cnt = 0 # span correct

        query_loss_all = []

        eval_iter = min(eval_iter, len(eval_dataset))

        # print test inference
        # if ckpt is not None:
        #     if self.args.save_test_inference is not 'none':
        #         # generate save path
        #         if self.args.dataset == 'fewnerd':
        #             save_path = '_'.join([self.args.save_test_inference, self.args.dataset, self.args.mode, self.args.model, str(self.args.N), str(self.args.K)])
        #         else:
        #             save_path = '_'.join([self.args.save_test_inference, self.args.dataset, self.args.model, str(self.args.N), str(self.args.K)])
        #         f_write = open(save_path + '.txt', 'a', encoding='utf-8')


        it = 0
        while it + 1 < eval_iter:
            for _, (support, query) in enumerate(eval_dataset):
                if torch.cuda.is_available():
                    for k in support:
                        if k != 'label' and k != 'sentence_num':
                            support[k] = support[k].cuda()
                            query[k] = query[k].cuda()
                    query_label = torch.cat(query['label'], 0)
                    query_label = query_label.cuda()

                # get proto init rep
                label_data = self.__generate_label_data__(query)
                label_data_emb = model.word_encoder(label_data['word'], label_data['mask'])  # [num_label_sent, 10, 768]
                support_emb = model.word_encoder(support['word'], support['mask'])
                Proto = self.__get_proto__(label_data_emb, label_data['text_mask'], support_emb, support, model)  #[N, 768]
                
                # support,proto -> MLP -> new emb
                support_label = torch.cat(support['label'], dim=0)
                support_emb = support_emb[support['text_mask']==1]
                support_afterMLP_emb = model(support_emb)
                proto_afterMLP_emb = model(Proto)
                if self.args.use_proto_as_neg == True:
                    if self.args.have_otherO is True:
                        support_label = torch.cat((support_label, torch.tensor([0 for _ in range(self.args.N)],dtype=torch.int64)))
                    else:
                        support_label = torch.cat((support_label, torch.tensor([0 for _ in range(1,self.args.N)],dtype=torch.int64)))
                    support_afterMLP_emb = torch.cat((support_afterMLP_emb, proto_afterMLP_emb+1e-8),dim=0)
                    support_dis = []
                    if self.args.have_otherO is True:
                        for i in range(self.args.N+1):
                            support_dis_one_line = self.__pos_dist__(proto_afterMLP_emb[i].view(1, -1), support_afterMLP_emb)
                            temp_lst = [0 for _ in range(support_dis_one_line.shape[1])]
                            temp_lst[-(self.args.N+1-i)] = 1
                            temp_lst = np.array(temp_lst)
                            support_dis_one_line = support_dis_one_line.view(-1)[temp_lst == 0].view(1, -1)
                            support_dis.append(support_dis_one_line)
                    else:
                        for i in range(self.args.N):
                            support_dis_one_line = self.__pos_dist__(proto_afterMLP_emb[i].view(1, -1), support_afterMLP_emb)
                            temp_lst = [0 for _ in range(support_dis_one_line.shape[1])]
                            temp_lst[-(self.args.N-i)] = 1
                            temp_lst = np.array(temp_lst)
                            support_dis_one_line = support_dis_one_line.view(-1)[temp_lst == 0].view(1, -1)
                            support_dis.append(support_dis_one_line)
                    support_dis = torch.cat(support_dis, dim=0).view(-1)  # [N+1, N*K]
                else:
                    support_dis = self.__pos_dist__(proto_afterMLP_emb, support_afterMLP_emb).view(-1)  # [N, N*K]
                # if self.args.use_diff_threshold == False:
                # print("support_dis_before_norm:", torch.max(support_dis).item(), torch.min(support_dis).item(), torch.mean(support_dis).item())
                support_dis = F.layer_norm(support_dis, normalized_shape=[support_dis.shape[0]], bias=torch.full((support_dis.shape[0],), self.args.ln_bias).cuda())
                # print("support_dis_after_norm:", torch.max(support_dis).item(), torch.min(support_dis).item(), torch.mean(support_dis).item())
                if self.args.have_otherO is True:
                    support_dis = support_dis.view(self.args.N+1, -1)
                else:
                    support_dis = support_dis.view(self.args.N, -1)
                # if self.args.use_diff_threshold == True:
                #     margin = torch.mean(support_dis)
                support_loss = loss_func(support_dis, support_label, margin, alpha)
                
                self.__zero_grad__(model.fc.parameters())  
                grads_fc = autograd.grad(support_loss, model.fc.parameters(), allow_unused=True, retain_graph=True)
                fast_weights_fc, orderd_params_fc = model.cloned_fc_dict(), OrderedDict()
                for (key, val), grad in zip(model.fc.named_parameters(), grads_fc):
                    fast_weights_fc[key] = orderd_params_fc[key] = val - self.args.task_lr * grad  # grad中weight数量级是1e-4，bias是1e-11，有点太小了？
                
                fast_weights = {}
                fast_weights['fc'] = fast_weights_fc

                train_support_loss = []
                for _ in range(self.args.train_support_iter - 1):
                    support_afterMLP_emb = model(support_emb, fast_weights)
                    proto_afterMLP_emb = model(Proto, fast_weights)
                    if self.args.use_proto_as_neg == True:
                        # support_label = torch.cat((support_label, torch.tensor([0 for _ in range(1,self.args.N)],dtype=torch.int64)))
                        support_afterMLP_emb = torch.cat((support_afterMLP_emb, proto_afterMLP_emb+1e-8),dim=0)
                        support_dis = []
                        if self.args.have_otherO is True:
                            for i in range(self.args.N+1):
                                support_dis_one_line = self.__pos_dist__(proto_afterMLP_emb[i].view(1, -1), support_afterMLP_emb)
                                temp_lst = [0 for _ in range(support_dis_one_line.shape[1])]
                                temp_lst[-(self.args.N+1-i)] = 1
                                temp_lst = np.array(temp_lst)
                                support_dis_one_line = support_dis_one_line.view(-1)[temp_lst == 0].view(1, -1)
                                support_dis.append(support_dis_one_line)
                        else:
                            for i in range(self.args.N):
                                support_dis_one_line = self.__pos_dist__(proto_afterMLP_emb[i].view(1, -1), support_afterMLP_emb)
                                temp_lst = [0 for _ in range(support_dis_one_line.shape[1])]
                                temp_lst[-(self.args.N-i)] = 1
                                temp_lst = np.array(temp_lst)
                                support_dis_one_line = support_dis_one_line.view(-1)[temp_lst == 0].view(1, -1)
                                support_dis.append(support_dis_one_line)
                        support_dis = torch.cat(support_dis, dim=0).view(-1)
                    else:
                        support_dis = self.__pos_dist__(proto_afterMLP_emb, support_afterMLP_emb).view(-1)
                    # if self.args.use_diff_threshold == False:  
                    # print("support_dis_before_norm:", torch.max(support_dis).item(), torch.min(support_dis).item(), torch.mean(support_dis).item())
                    support_dis = F.layer_norm(support_dis, normalized_shape=[support_dis.shape[0]], bias=torch.full((support_dis.shape[0],), self.args.ln_bias).cuda())
                    # print("support_dis_after_norm:", torch.max(support_dis).item(), torch.min(support_dis).item(), torch.mean(support_dis).item())
                    if self.args.have_otherO is True:
                        support_dis = support_dis.view(self.args.N+1, -1)
                    else:
                        support_dis = support_dis.view(self.args.N, -1)
                    # if self.args.use_diff_threshold == True:
                    #     margin = torch.mean(support_dis)
                    support_loss = loss_func(support_dis, support_label, margin, alpha)
                    train_support_loss.append(support_loss.item())
                    # print_info = 'train_support, ' + str(support_loss.item())
                    # print('\033[0;31;40m{}\033[0m'.format(print_info))
                    self.__zero_grad__(orderd_params_fc.values())

                    grads_fc = torch.autograd.grad(support_loss, orderd_params_fc.values(), allow_unused=True, retain_graph=True)
                    for (key, val), grad in zip(orderd_params_fc.items(), grads_fc):
                        if grad is not None:
                            fast_weights['fc'][key] = orderd_params_fc[key] = val - self.args.task_lr * grad
                
                # query, proto -> MLP -> new emb
                query_emb = model.word_encoder(query['word'], query['mask'])
                query_emb = query_emb[query['text_mask']==1]
                query_afterMLP_emb = model(query_emb, fast_weights)
                proto_afterMLP_emb = model(Proto, fast_weights)
                if self.args.save_query_ebd is True:
                    save_qe_path = '_'.join([self.args.dataset, self.args.mode, self.args.model, str(self.args.N), str(self.args.K), str(self.args.Q), str(int(round(time.time() * 1000)))])
                    if not os.path.exists(save_qe_path):
                        os.mkdir(save_qe_path)
                    f_write = open(os.path.join(save_qe_path, 'label2tag.txt'), 'w', encoding='utf-8')
                    for ln in query['label2tag'][0]:
                        f_write.write(query['label2tag'][0][ln] + '\n')
                        f_write.flush()
                    f_write.close()
                    np.save(os.path.join(save_qe_path, 'proto.npy'), proto_afterMLP_emb.cpu().detach().numpy())
                    np.save(os.path.join(save_qe_path, '0.npy'), query_afterMLP_emb[query_label == 0].cpu().detach().numpy())
                    np.save(os.path.join(save_qe_path, '1.npy'), query_afterMLP_emb[query_label == 1].cpu().detach().numpy())
                    np.save(os.path.join(save_qe_path, '2.npy'), query_afterMLP_emb[query_label == 2].cpu().detach().numpy())
                    np.save(os.path.join(save_qe_path, '3.npy'), query_afterMLP_emb[query_label == 3].cpu().detach().numpy())
                    np.save(os.path.join(save_qe_path, '4.npy'), query_afterMLP_emb[query_label == 4].cpu().detach().numpy())
                    np.save(os.path.join(save_qe_path, '5.npy'), query_afterMLP_emb[query_label == 5].cpu().detach().numpy())
                    sys.exit()
                    
                if self.args.use_proto_as_neg == True:
                    if self.args.have_otherO is True:
                        query_label = torch.cat((query_label, torch.tensor([0 for _ in range(self.args.N)],dtype=torch.int64).cuda()))
                    else:
                        query_label = torch.cat((query_label, torch.tensor([0 for _ in range(1,self.args.N)],dtype=torch.int64).cuda()))
                    query_afterMLP_emb = torch.cat((query_afterMLP_emb,proto_afterMLP_emb+1e-8),dim=0)
                    query_dis = []
                    if self.args.have_otherO is True:
                        for i in range(self.args.N+1):
                            query_dis_one_line = self.__pos_dist__(proto_afterMLP_emb[i].view(1, -1), query_afterMLP_emb)
                            temp_lst = [0 for _ in range(query_dis_one_line.shape[1])]
                            temp_lst[-(self.args.N+1-i)] = 1
                            temp_lst = np.array(temp_lst)
                            query_dis_one_line = query_dis_one_line.view(-1)[temp_lst == 0].view(1, -1)
                            query_dis.append(query_dis_one_line)
                    else:
                        for i in range(self.args.N):
                            query_dis_one_line = self.__pos_dist__(proto_afterMLP_emb[i].view(1, -1), query_afterMLP_emb)
                            temp_lst = [0 for _ in range(query_dis_one_line.shape[1])]
                            temp_lst[-(self.args.N-i)] = 1
                            temp_lst = np.array(temp_lst)
                            query_dis_one_line = query_dis_one_line.view(-1)[temp_lst == 0].view(1, -1)
                            query_dis.append(query_dis_one_line)
                    query_dis = torch.cat(query_dis, dim=0).view(-1)
                else:
                    query_dis = self.__pos_dist__(proto_afterMLP_emb, query_afterMLP_emb).view(-1)  # [N, N*K]
                # if self.args.use_diff_threshold == False:
                # print("query_dis_before_norm:", torch.max(query_dis).item(), torch.min(query_dis).item(), torch.mean(query_dis).item())
                query_dis = F.layer_norm(query_dis, normalized_shape=[query_dis.shape[0]], bias=torch.full((query_dis.shape[0],), self.args.ln_bias).cuda())
                # print("query_dis_after_norm:", torch.max(query_dis).item(), torch.min(query_dis).item(), torch.mean(query_dis).item())
                if self.args.have_otherO is True:
                    query_dis = query_dis.view(self.args.N+1, -1)
                else:
                    query_dis = query_dis.view(self.args.N, -1)
                # if self.args.use_diff_threshold == True:
                #     margin = torch.mean(query_dis)
                query_loss = loss_func(query_dis, query_label, margin, alpha)
                query_loss_all.append(query_loss.item())

                # make prediction
                if self.args.use_proto_as_neg == True:
                    if self.args.have_otherO is True:
                        query_dis = query_dis[:, :-(self.args.N)]
                        query_label = query_label[:-(self.args.N)]
                    else:
                        query_dis = query_dis[:, :-(self.args.N-1)]
                        query_label = query_label[:-(self.args.N-1)]

                if self.args.use_diff_threshold == True:
                    threshold = []
                    if self.args.have_otherO is True:
                        for i in range(0, self.args.N+1):
                            if self.args.threshold_mode == 'mean':
                                threshold.append(torch.mean(query_dis[i][query_label==i]).item())
                            elif self.args.threshold_mode == 'max':
                                threshold.append(torch.max(query_dis[i][query_label==i]).item())
                            else:
                                raise NotImplementedError
                    else:
                        for i in range(1, self.args.N+1):
                            if self.args.threshold_mode == 'mean':
                                threshold.append(torch.mean(query_dis[i-1][query_label==i]).item())
                            elif self.args.threshold_mode == 'max':
                                threshold.append(torch.max(query_dis[i-1][query_label==i]).item())
                            else:
                                raise NotImplementedError

                    print("threshold:", threshold)
                    query_dis = query_dis.t()  # [x, N]
                    query_pred = []
                    
                    if self.args.have_otherO is True:
                        for tmp in query_dis:
                            query_pred.append(torch.min(tmp, dim=0)[1].item())
                    else:
                        for tmp in query_dis:
                            flag = []
                            for tm, th in zip(tmp, threshold):
                                if tm < th:
                                    flag.append(tm.view(1,-1))
                                else:
                                    flag.append(torch.tensor([[9999.0]]).cuda())
                            flag = torch.cat(flag).view(-1)
                            if torch.min(flag) != 9999.0:
                                query_pred.append(torch.min(tmp, dim=0)[1].item()+1)
                            else:
                                query_pred.append(0)
                    query_pred = torch.Tensor(query_pred).cuda()
                else:
                    query_dis = query_dis.t()  # [x, N]
                    query_pred = []
                    
                    if self.args.have_otherO is True:
                        for tmp in query_dis:
                            query_pred.append(torch.min(tmp, dim=0)[1].item())
                    else:
                        if self.args.multi_margin is True:
                            for tmp in query_dis:
                                flag = []
                                for tm, th in zip(tmp, margin):
                                    if tm < th:
                                        flag.append(tm.view(1,-1))
                                    else:
                                        flag.append(torch.tensor([[9999.0]]).cuda())
                                flag = torch.cat(flag).view(-1)
                                if torch.min(flag) != 9999.0:
                                    query_pred.append(torch.min(tmp, dim=0)[1].item()+1)
                                else:
                                    query_pred.append(0)
                        else:
                            for tmp in query_dis:
                                if any(t < margin for t in tmp):
                                    query_pred.append(torch.min(tmp, dim=0)[1].item()+1)
                                else:
                                    query_pred.append(0)
                    query_pred = torch.Tensor(query_pred).cuda()
                
                assert query_pred.shape[0] == query_label.shape[0]
                
                tmp_pred_cnt, tmp_label_cnt, correct = model.metrics_by_entity(query_pred, query_label)

                fp, fn, token_cnt, within, outer, total_span = model.error_analysis(query_pred, query_label, query)
                pred_cnt += tmp_pred_cnt
                label_cnt += tmp_label_cnt
                correct_cnt += correct

                fn_cnt += self.item(fn.data)
                fp_cnt += self.item(fp.data)
                total_token_cnt += token_cnt
                outer_cnt += outer
                within_cnt += within
                total_span_cnt += total_span

                # # if ckpt is not None:
                # #     if self.args.save_test_inference is not 'none':
                # #         sentence_list, real_label_list, pred_label_list, label_name_list = self.__save_test_inference__(query_logits, query_label_no_neg, query)
                # #         assert len(sentence_list) == len(real_label_list) == len(pred_label_list) == len(label_name_list)
                # #         for i in range(len(sentence_list)):
                # #             assert len(sentence_list[i]) == len(real_label_list[i]) == len(pred_label_list[i]) == len(label_name_list[i])
                # #             for j in range(len(sentence_list[i])):
                # #                 f_write.write(sentence_list[i][j] + '\t' + real_label_list[i][j] + '\t' + pred_label_list[i][j] + '\n')
                # #                 f_write.flush()
                # #             f_write.write('\n')
                # #             f_write.flush()

                if it + 1 == eval_iter:
                    break
                it += 1

        precision = correct_cnt / (pred_cnt + 1e-10)
        recall = correct_cnt / (label_cnt + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)
        fp_error = fp_cnt / total_token_cnt
        fn_error = fn_cnt / total_token_cnt
        within_error = within_cnt / total_span_cnt
        outer_error = outer_cnt / total_span_cnt
        qloss = np.mean(np.array(query_loss_all))
        print('[EVAL] step: {0:4} loss: {4:3.4f}| [ENTITY] precision: {1:3.4f}, recall: {2:3.4f}, f1: {3:3.4f}'.format(it + 1, precision, recall, f1, qloss) + '\r')

        # sys.stdout.write('[EVAL] step: {0:4} | [ENTITY] precision: {1:3.4f}, recall: {2:3.4f}, f1: {3:3.4f}'.format(it + 1, precision, recall, f1) + '\r')
        # sys.stdout.flush()
        # print("")
        # if ckpt is not None:
        #         if self.args.save_test_inference is not 'none':
        #             f_write.close()

        return precision, recall, f1, fp_error, fn_error, within_error, outer_error


class FewShotNERFramework_draw:

    def __init__(self, train_data_loader, val_data_loader, test_data_loader, args, tokenizer, use_sampled_data=False):
        '''
        train_data_loader: DataLoader for training.
        val_data_loader: DataLoader for validating.
        test_data_loader: DataLoader for testing.
        '''
        self.train_data_loader = train_data_loader
        self.val_data_loader = val_data_loader
        self.test_data_loader = test_data_loader
        self.args = args
        self.tokenizer = tokenizer
        self.use_sampled_data = use_sampled_data

    def __load_model__(self, ckpt):
        '''
        ckpt: Path of the checkpoint
        return: Checkpoint dict
        '''
        if os.path.isfile(ckpt):
            checkpoint = torch.load(ckpt)
            print("Successfully loaded checkpoint '%s'" % ckpt)
            return checkpoint
        else:
            raise Exception("No checkpoint found at '%s'" % ckpt)
    
    def item(self, x):
        '''
        PyTorch before and after 0.4
        '''
        torch_version = torch.__version__.split('.')
        if int(torch_version[0]) == 0 and int(torch_version[1]) < 4:
            return x[0]
        else:
            return x.item()

    def __generate_label_data__(self, query):
        label_tokens_index = []
        label_tokens_mask = []
        label_text_mask = []
        if self.args.have_otherO is True:
            for label_dic in query['label2tag']:
                for label_id in label_dic:
                    if label_id == 0:
                        label_tokens = ['other']
                    else:
                        label_tokens = label_dic[label_id].split('-')
                    label_tokens = ['[CLS]'] + label_tokens + ['[SEP]']
                    indexed_label_tokens = self.tokenizer.convert_tokens_to_ids(label_tokens)
                    # padding
                    while len(indexed_label_tokens) < 10:
                        indexed_label_tokens.append(0)
                    label_tokens_index.append(indexed_label_tokens)
                    # mask
                    mask = np.zeros((10), dtype=np.int32)
                    mask[:len(label_tokens)] = 1
                    label_tokens_mask.append(mask)
                    # text mask, also mask [CLS] and [SEP]
                    text_mask = np.zeros((10), dtype=np.int32)
                    text_mask[1:len(label_tokens)-1] = 1
                    label_text_mask.append(text_mask)
        else:
            for label_dic in query['label2tag']:
                for label_id in label_dic:
                    if label_id != 0:
                        label_tokens = label_dic[label_id].split('-')
                        label_tokens = ['[CLS]'] + label_tokens + ['[SEP]']
                        indexed_label_tokens = self.tokenizer.convert_tokens_to_ids(label_tokens)
                        # padding
                        while len(indexed_label_tokens) < 10:
                            indexed_label_tokens.append(0)
                        label_tokens_index.append(indexed_label_tokens)
                        # mask
                        mask = np.zeros((10), dtype=np.int32)
                        mask[:len(label_tokens)] = 1
                        label_tokens_mask.append(mask)
                        # text mask, also mask [CLS] and [SEP]
                        text_mask = np.zeros((10), dtype=np.int32)
                        text_mask[1:len(label_tokens)-1] = 1
                        label_text_mask.append(text_mask)

        label_tokens_index = torch.Tensor(label_tokens_index).long().cuda()
        label_tokens_mask = torch.Tensor(label_tokens_mask).long().cuda()
        label_text_mask = torch.Tensor(label_text_mask).long().cuda()

        label_data = {}
        label_data['word'] = label_tokens_index
        label_data['mask'] = label_tokens_mask
        label_data['text_mask'] = label_text_mask
        return label_data

    def __zero_grad__(self, params):
        for p in params:
            if p.grad is not None:
                p.grad.zero_()

    def __get_sample_pairs__(self, data):
        data_1 = {}
        data_2 = {}
        data_1['word_emb'] = data['word_emb'][[l in [*range(1, self.args.N+1)] for l in data['label']]]
        data_1['label'] = data['label'][[l in [*range(1, self.args.N+1)] for l in data['label']]]
        data_2['word_emb'] = data['word_emb'][[l in [*range(0, self.args.N+1)] for l in data['label']]]
        data_2['label'] = data['label'][[l in [*range(0, self.args.N+1)] for l in data['label']]]

        return data_1, data_2

    def __generate_pair_label__(self, label1, label2):
        pair_label = []
        for l1 in label1:
            for l2 in label2:
                if l1 == l2:
                    pair_label.append(1.0)
                else:
                    pair_label.append(0.0)
        return torch.Tensor(pair_label).cuda()

    def __generate_query_pair_label__(self, query_dis, query_label):
        query_pair_label = []
        after_query_dis = []
        for i, l in enumerate(query_label):
            tmp = torch.zeros([1, self.args.N])
            if l == -1:
                continue
            elif l == 0:
                query_pair_label.append(tmp)
                after_query_dis.append(query_dis[i])
            else:
                tmp[0, l-1] = 1
                query_pair_label.append(tmp)
                after_query_dis.append(query_dis[i])
        query_pair_label = torch.cat(query_pair_label, dim=0).view(-1).cuda()
        after_query_dis = torch.stack(after_query_dis).view(-1).cuda()

        return after_query_dis, query_pair_label
    
    def __get_proto__(self, label_data_emb, label_data_text_mask, support_emb, support, model):
        if self.args.label_name_mode == 'mean':
            temp_word_list = []
            for i, word_emb_list in enumerate(support_emb):
                temp_word_list.append(word_emb_list[support['text_mask'][i]==1])
            temp_label_list = []
            temp_word_list = torch.cat(temp_word_list)  # [x, 768]
            temp_label_list = torch.cat(support['label'], dim=0)  # [x,]
            assert temp_word_list.shape[0] == temp_label_list.shape[0]
            Proto = []
            if self.args.have_otherO is True:
                for i in range(self.args.N+1):
                    Proto.append(torch.mean(temp_word_list[temp_label_list==i], dim=0).view(1,-1))
            else:
                for i in range(self.args.N):
                    Proto.append(torch.mean(temp_word_list[temp_label_list==i+1], dim=0).view(1,-1))
            Proto = torch.cat(Proto)
        elif self.args.label_name_mode == 'LnAsQ':
            # get Q = mean(init label name)
            Q = []
            K = {}
            assert label_data_emb.shape[0] == label_data_text_mask.shape[0]
            for i, l_ebd in enumerate(label_data_emb):  # [10, 768]
                p = l_ebd[label_data_text_mask[i]==1]
                # K[i] = p
                p = p.mean(dim=0)
                Q.append(p.view(1,-1))
            Q = torch.cat(Q,0)

            # get K or V = cat(label name and word in class i)
            temp_word_list = []
            for i, word_emb_list in enumerate(support_emb):
                temp_word_list.append(word_emb_list[support['text_mask'][i]==1])
            temp_label_list = []
            temp_word_list = torch.cat(temp_word_list)  # [x, 768]
            temp_label_list = torch.cat(support['label'], dim=0)  # [x,]
            assert temp_word_list.shape[0] == temp_label_list.shape[0]
            if self.args.have_otherO is True:
                for i in range(self.args.N+1):
                    K[i] = temp_word_list[temp_label_list==i]
            else:
                for i in range(self.args.N):
                    K[i] = temp_word_list[temp_label_list==i+1]
            
            # Attention
            Proto = []
            for i, q in enumerate(Q):
                temp = torch.mm(model.att(q.view(1, -1)), K[i].t())
                att_weights = F.softmax(F.layer_norm(temp, normalized_shape=(temp.shape[0],temp.shape[1])), dim=1)
                # print("att_weights:", att_weights)
                proto = torch.mm(att_weights, K[i])  # [1, 768]
                Proto.append(proto)
            Proto = torch.cat(Proto)
        elif self.args.label_name_mode == 'LnAsQKV':
            # get Q = mean(init label name)
            Q = []
            K = {}
            assert label_data_emb.shape[0] == label_data_text_mask.shape[0]
            for i, l_ebd in enumerate(label_data_emb):  # [10, 768]
                p = l_ebd[label_data_text_mask[i]==1]
                K[i] = p
                p = p.mean(dim=0)
                Q.append(p.view(1,-1))
            Q = torch.cat(Q,0)

            # get K or V = cat(label name and word in class i)
            temp_word_list = []
            for i, word_emb_list in enumerate(support_emb):
                temp_word_list.append(word_emb_list[support['text_mask'][i]==1])
            temp_label_list = []
            temp_word_list = torch.cat(temp_word_list)  # [x, 768]
            temp_label_list = torch.cat(support['label'], dim=0)  # [x,]
            assert temp_word_list.shape[0] == temp_label_list.shape[0]

            if self.args.have_otherO is True:
                for i in range(self.args.N+1):
                    K[i] = torch.cat((K[i],temp_word_list[temp_label_list==i]),dim=0)
            else:
                for i in range(self.args.N):
                    K[i] = torch.cat((K[i],temp_word_list[temp_label_list==i+1]),dim=0)
            
            # Attention
            Proto = []
            for i, q in enumerate(Q):
                temp = torch.mm(model.att(q.view(1, -1)), K[i].t())
                att_weights = F.softmax(F.layer_norm(temp, normalized_shape=(temp.shape[0],temp.shape[1])), dim=1)
                # print("att_weights:", att_weights)
                proto = torch.mm(att_weights, K[i])  # [1, 768]
                Proto.append(proto)
            Proto = torch.cat(Proto)
        else:
            raise NotImplementedError

        return Proto

    def __pos_dist__(self, instances, class_proto):  # ins:[N*K, 256], cla:[N, 256]
        return torch.pow(torch.pow(class_proto.unsqueeze(0) - instances.unsqueeze(1), 2).sum(-1), 0.5)

    def __save_test_inference__(self, logits, pred, query):

        # query 去掉-1
        new_query_label = []
        new_query_word = []
        new_query_textmask = []
        for lq, wq, tq in zip(query['label'], query['word'], query['text_mask']):
            pass

                
        # logits = F.softmax(logits, dim=-1)
        # 将word转为真实单词
        sentence_list = []  # 二维列表
        for words, mask in zip(query['word'], query['text_mask']):
            real_words = []
            for word in words[mask==1]:
                real_words.append(self.tokenizer.decode(word).replace(" ", ""))
            sentence_list.append(real_words)

        # 将label和pred转为label word
        sentence_num = []
        sentence_num.append(0)
        tmp = 0
        for i in query['sentence_num']:
            tmp += i
            sentence_num.append(tmp)
        real_label_list = []  # 二维列表
        pred_label_list = []  # 二维列表
        label_name_list = []  # 二维列表
        # pred和logits切成二维矩阵
        pred_list = []
        # logits_list = []
        sentence_len = []
        sentence_len.append(0)
        tmp = 0
        for _labels in query['label']:
            tmp += _labels.shape[0]
            sentence_len.append(tmp)
        for i in range(len(sentence_len)-1):
            tmp2 = pred[sentence_len[i]: sentence_len[i+1]]
            # tmp3 = logits[sentence_len[i]: sentence_len[i+1]]
            pred_list.append(tmp2.cpu())
            # logits_list.append(tmp3.cpu().detach().numpy().tolist())

        for i in range(len(sentence_num)-1):
            for j in range(sentence_num[i], sentence_num[i+1]):
                tmp_label_list = []
                tmp_pred_list = []
                tmp_label_name_list = []
                assert query['label'][j].shape[0] == pred_list[j].shape[0]
                for l, p in zip(query['label'][j], pred_list[j]):
                    if l == -1:
                        tmp_label_list.append(str(-1))
                    else:
                        tmp_label_list.append(query['label2tag'][i][l.item()])
                    tmp_pred_list.append(query['label2tag'][i][p.item()])
                    tmp_label_name_list.append(str(query['label2tag'][i]))
                real_label_list.append(tmp_label_list)
                pred_label_list.append(tmp_pred_list)
                label_name_list.append(tmp_label_name_list)  # 每个元任务的label_list
        
        return sentence_list, real_label_list, pred_label_list, label_name_list

    def eval(self,
            model_proto,
            model_metnet,
            eval_iter,
            model_nn=None,
            model_struct=None,
            ckpt_proto=None,
            ckpt_metnet=None,
            ckpt_nn=None,
            ckpt_struct=None): 
        '''
        model: a FewShotREModel instance
        B: Batch size
        N: Num of classes for each batch
        K: Num of instances for each class in the support set
        Q: Num of instances for each class in the query set
        eval_iter: Num of iterations
        ckpt: Checkpoint path. Set as None if using current model parameters.
        return: Accuracy
        '''
        print("evaluating...")
        
        model_proto.eval()
        model_metnet.eval()
        # model_nn.eval()
        # model_struct.eval()

        loss_func = TripletLoss(args=self.args)
        
        if (ckpt_proto is None) or (ckpt_metnet is None):  # or (ckpt_nn is None) or (ckpt_struct is None):
            print('ckpt_proto:', ckpt_proto)
            print('ckpt_proto:', ckpt_metnet)

        print("Use test dataset")
        eval_dataset = self.test_data_loader

        pred_cnt = 0 # pred entity cnt
        label_cnt = 0 # true label entity cnt
        correct_cnt = 0 # correct predicted entity cnt

        fp_cnt = 0 # misclassify O as I-
        fn_cnt = 0 # misclassify I- as O
        total_token_cnt = 0 # total token cnt
        within_cnt = 0 # span correct but of wrong fine-grained type 
        outer_cnt = 0 # span correct but of wrong coarse-grained type
        total_span_cnt = 0 # span correct

        query_loss_all = []

        eval_iter = min(eval_iter, len(eval_dataset))

        # print test inference
        # if ckpt is not None:
        #     if self.args.save_test_inference is not 'none':
        #         # generate save path
        #         if self.args.dataset == 'fewnerd':
        #             save_path = '_'.join([self.args.save_test_inference, self.args.dataset, self.args.mode, self.args.model, str(self.args.N), str(self.args.K)])
        #         else:
        #             save_path = '_'.join([self.args.save_test_inference, self.args.dataset, self.args.model, str(self.args.N), str(self.args.K)])
        #         f_write = open(save_path + '.txt', 'a', encoding='utf-8')


        it = 0
        while it + 1 < eval_iter:
            for _, (support, query) in enumerate(eval_dataset):
                
                if ckpt_metnet != 'none':
                    state_dict = self.__load_model__(ckpt_metnet)['state_dict']
                    own_state = model_metnet.state_dict()
                    for name, param in state_dict.items():
                        if name not in own_state:
                            continue
                        own_state[name].copy_(param)
                    
                    margin = model_metnet.param
                    alpha = model_metnet.alpha
                    print("margin:", margin)
                    print("alpha:", alpha)

                    if self.args.margin != -1:
                        margin = self.args.margin
                        print('set margin:', margin)

                    if torch.cuda.is_available():
                        for k in support:
                            if k != 'label' and k != 'sentence_num':
                                support[k] = support[k].cuda()
                                query[k] = query[k].cuda()
                        query_label = torch.cat(query['label'], 0)
                        query_label = query_label.cuda()

                    # get proto init rep
                    label_data = self.__generate_label_data__(query)
                    label_data_emb = model_metnet.word_encoder(label_data['word'], label_data['mask'])  # [num_label_sent, 10, 768]
                    support_emb = model_metnet.word_encoder(support['word'], support['mask'])
                    Proto = self.__get_proto__(label_data_emb, label_data['text_mask'], support_emb, support, model_metnet)  #[N, 768]
                    
                    # support,proto -> MLP -> new emb
                    support_label = torch.cat(support['label'], dim=0)
                    support_emb = support_emb[support['text_mask']==1]
                    support_afterMLP_emb = model_metnet(support_emb)
                    proto_afterMLP_emb = model_metnet(Proto)
                    if self.args.use_proto_as_neg == True:
                        if self.args.have_otherO is True:
                            support_label = torch.cat((support_label, torch.tensor([0 for _ in range(self.args.N)],dtype=torch.int64)))
                        else:
                            support_label = torch.cat((support_label, torch.tensor([0 for _ in range(1,self.args.N)],dtype=torch.int64)))
                        support_afterMLP_emb = torch.cat((support_afterMLP_emb, proto_afterMLP_emb+1e-8),dim=0)
                        support_dis = []
                        if self.args.have_otherO is True:
                            for i in range(self.args.N+1):
                                support_dis_one_line = self.__pos_dist__(proto_afterMLP_emb[i].view(1, -1), support_afterMLP_emb)
                                temp_lst = [0 for _ in range(support_dis_one_line.shape[1])]
                                temp_lst[-(self.args.N+1-i)] = 1
                                temp_lst = np.array(temp_lst)
                                support_dis_one_line = support_dis_one_line.view(-1)[temp_lst == 0].view(1, -1)
                                support_dis.append(support_dis_one_line)
                        else:
                            for i in range(self.args.N):
                                support_dis_one_line = self.__pos_dist__(proto_afterMLP_emb[i].view(1, -1), support_afterMLP_emb)
                                temp_lst = [0 for _ in range(support_dis_one_line.shape[1])]
                                temp_lst[-(self.args.N-i)] = 1
                                temp_lst = np.array(temp_lst)
                                support_dis_one_line = support_dis_one_line.view(-1)[temp_lst == 0].view(1, -1)
                                support_dis.append(support_dis_one_line)
                        support_dis = torch.cat(support_dis, dim=0).view(-1)  # [N+1, N*K]
                    else:
                        support_dis = self.__pos_dist__(proto_afterMLP_emb, support_afterMLP_emb).view(-1)  # [N, N*K]
                    # if self.args.use_diff_threshold == False:
                    # print("support_dis_before_norm:", torch.max(support_dis).item(), torch.min(support_dis).item(), torch.mean(support_dis).item())
                    support_dis = F.layer_norm(support_dis, normalized_shape=[support_dis.shape[0]], bias=torch.full((support_dis.shape[0],), self.args.ln_bias).cuda())
                    # print("support_dis_after_norm:", torch.max(support_dis).item(), torch.min(support_dis).item(), torch.mean(support_dis).item())
                    if self.args.have_otherO is True:
                        support_dis = support_dis.view(self.args.N+1, -1)
                    else:
                        support_dis = support_dis.view(self.args.N, -1)
                    # if self.args.use_diff_threshold == True:
                    #     margin = torch.mean(support_dis)
                    support_loss = loss_func(support_dis, support_label, margin, alpha)
                    
                    self.__zero_grad__(model_metnet.fc.parameters())  
                    grads_fc = autograd.grad(support_loss, model_metnet.fc.parameters(), allow_unused=True, retain_graph=True)
                    fast_weights_fc, orderd_params_fc = model_metnet.cloned_fc_dict(), OrderedDict()
                    for (key, val), grad in zip(model_metnet.fc.named_parameters(), grads_fc):
                        fast_weights_fc[key] = orderd_params_fc[key] = val - self.args.task_lr * grad  # grad中weight数量级是1e-4，bias是1e-11，有点太小了？
                    
                    fast_weights = {}
                    fast_weights['fc'] = fast_weights_fc

                    train_support_loss = []
                    for _ in range(self.args.train_support_iter - 1):
                        support_afterMLP_emb = model_metnet(support_emb, fast_weights)
                        proto_afterMLP_emb = model_metnet(Proto, fast_weights)
                        if self.args.use_proto_as_neg == True:
                            # support_label = torch.cat((support_label, torch.tensor([0 for _ in range(1,self.args.N)],dtype=torch.int64)))
                            support_afterMLP_emb = torch.cat((support_afterMLP_emb, proto_afterMLP_emb+1e-8),dim=0)
                            support_dis = []
                            if self.args.have_otherO is True:
                                for i in range(self.args.N+1):
                                    support_dis_one_line = self.__pos_dist__(proto_afterMLP_emb[i].view(1, -1), support_afterMLP_emb)
                                    temp_lst = [0 for _ in range(support_dis_one_line.shape[1])]
                                    temp_lst[-(self.args.N+1-i)] = 1
                                    temp_lst = np.array(temp_lst)
                                    support_dis_one_line = support_dis_one_line.view(-1)[temp_lst == 0].view(1, -1)
                                    support_dis.append(support_dis_one_line)
                            else:
                                for i in range(self.args.N):
                                    support_dis_one_line = self.__pos_dist__(proto_afterMLP_emb[i].view(1, -1), support_afterMLP_emb)
                                    temp_lst = [0 for _ in range(support_dis_one_line.shape[1])]
                                    temp_lst[-(self.args.N-i)] = 1
                                    temp_lst = np.array(temp_lst)
                                    support_dis_one_line = support_dis_one_line.view(-1)[temp_lst == 0].view(1, -1)
                                    support_dis.append(support_dis_one_line)
                            support_dis = torch.cat(support_dis, dim=0).view(-1)
                        else:
                            support_dis = self.__pos_dist__(proto_afterMLP_emb, support_afterMLP_emb).view(-1)
                        # if self.args.use_diff_threshold == False:  
                        # print("support_dis_before_norm:", torch.max(support_dis).item(), torch.min(support_dis).item(), torch.mean(support_dis).item())
                        support_dis = F.layer_norm(support_dis, normalized_shape=[support_dis.shape[0]], bias=torch.full((support_dis.shape[0],), self.args.ln_bias).cuda())
                        # print("support_dis_after_norm:", torch.max(support_dis).item(), torch.min(support_dis).item(), torch.mean(support_dis).item())
                        if self.args.have_otherO is True:
                            support_dis = support_dis.view(self.args.N+1, -1)
                        else:
                            support_dis = support_dis.view(self.args.N, -1)
                        # if self.args.use_diff_threshold == True:
                        #     margin = torch.mean(support_dis)
                        support_loss = loss_func(support_dis, support_label, margin, alpha)
                        train_support_loss.append(support_loss.item())
                        # print_info = 'train_support, ' + str(support_loss.item())
                        # print('\033[0;31;40m{}\033[0m'.format(print_info))
                        self.__zero_grad__(orderd_params_fc.values())

                        grads_fc = torch.autograd.grad(support_loss, orderd_params_fc.values(), allow_unused=True, retain_graph=True)
                        for (key, val), grad in zip(orderd_params_fc.items(), grads_fc):
                            if grad is not None:
                                fast_weights['fc'][key] = orderd_params_fc[key] = val - self.args.task_lr * grad
                    
                    # query, proto -> MLP -> new emb
                    query_emb = model_metnet.word_encoder(query['word'], query['mask'])
                    query_emb = query_emb[query['text_mask']==1]
                    query_afterMLP_emb = model_metnet(query_emb, fast_weights)
                    proto_afterMLP_emb = model_metnet(Proto, fast_weights)
                    if self.args.save_query_ebd is True:
                        save_qe_path = '_'.join([self.args.dataset, self.args.mode, 'metnet',str(self.args.N), str(self.args.K), str(self.args.Q), str(int(round(time.time() * 1000)))])
                        if not os.path.exists(save_qe_path):
                            os.mkdir(save_qe_path)
                        f_write = open(os.path.join(save_qe_path, 'label2tag.txt'), 'w', encoding='utf-8')
                        for ln in query['label2tag'][0]:
                            f_write.write(query['label2tag'][0][ln] + '\n')
                            f_write.flush()
                        f_write.close()
                        np.save(os.path.join(save_qe_path, 'proto.npy'), proto_afterMLP_emb.cpu().detach().numpy())
                        np.save(os.path.join(save_qe_path, '0.npy'), query_afterMLP_emb[query_label == 0].cpu().detach().numpy())
                        np.save(os.path.join(save_qe_path, '1.npy'), query_afterMLP_emb[query_label == 1].cpu().detach().numpy())
                        np.save(os.path.join(save_qe_path, '2.npy'), query_afterMLP_emb[query_label == 2].cpu().detach().numpy())
                        np.save(os.path.join(save_qe_path, '3.npy'), query_afterMLP_emb[query_label == 3].cpu().detach().numpy())
                        np.save(os.path.join(save_qe_path, '4.npy'), query_afterMLP_emb[query_label == 4].cpu().detach().numpy())
                        np.save(os.path.join(save_qe_path, '5.npy'), query_afterMLP_emb[query_label == 5].cpu().detach().numpy())
                    
                    del model_metnet
                    del support_emb
                    del support_afterMLP_emb
                    del Proto
                    del proto_afterMLP_emb
                    del query_emb
                    del query_afterMLP_emb


                # PROTO
                if ckpt_proto != 'none':
                    state_dict = self.__load_model__(ckpt_proto)['state_dict']
                    own_state = model_proto.state_dict()
                    for name, param in state_dict.items():
                        if name not in own_state:
                            continue
                        own_state[name].copy_(param)

                    if torch.cuda.is_available():
                        for k in support:
                            if k != 'label' and k != 'sentence_num':
                                support[k] = support[k].cuda()
                                query[k] = query[k].cuda()
                        label = torch.cat(query['label'], 0)
                        label = label.cuda()
                        support_label = torch.cat(support['label'], 0).cuda()
                    
                    logits, pred = model_proto(support, query)
                    sys.exit()
                # del model_proto
                

                # # nnshot
                # if ckpt_nn != 'none':
                # state_dict = self.__load_model__(ckpt_nn)['state_dict']
                # own_state = model_nn.state_dict()
                # for name, param in state_dict.items():
                #     if name not in own_state:
                #         continue
                #     own_state[name].copy_(param)
                
                # if torch.cuda.is_available():
                #     for k in support:
                #         if k != 'label' and k != 'sentence_num':
                #             support[k] = support[k].cuda()
                #             query[k] = query[k].cuda()
                #     label = torch.cat(query['label'], 0)
                #     label = label.cuda()
                #     support_label = torch.cat(support['label'], 0).cuda()
                
                # logits, pred = model_nn(support, query)
                # sys.exit()
                
                # del model_nn


                # # structshot
                # if ckpt_struct != 'none':
                # state_dict = self.__load_model__(ckpt_struct)['state_dict']
                # own_state = model_struct.state_dict()
                # for name, param in state_dict.items():
                #     if name not in own_state:
                #         continue
                #     own_state[name].copy_(param)
                
                # if torch.cuda.is_available():
                #     for k in support:
                #         if k != 'label' and k != 'sentence_num':
                #             support[k] = support[k].cuda()
                #             query[k] = query[k].cuda()
                #     label = torch.cat(query['label'], 0)
                #     label = label.cuda()
                #     support_label = torch.cat(support['label'], 0).cuda()
                
                # logits, pred = model_struct(support, query)
                # sys.exit()
                


 