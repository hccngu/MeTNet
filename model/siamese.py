from cProfile import label
import sys
from this import d
sys.path.append('..')
import utils
import torch
from torch import autograd, optim, nn
from torch.autograd import Variable
from torch.nn import functional as F

class Siamese(utils.framework.FewShotNERModel):
    
    def __init__(self, word_encoder, dot=False, args=None, ignore_index=-1):
        utils.framework.FewShotNERModel.__init__(self, args, word_encoder, ignore_index=ignore_index)
        self.drop = nn.Dropout(p=args.dropout)
        self.dot = dot
        self.args = args
        self.fc = nn.Linear(768, 512)
        # self.fc2 = nn.Linear(512, 128)
        # self.ln = nn.LayerNorm(768)
        # self.bn1 = nn.BatchNorm1d(768*2)
        # self.bn2 = nn.BatchNorm1d(1)
        # self.fc1 = nn.Linear(768, 512)
        # self.fc2 = nn.Linear(512, 256)
        # self.fc3 = nn.Linear(256, 128)
        # self.fc4 = nn.Linear(128, args.N+1)
        # self.loss = nn.CrossEntropyLoss()

    def __dist__(self, x, y, dim):
        if self.dot:
            return (x * y).sum(dim)
        else:
            return -(torch.pow(x - y, 2)).sum(dim)

    def __batch_dist__(self, S, Q, q_mask):
        # S [class, embed_dim], Q [num_of_sent, num_of_tokens, embed_dim]
        assert Q.size()[:2] == q_mask.size()
        Q = Q[q_mask==1].view(-1, Q.size(-1)) # [num_of_all_text_tokens, embed_dim]
        return self.__dist__(S.unsqueeze(0), Q.unsqueeze(1), 2)

    def __neg_dist__(self, instances, class_proto):  # ins:[N*K, 256], cla:[N, 256]
        return -torch.pow(torch.pow(class_proto.unsqueeze(0) - instances.unsqueeze(1), 2).sum(-1), 0.5)

    def __pos_dist__(self, instances, class_proto):  # ins:[N*K, 256], cla:[N, 256]
        return torch.pow(torch.pow(class_proto.unsqueeze(0) - instances.unsqueeze(1), 2).sum(-1), 0.5)

    def __get_proto__(self, label_data_emb, mask):
        proto = []
        assert label_data_emb.shape[0] == mask.shape[0]
        for i, l_ebd in enumerate(label_data_emb):  # [10, 768]
            p = l_ebd[mask[i]].mean(0)
            proto.append(p.view(1,-1))
        proto = torch.cat(proto,0)
        proto = proto.view((-1,self.args.N,768))
        return proto

    def __forward_once__(self, emb):
        emb = self.fc(emb)
        # emb = self.fc2(emb)
        return emb
    
    def __forward_once_with_param__(self, emb, param):
        emb = F.linear(emb, param['fc']['weight'])
        return emb

    def __get_sample_pairs__(self, data):
        data_1 = {}
        data_2 = {}
        data['word_emb'] = data['word_emb'][data['text_mask']==1]
        data['label'] = torch.cat(data['label'], dim=0)
        data_1['word_emb'] = data['word_emb'][[l in [*range(1, self.args.N+1)] for l in data['label']]]
        data_1['label'] = data['label'][[l in [*range(1, self.args.N+1)] for l in data['label']]]
        data_2['word_emb'] = data['word_emb'][[l in [*range(0, self.args.N+1)] for l in data['label']]]
        data_2['label'] = data['label'][[l in [*range(0, self.args.N+1)] for l in data['label']]]

        return data_1, data_2
    
    def forward(self, support, query, query_flag=False, margin=None, model_parameters=None):
        '''
        data : support or query
        support: Inputs of the support set.
        query: Inputs of the query set.
        N: Num of classes
        K: Num of instances for each class in the support set
        Q: Num of instances in the query set
        '''

        support_emb = self.word_encoder(support['word'], support['mask']) # [num_sent, number_of_tokens, 768]
        support_emb = self.drop(support_emb)
        support['word_emb'] = support_emb
        
        if query_flag is not True:
        
            # get sample pairs
            support_1, support_2 = self.__get_sample_pairs__(support)

            if model_parameters is None:
                out1 = self.__forward_once__(support_1['word_emb'])  # [x, 768] -> [x, 256]
                out2 = self.__forward_once__(support_2['word_emb'])
            else:
                out1 = self.__forward_once_with_param__(support_1['word_emb'], model_parameters)
                out2 = self.__forward_once_with_param__(support_2['word_emb'], model_parameters)

            # calculate distance
            dis = self.__pos_dist__(out1, out2).view(-1)
            print('out1', out1)
            print('out2', out2)
            print('support12', support_1['word_emb'], support_2['word_emb'])
            print('dis', dis)
            dis = F.layer_norm(dis, normalized_shape=[dis.shape[0]], bias=torch.full((dis.shape[0],),10.).cuda())  # weight是乘，bias是加，shape和dis相同
            print('dis_after_ln', dis)
            pair_label = self.__generate_pair_label__(support_1['label'], support_2['label'])
            
            return dis, pair_label
        else:
            query_emb = self.word_encoder(query['word'], query['mask']) # [num_sent, number_of_tokens, 768]
            query_emb = self.drop(query_emb)
            query['word_emb'] = query_emb

            support_out = self.__forward_once__(support_emb)[support['text_mask']==1]
            query_out = self.__forward_once__(query_emb)[query['text_mask']==1]
            proto = []
            assert support_out.shape[0] == support['label'].shape[0]
            for label in range(1, self.args.N+1):
                proto.append(torch.mean(support_out[support['label']==label], dim=0))
            proto = torch.stack(proto)
            query_dis = self.__pos_dist__(query_out, proto).view(-1)
            query_dis = F.layer_norm(query_dis, normalized_shape=[query_dis.shape[0]], bias=torch.full((query_dis.shape[0],),10.).cuda()).view(-1, proto.shape[0])
            query_pred = []
            for tmp in query_dis:
                if any(t < margin for t in tmp):
                    query_pred.append(torch.min(tmp, dim=0)[1].item()+1)
                else:
                    query_pred.append(0)
            return torch.Tensor(query_pred).cuda()



    def __generate_pair_label__(self, label1, label2):
        pair_label = []
        for l1 in label1:
            for l2 in label2:
                if l1 == l2:
                    pair_label.append(1.0)
                else:
                    pair_label.append(0.0)
        return torch.Tensor(pair_label).cuda()

    def cloned_fc_dict(self):
        return {key: val.clone() for key, val in self.fc.state_dict().items()}

    def cloned_fc1_dict(self):
        return {key: val.clone() for key, val in self.fc1.state_dict().items()}

    def cloned_fc2_dict(self):
        return {key: val.clone() for key, val in self.fc2.state_dict().items()}

    def cloned_fc3_dict(self):
        return {key: val.clone() for key, val in self.fc3.state_dict().items()}

    def cloned_fc4_dict(self):
        return {key: val.clone() for key, val in self.fc4.state_dict().items()}

    


class SiameseMAML(utils.framework.FewShotNERModel):
    
    def __init__(self, word_encoder, dot=False, args=None, ignore_index=-1):
        utils.framework.FewShotNERModel.__init__(self, args, word_encoder, ignore_index=ignore_index)
        self.drop = nn.Dropout(p=args.dropout)
        self.dot = dot
        self.args = args
        self.fc = nn.Linear(768, 512)
        if args.margin_num == -1:
            self.param = nn.Parameter(torch.Tensor([args.trainable_margin_init]))  # 8.5

    def __dist__(self, x, y, dim):
        if self.dot:
            return (x * y).sum(dim)
        else:
            return -(torch.pow(x - y, 2)).sum(dim)

    def __batch_dist__(self, S, Q, q_mask):
        # S [class, embed_dim], Q [num_of_sent, num_of_tokens, embed_dim]
        assert Q.size()[:2] == q_mask.size()
        Q = Q[q_mask==1].view(-1, Q.size(-1)) # [num_of_all_text_tokens, embed_dim]
        return self.__dist__(S.unsqueeze(0), Q.unsqueeze(1), 2)

    def __neg_dist__(self, instances, class_proto):  # ins:[N*K, 256], cla:[N, 256]
        return -torch.pow(torch.pow(class_proto.unsqueeze(0) - instances.unsqueeze(1), 2).sum(-1), 0.5)

    def __pos_dist__(self, instances, class_proto):  # ins:[N*K, 256], cla:[N, 256]
        return torch.pow(torch.pow(class_proto.unsqueeze(0) - instances.unsqueeze(1), 2).sum(-1), 0.5)

    def __get_proto__(self, label_data_emb, mask):
        proto = []
        assert label_data_emb.shape[0] == mask.shape[0]
        for i, l_ebd in enumerate(label_data_emb):  # [10, 768]
            p = l_ebd[mask[i]].mean(0)
            proto.append(p.view(1,-1))
        proto = torch.cat(proto,0)
        proto = proto.view((-1,self.args.N,768))
        return proto

    def __forward_once__(self, emb):
        emb = self.fc(emb)
        # emb = self.fc2(emb)
        return emb
    
    def __forward_once_with_param__(self, emb, param):
        emb = F.linear(emb, param['fc']['weight'])
        return emb

    def __get_sample_pairs__(self, data):
        data_1 = {}
        data_2 = {}
        data['word_emb'] = data['word_emb'][data['text_mask']==1]
        data['label'] = torch.cat(data['label'], dim=0)
        data_1['word_emb'] = data['word_emb'][[l in [*range(1, self.args.N+1)] for l in data['label']]]
        data_1['label'] = data['label'][[l in [*range(1, self.args.N+1)] for l in data['label']]]
        data_2['word_emb'] = data['word_emb'][[l in [*range(0, self.args.N+1)] for l in data['label']]]
        data_2['label'] = data['label'][[l in [*range(0, self.args.N+1)] for l in data['label']]]

        return data_1, data_2
    
    def __generate_query_pair_label__(query, query_dis_output):
        query_label = torch.cat(query['label'], 0)
        query_label = query_label.cuda()
        assert query_label.shape[0] == query_dis_output.shape[0]
        query_dis_output = query_dis_output[query_label!=-1]
        query_label = query_label[query_label!=-1]
        
    def forward(self, data1, data2, model_parameters=None):
        '''
        data : support or query
        support: Inputs of the support set.
        query: Inputs of the query set.
        N: Num of classes
        K: Num of instances for each class in the support set
        Q: Num of instances in the query set
        '''

        if model_parameters is None:
            out1 = self.__forward_once__(data1['word_emb'])  # [x, 768] -> [x, 256]
            out2 = self.__forward_once__(data2['word_emb'])
        else:
            out1 = self.__forward_once_with_param__(data1['word_emb'], model_parameters)
            out2 = self.__forward_once_with_param__(data2['word_emb'], model_parameters)

        # calculate distance
        dis = self.__pos_dist__(out1, out2).view(-1)
        dis = F.layer_norm(dis, normalized_shape=[dis.shape[0]], bias=torch.full((dis.shape[0],), self.args.ln_bias).cuda())  # weight是乘，bias是加，shape和dis相同
        
        return dis


        '''
        support_emb = self.word_encoder(support['word'], support['mask']) # [num_sent, number_of_tokens, 768]
        support_emb = self.drop(support_emb)
        support['word_emb'] = support_emb
        
        if query_flag is not True:
        
            # get sample pairs
            support_1, support_2 = self.__get_sample_pairs__(support)

            if model_parameters is None:
                out1 = self.__forward_once__(support_1['word_emb'])  # [x, 768] -> [x, 256]
                out2 = self.__forward_once__(support_2['word_emb'])
            else:
                out1 = self.__forward_once_with_param__(support_1['word_emb'], model_parameters)
                out2 = self.__forward_once_with_param__(support_2['word_emb'], model_parameters)

            # calculate distance
            dis = self.__pos_dist__(out1, out2).view(-1)
            # print('out1', out1)
            # print('out2', out2)
            # print('support12', support_1['word_emb'], support_2['word_emb'])
            print('dis', dis)
            dis = F.layer_norm(dis, normalized_shape=[dis.shape[0]], bias=torch.full((dis.shape[0],), self.args.ln_bias).cuda())  # weight是乘，bias是加，shape和dis相同
            print('dis_after_ln', dis)
            pair_label = self.__generate_pair_label__(support_1['label'], support_2['label'])
            
            return dis, pair_label
        else:
            query_emb = self.word_encoder(query['word'], query['mask']) # [num_sent, number_of_tokens, 768]
            query_emb = self.drop(query_emb)
            query_emb = query_emb[query['text_mask']==1]
            query['word_emb'] = query_emb

            support_out = self.__forward_once_with_param__(support_emb, model_parameters)[support['text_mask']==1]
            query_out = self.__forward_once_with_param__(query_emb, model_parameters)[query['text_mask']==1]
            proto = []
            assert support_out.shape[0] == support['label'].shape[0]
            for label in range(1, self.args.N+1):
                proto.append(torch.mean(support_out[support['label']==label], dim=0))
            proto = torch.stack(proto)
            query_dis = self.__pos_dist__(query_out, proto).view(-1)
            query_dis_output = F.layer_norm(query_dis, normalized_shape=[query_dis.shape[0]], bias=torch.full((query_dis.shape[0],), self.args.ln_bias).cuda())
            query_dis = query_dis_output.view(-1, proto.shape[0])
            query_pred = []
            for tmp in query_dis:
                if any(t < margin for t in tmp):
                    query_pred.append(torch.min(tmp, dim=0)[1].item()+1)
                else:
                    query_pred.append(0)
            query_dis_output, query_pair_label = self.__generate_query_pair_label__(query, query_dis_output)
            return torch.Tensor(query_pred).cuda(), query_dis_output, query_pair_label
            '''

    def __generate_pair_label__(self, label1, label2):
        pair_label = []
        for l1 in label1:
            for l2 in label2:
                if l1 == l2:
                    pair_label.append(1.0)
                else:
                    pair_label.append(0.0)
        return torch.Tensor(pair_label).cuda()

    def cloned_fc_dict(self):
        return {key: val.clone() for key, val in self.fc.state_dict().items()}

    def cloned_fc1_dict(self):
        return {key: val.clone() for key, val in self.fc1.state_dict().items()}

    def cloned_fc2_dict(self):
        return {key: val.clone() for key, val in self.fc2.state_dict().items()}

    def cloned_fc3_dict(self):
        return {key: val.clone() for key, val in self.fc3.state_dict().items()}

    def cloned_fc4_dict(self):
        return {key: val.clone() for key, val in self.fc4.state_dict().items()}

  
