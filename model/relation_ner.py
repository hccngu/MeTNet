from cProfile import label
import sys
sys.path.append('..')
import utils
import torch
from torch import autograd, optim, nn
from torch.autograd import Variable
from torch.nn import functional as F

class RelationNER(utils.framework.FewShotNERModel):
    
    def __init__(self, word_encoder, dot=False, args=None, ignore_index=-1):
        utils.framework.FewShotNERModel.__init__(self, args, word_encoder, ignore_index=ignore_index)
        self.drop = nn.Dropout(p=args.dropout)
        self.dot = dot
        self.args = args
        self.fc = nn.Linear(768*2, 1)
        self.bn1 = nn.BatchNorm1d(768*2)
        if self.args.alpha == -1:
            self.param = nn.Parameter(torch.Tensor([0.5]))
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

    def __get_proto__(self, label_data_emb, mask):
        proto = []
        assert label_data_emb.shape[0] == mask.shape[0]
        for i, l_ebd in enumerate(label_data_emb):  # [10, 768]
            p = l_ebd[mask[i]].mean(0)
            proto.append(p.view(1,-1))
        proto = torch.cat(proto,0)
        proto = proto.view((-1,self.args.N,768))
        return proto

    def forward(self, data, label_data, model_parameters=None):
        '''
        data : support or query
        support: Inputs of the support set.
        query: Inputs of the query set.
        N: Num of classes
        K: Num of instances for each class in the support set
        Q: Num of instances in the query set
        '''

        # 1. get prototypes by label name
        label_data_emb = self.word_encoder(label_data['word'], label_data['mask'])  # [num_label_sent, 10, 768]
        proto_list = self.__get_proto__(label_data_emb, label_data['text_mask'])  # [batch_num, N, 768]

        data_emb = self.word_encoder(data['word'], data['mask']) # [num_sent, number_of_tokens, 768]
        data_emb = self.drop(data_emb)

        word_label = data['label']
        temp_sen_num = []
        temp_sent = []  # [x,768]
        temp_label = []

        temp_count = 0
        temp_sen_num.append(0)
        for num in data['sentence_num']:
            temp_count += num
            temp_sen_num.append(temp_count)

        for i, line in enumerate(data_emb):
            temp_sent.append(line[data['text_mask'][i]==1])
        # data_emb = torch.cat(temp_sent, 0)  # [num_sent*number_of_tokens, N+1]
        temp_label_list = []  # [batch_num,x]
        temp_sent_list = []  # [batch_num,x,768]
        for i in range(len(temp_sen_num)-1):
            temp_sent_list.append(temp_sent[temp_sen_num[i]: temp_sen_num[i+1]])
            temp_label_list.append(word_label[temp_sen_num[i]: temp_sen_num[i+1]])

        sample_pairs = []  # n行，每行是一个tensor，维度[768*2]
        pair_labels = []  # 0/1向量[0,0,0,1,...,0]
        for samples, protos, labels in zip(temp_sent_list, proto_list, temp_label_list):
            for _sample, _label in zip(samples, labels):
                assert _sample.shape[0] == _label.shape[0]
                for sample, label in zip(_sample, _label):
                    if label == -1:
                        continue
                    for i, proto in enumerate(protos):
                        tmp_pair = torch.cat((sample, proto), dim=0)
                        sample_pairs.append(tmp_pair.view(1,-1))
                        if label == i+1:
                            pair_labels.append(1)
                        else:
                            pair_labels.append(0)
        emb = torch.cat(sample_pairs, dim=0)
        emb = self.bn1(emb)

        if model_parameters == None:
            # emb = self.fc1(data_emb)  # [num_sent*number_of_tokens, N+1]
            # emb = F.relu(emb)
            # emb = self.fc2(emb)
            # emb = F.relu(emb)
            # emb = self.fc3(emb)
            # emb = F.relu(emb)
            emb = self.fc(emb)
            logits = torch.sigmoid(emb)
            # _, pred = torch.max(logits, 1)
        else:
            # emb = F.linear(data_emb, model_parameters['fc1']['weight'])
            # emb = F.relu(emb)
            # emb = F.linear(emb, model_parameters['fc2']['weight'])
            # emb = F.relu(emb)
            # emb = F.linear(emb, model_parameters['fc3']['weight'])
            # emb = F.relu(emb)
            emb = F.linear(emb, model_parameters['fc']['weight'])
            logits = torch.sigmoid(emb)  # [num*N, 1]
            # _, pred = torch.max(logits, 1)

        # get pred
        pred = []
        tmp_logits = logits.view(-1, self.args.N)
        for tmp in tmp_logits:
            if any(t > self.args.alpha for t in tmp):
                pred.append(torch.max(tmp,dim=0)[1].item()+1)
            else:
                pred.append(0)
        # _, pred = torch.max(logits.view(-1, self.args.N), 1)
        
        return emb, pair_labels, pred  # loss自带sigmoid

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

    



