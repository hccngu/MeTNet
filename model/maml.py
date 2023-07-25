import sys
sys.path.append('..')
import utils
import torch
from torch import autograd, optim, nn
from torch.autograd import Variable
from torch.nn import functional as F

class MAML(utils.framework.FewShotNERModel):
    
    def __init__(self, word_encoder, dot=False, args=None, ignore_index=-1):
        utils.framework.FewShotNERModel.__init__(self, args, word_encoder, ignore_index=ignore_index)
        self.drop = nn.Dropout()
        self.dot = dot
        self.fc = nn.Linear(768, args.N+1)
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

    def __get_proto__(self, embedding, tag, mask):
        proto = []
        embedding = embedding[mask==1].view(-1, embedding.size(-1))
        tag = torch.cat(tag, 0)
        assert tag.size(0) == embedding.size(0)
        for label in range(torch.max(tag)+1):
            proto.append(torch.mean(embedding[tag==label], 0))
        proto = torch.stack(proto)
        return proto

    def forward(self, data, model_parameters=None):
        '''
        data : support or query
        support: Inputs of the support set.
        query: Inputs of the query set.
        N: Num of classes
        K: Num of instances for each class in the support set
        Q: Num of instances in the query set
        '''
        data_emb = self.word_encoder(data['word'], data['mask']) # [num_sent, number_of_tokens, 768]
        data_emb = self.drop(data_emb)
        # data_emb = data_emb.view(-1, 768)  # [num_sent*number_of_tokens, 768]
        temp_sent = []
        for i, line in enumerate(data_emb):
            temp_sent.append(line[data['text_mask'][i]==1])
        data_emb = torch.cat(temp_sent, 0)
        if model_parameters == None:
            # emb = self.fc1(data_emb)  # [num_sent*number_of_tokens, N+1]
            # emb = F.relu(emb)
            # emb = self.fc2(emb)
            # emb = F.relu(emb)
            # emb = self.fc3(emb)
            # emb = F.relu(emb)
            # logits = self.fc4(emb)
            logits = self.fc(data_emb)
            _, pred = torch.max(logits, 1)
        else:
            # emb = F.linear(data_emb, model_parameters['fc1']['weight'])
            # emb = F.relu(emb)
            # emb = F.linear(emb, model_parameters['fc2']['weight'])
            # emb = F.relu(emb)
            # emb = F.linear(emb, model_parameters['fc3']['weight'])
            # emb = F.relu(emb)
            # logits = F.linear(emb, model_parameters['fc4']['weight'])
            logits = F.linear(data_emb, model_parameters['fc']['weight'])
            _, pred = torch.max(logits, 1)
        
        return logits, pred

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

    



