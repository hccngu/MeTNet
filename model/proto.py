import sys
sys.path.append('..')
import utils
import numpy as np
import os
import time
import torch
from torch import autograd, optim, nn
from torch.autograd import Variable
from torch.nn import functional as F

class Proto(utils.framework.FewShotNERModel):
    
    def __init__(self, args, word_encoder, dot=False, ignore_index=-1):
        utils.framework.FewShotNERModel.__init__(self, args, word_encoder, ignore_index=ignore_index)
        self.drop = nn.Dropout()
        self.dot = dot
        self.args = args
        if self.args.mlp is True:
            self.fc = nn.Linear(768, 256)

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

    def __get_proto_for_BIO__(self, embedding, tag, mask):
        proto = []
        embedding = embedding[mask==1].view(-1, embedding.size(-1))
        tag = torch.cat(tag, 0)
        assert tag.size(0) == embedding.size(0)
        for label in range(self.args.N*2+1):
            if embedding[tag==label].shape[0] == 0:
                proto.append(proto[0])
            else:
                proto.append(torch.mean(embedding[tag==label], 0))
                # print(torch.mean(embedding[tag==label], 0))
        proto = torch.stack(proto)
        return proto

    def __get_proto__(self, embedding, tag, mask):
        proto = []
        embedding = embedding[mask==1].view(-1, embedding.size(-1))
        tag = torch.cat(tag, 0)
        assert tag.size(0) == embedding.size(0)
        for label in range(torch.max(tag)+1):
            proto.append(torch.mean(embedding[tag==label], 0))
        proto = torch.stack(proto)
        return proto

    def forward(self, support, query):
        '''
        support: Inputs of the support set.
        query: Inputs of the query set.
        N: Num of classes
        K: Num of instances for each class in the support set
        Q: Num of instances in the query set
        '''
        support_emb = self.word_encoder(support['word'], support['mask']) # [num_sent, number_of_tokens, 768]
        query_emb = self.word_encoder(query['word'], query['mask']) # [num_sent, number_of_tokens, 768]
        if (self.args.only_test is True) and (self.args.save_query_ebd is True):
            save_qe_path = '_'.join(['425', self.args.dataset, self.args.mode, 'proto', str(self.args.N), str(self.args.K), str(self.args.Q), str(int(round(time.time() * 1000)))])
            if not os.path.exists(save_qe_path):
                os.mkdir(save_qe_path)
            f_write = open(os.path.join(save_qe_path, 'label2tag.txt'), 'w', encoding='utf-8')
            for ln in query['label2tag'][0]:
                f_write.write(query['label2tag'][0][ln] + '\n')
                f_write.flush()
            f_write.close()
            se = support_emb[support['text_mask']==1].view(-1, support_emb.size(-1))
            qe = query_emb[query['text_mask']==1].view(-1, query_emb.size(-1))
            sl = torch.cat(support['label'], 0)
            ql = torch.cat(query['label'], 0)
            for i in range(self.args.N+1):
                if i == 0:
                    p = torch.mean(se[sl==i], dim=0, keepdim=True)
                else:
                    p = torch.cat((p, torch.mean(se[sl==i], dim=0, keepdim=True)), dim=0)
                np.save(os.path.join(save_qe_path, str(i)+'.npy'), qe[ql==i].cpu().detach().numpy())
            np.save(os.path.join(save_qe_path, 'proto.npy'), p.cpu().detach().numpy())
            sys.exit()

        support_emb = self.drop(support_emb)
        query_emb = self.drop(query_emb)
        if self.args.mlp is True:
            support_emb = self.fc(support_emb)
            query_emb = self.fc(query_emb)

        # Prototypical Networks
        logits = []
        current_support_num = 0
        current_query_num = 0
        assert support_emb.size()[:2] == support['mask'].size()
        assert query_emb.size()[:2] == query['mask'].size()

        if self.args.dataset_mode == 'BIO':
            get_proto = self.__get_proto_for_BIO__
        else:
            get_proto = self.__get_proto__

        for i, sent_support_num in enumerate(support['sentence_num']):
            sent_query_num = query['sentence_num'][i]
            # Calculate prototype for each class
            support_proto = get_proto(
                support_emb[current_support_num:current_support_num+sent_support_num], 
                support['label'][current_support_num:current_support_num+sent_support_num], 
                support['text_mask'][current_support_num: current_support_num+sent_support_num])
            # calculate distance to each prototype
            logits.append(self.__batch_dist__(  # logits[0]:[110, 6](110个词和6个proto的距离)
                support_proto, 
                query_emb[current_query_num:current_query_num+sent_query_num],
                query['text_mask'][current_query_num: current_query_num+sent_query_num])) # [num_of_query_tokens, class_num]
            current_query_num += sent_query_num
            current_support_num += sent_support_num
        logits = torch.cat(logits, 0)
        _, pred = torch.max(logits, 1)
        return logits, pred


class ProtoMAML(utils.framework.FewShotNERModel):
    
    def __init__(self, args, word_encoder, dot=False, ignore_index=-1):
        utils.framework.FewShotNERModel.__init__(self, args, word_encoder, ignore_index=ignore_index)
        self.drop = nn.Dropout()
        self.dot = dot
        self.args = args
        self.fc = nn.Linear(768, 256)
        self.bn = nn.LayerNorm(768)

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

    def __get_proto_for_BIO__(self, embedding, tag, mask):
        proto = []
        embedding = embedding[mask==1].view(-1, embedding.size(-1))
        tag = torch.cat(tag, 0)
        assert tag.size(0) == embedding.size(0)
        for label in range(self.args.N*2+1):
            if embedding[tag==label].shape[0] == 0:
                proto.append(proto[0])
            else:
                proto.append(torch.mean(embedding[tag==label], 0))
                # print(torch.mean(embedding[tag==label], 0))
        proto = torch.stack(proto)
        return proto

    def __get_proto__(self, embedding, tag, mask):
        proto = []
        embedding = embedding[mask==1].view(-1, embedding.size(-1))
        tag = torch.cat(tag, 0)
        assert tag.size(0) == embedding.size(0)
        for label in range(torch.max(tag)+1):
            proto.append(torch.mean(embedding[tag==label], 0))
        proto = torch.stack(proto)
        return proto

    def forward(self, data, query, query_flag=False, model_parameters=None):
        '''
        support: Inputs of the support set.
        query: Inputs of the query set.
        N: Num of classes
        K: Num of instances for each class in the support set
        Q: Num of instances in the query set
        '''
        data_emb = self.word_encoder(data['word'], data['mask']) # [num_sent, number_of_tokens, 768]
        # query_emb = self.word_encoder(query['word'], query['mask']) # [num_sent, number_of_tokens, 768]
        data_emb = self.drop(data_emb)
        # print(data_emb.shape)
        data_emb = self.bn(data_emb)
        # query_emb = self.drop(query_emb)
        if model_parameters == None:
            data_emb = self.fc(data_emb)
        else:
            data_emb = F.linear(data_emb, model_parameters['fc']['weight'])
        # query_emb = self.fc(query_emb)
        if query_flag is True:
            query_emb = self.word_encoder(query['word'], query['mask']) # [num_sent, number_of_tokens, 768]
            query_emb = self.drop(query_emb)
            if model_parameters == None:
                query_emb = self.fc(query_emb)
            else:
                query_emb = F.linear(query_emb, model_parameters['fc']['weight'])
        else:
            query = data
            query_emb = data_emb
        
        # Prototypical Networks
        logits = []
        current_support_num = 0
        current_query_num = 0
        assert data_emb.size()[:2] == data['mask'].size()
        assert query_emb.size()[:2] == query['mask'].size()

        if self.args.dataset_mode == 'BIO':
            get_proto = self.__get_proto_for_BIO__
        else:
            get_proto = self.__get_proto__

        for i, sent_support_num in enumerate(data['sentence_num']):
            sent_query_num = query['sentence_num'][i]
            # Calculate prototype for each class
            support_proto = get_proto(
                data_emb[current_support_num:current_support_num+sent_support_num], 
                data['label'][current_support_num:current_support_num+sent_support_num], 
                data['text_mask'][current_support_num: current_support_num+sent_support_num])
            # calculate distance to each prototype
            logits.append(self.__batch_dist__(  # logits[0]:[110, 6](110个词和6个proto的距离)
                support_proto, 
                query_emb[current_query_num:current_query_num+sent_query_num],
                query['text_mask'][current_query_num: current_query_num+sent_query_num])) # [num_of_query_tokens, class_num]
            current_query_num += sent_query_num
            current_support_num += sent_support_num
        logits = torch.cat(logits, 0)
        _, pred = torch.max(logits, 1)
        return logits, pred
    
    def cloned_fc_dict(self):
        return {key: val.clone() for key, val in self.fc.state_dict().items()}


class NoOtherProto(utils.framework.FewShotNERModel):
    
    def __init__(self,word_encoder, dot=False, ignore_index=-1):
        utils.framework.FewShotNERModel.__init__(self, word_encoder, ignore_index=ignore_index)
        self.drop = nn.Dropout()
        self.dot = dot

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

    def forward(self, support, query):
        '''
        support: Inputs of the support set.
        query: Inputs of the query set.
        N: Num of classes
        K: Num of instances for each class in the support set
        Q: Num of instances in the query set
        '''
        support_emb = self.word_encoder(support['word'], support['mask']) # [num_sent, number_of_tokens, 768]
        query_emb = self.word_encoder(query['word'], query['mask']) # [num_sent, number_of_tokens, 768]
        support_emb = self.drop(support_emb)
        query_emb = self.drop(query_emb)

        # Prototypical Networks
        logits = []
        current_support_num = 0
        current_query_num = 0
        assert support_emb.size()[:2] == support['mask'].size()
        assert query_emb.size()[:2] == query['mask'].size()

        for i, sent_support_num in enumerate(support['sentence_num']):
            sent_query_num = query['sentence_num'][i]
            # Calculate prototype for each class
            support_proto = self.__get_proto__(
                support_emb[current_support_num:current_support_num+sent_support_num], 
                support['label'][current_support_num:current_support_num+sent_support_num], 
                support['text_mask'][current_support_num: current_support_num+sent_support_num])
            # calculate distance to each prototype
            logits.append(self.__batch_dist__(  # logits[0]:[110, 6](110个词和6个proto的距离)
                support_proto, 
                query_emb[current_query_num:current_query_num+sent_query_num],
                query['text_mask'][current_query_num: current_query_num+sent_query_num])) # [num_of_query_tokens, class_num]
            current_query_num += sent_query_num
            current_support_num += sent_support_num
        logits = torch.cat(logits, 0)  # [x, 6]
        logits = -logits[:, 1:]/torch.mean(logits[:,1:])
        logits = F.sigmoid(logits)
        _, pred = torch.max(logits, 1)
        return logits, pred


class Proto_multiOclass(utils.framework.FewShotNERModel):
    
    def __init__(self, args, word_encoder, dot=False, ignore_index=-1):
        utils.framework.FewShotNERModel.__init__(self, args, word_encoder, ignore_index=ignore_index)
        self.drop = nn.Dropout()
        self.dot = dot
        self.args = args
        if self.args.mlp is True:
            self.fc = nn.Linear(768, 256)

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

    def __get_proto_for_BIO__(self, embedding, tag, mask):
        proto = []
        embedding = embedding[mask==1].view(-1, embedding.size(-1))
        tag = torch.cat(tag, 0)
        assert tag.size(0) == embedding.size(0)
        for label in range(self.args.N*2+1):
            if embedding[tag==label].shape[0] == 0:
                proto.append(proto[0])
            else:
                proto.append(torch.mean(embedding[tag==label], 0))
                # print(torch.mean(embedding[tag==label], 0))
        proto = torch.stack(proto)
        return proto

    def __get_proto__(self, embedding, tag, mask):
        proto = []
        embedding = embedding[mask==1].view(-1, embedding.size(-1))
        tag = torch.cat(tag, 0)
        assert tag.size(0) == embedding.size(0)
        for label in range(torch.max(tag)+1):
            proto.append(torch.mean(embedding[tag==label], 0))
        proto = torch.stack(proto)
        return proto

    def forward(self, support, query):
        '''
        support: Inputs of the support set.
        query: Inputs of the query set.
        N: Num of classes
        K: Num of instances for each class in the support set
        Q: Num of instances in the query set
        '''
        support_emb = self.word_encoder(support['word'], support['mask']) # [num_sent, number_of_tokens, 768]
        query_emb = self.word_encoder(query['word'], query['mask']) # [num_sent, number_of_tokens, 768]
        if (self.args.only_test is True) and (self.args.save_query_ebd is True):
            save_qe_path = '_'.join(['425', self.args.dataset, self.args.mode, 'proto', str(self.args.N), str(self.args.K), str(self.args.Q), str(int(round(time.time() * 1000)))])
            if not os.path.exists(save_qe_path):
                os.mkdir(save_qe_path)
            f_write = open(os.path.join(save_qe_path, 'label2tag.txt'), 'w', encoding='utf-8')
            for ln in query['label2tag'][0]:
                f_write.write(query['label2tag'][0][ln] + '\n')
                f_write.flush()
            f_write.close()
            se = support_emb[support['text_mask']==1].view(-1, support_emb.size(-1))
            qe = query_emb[query['text_mask']==1].view(-1, query_emb.size(-1))
            sl = torch.cat(support['label'], 0)
            ql = torch.cat(query['label'], 0)
            for i in range(self.args.N+1):
                if i == 0:
                    p = torch.mean(se[sl==i], dim=0, keepdim=True)
                else:
                    p = torch.cat((p, torch.mean(se[sl==i], dim=0, keepdim=True)), dim=0)
                np.save(os.path.join(save_qe_path, str(i)+'.npy'), qe[ql==i].cpu().detach().numpy())
            np.save(os.path.join(save_qe_path, 'proto.npy'), p.cpu().detach().numpy())
            sys.exit()

        support_emb = self.drop(support_emb)
        query_emb = self.drop(query_emb)
        if self.args.mlp is True:
            support_emb = self.fc(support_emb)
            query_emb = self.fc(query_emb)

        # Prototypical Networks
        logits = []
        current_support_num = 0
        current_query_num = 0
        assert support_emb.size()[:2] == support['mask'].size()
        assert query_emb.size()[:2] == query['mask'].size()

        if self.args.dataset_mode == 'BIO':
            get_proto = self.__get_proto_for_BIO__
        else:
            get_proto = self.__get_proto__

        for i, sent_support_num in enumerate(support['sentence_num']):
            sent_query_num = query['sentence_num'][i]
            # Calculate prototype for each class
            support_proto = get_proto(
                support_emb[current_support_num:current_support_num+sent_support_num], 
                support['label'][current_support_num:current_support_num+sent_support_num], 
                support['text_mask'][current_support_num: current_support_num+sent_support_num])
            # calculate distance to each prototype
            logits.append(self.__batch_dist__(  # logits[0]:[110, 6](110个词和6个proto的距离)
                support_proto, 
                query_emb[current_query_num:current_query_num+sent_query_num],
                query['text_mask'][current_query_num: current_query_num+sent_query_num])) # [num_of_query_tokens, class_num]
            current_query_num += sent_query_num
            current_support_num += sent_support_num
        logits = torch.cat(logits, 0)
        _, pred = torch.max(logits, 1)
        return logits, pred

  