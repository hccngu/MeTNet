from cmath import nan
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
from torch import autograd, optim, nn
from torch.autograd import Variable
from torch.nn import functional as F
# from pytorch_pretrained_bert import BertAdam
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.nn.parallel import DistributedDataParallel as DDP

from .viterbi import ViterbiDecoder
from utils.contrastiveloss import ContrastiveLoss


def get_abstract_transitions(train_fname, use_sampled_data=True):
    """
    Compute abstract transitions on the training dataset for StructShot
    """
    if use_sampled_data:
        samples = data_loader.FewShotNERDataset(train_fname, None, 1).samples
        tag_lists = []
        for sample in samples:
            tag_lists += sample['support']['label'] + sample['query']['label']
    else:
        samples = data_loader.FewShotNERDatasetWithRandomSampling(train_fname, None, 1, 1, 1, 1).samples
        tag_lists = [sample.tags for sample in samples]

    s_o, s_i = 0., 0.
    o_o, o_i = 0., 0.
    i_o, i_i, x_y = 0., 0., 0.
    for tags in tag_lists:
        if tags[0] == 'O': s_o += 1
        else: s_i += 1
        for i in range(len(tags)-1):
            p, n = tags[i], tags[i+1]
            if p == 'O':
                if n == 'O': o_o += 1
                else: o_i += 1
            else:
                if n == 'O':
                    i_o += 1
                elif p != n:
                    x_y += 1
                else:
                    i_i += 1

    trans = []
    trans.append(s_o / (s_o + s_i))
    trans.append(s_i / (s_o + s_i))
    trans.append(o_o / (o_o + o_i))
    trans.append(o_i / (o_o + o_i))
    trans.append(i_o / (i_o + i_i + x_y))
    trans.append(i_i / (i_o + i_i + x_y))
    trans.append(x_y / (i_o + i_i + x_y))
    return trans

def warmup_linear(global_step, warmup_step):
    if global_step < warmup_step:
        return global_step / warmup_step
    else:
        return 1.0

class FewShotNERModel(nn.Module):
    def __init__(self, args, my_word_encoder, ignore_index=-1):
        '''
        word_encoder: Sentence encoder
        
        You need to set self.cost as your own loss function.
        '''
        nn.Module.__init__(self)
        self.args = args
        self.ignore_index = ignore_index
        self.word_encoder = nn.DataParallel(my_word_encoder)  # 用多个GPU来加速训练
        self.cost = nn.CrossEntropyLoss(ignore_index=ignore_index)  # , weight=torch.Tensor([0.1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]))

    
    def forward(self, support, query, N, K, Q):
        '''
        support: Inputs of the support set.
        query: Inputs of the query set.
        N: Num of classes
        K: Num of instances for each class in the support set
        Q: Num of instances for each class in the query set
        return: logits, pred
        '''
        raise NotImplementedError

    def loss(self, logits, label):
        '''
        logits: Logits with the size (..., class_num)
        label: Label with whatever size. 
        return: [Loss] (A single value)
        '''
        N = logits.size(-1)
        return self.cost(logits.view(-1, N), label.view(-1))
    
    def __delete_ignore_index(self, pred, label):
        if self.args.model == 'relation_ner':
            label = label[label != self.ignore_index]
        else:
            pred = pred[label != self.ignore_index]
            label = label[label != self.ignore_index]
        assert pred.shape[0] == label.shape[0]
        return pred, label

    def accuracy(self, pred, label):
        '''
        pred: Prediction results with whatever size
        label: Label with whatever size
        return: [Accuracy] (A single value)
        '''
        pred, label = self.__delete_ignore_index(pred, label)
        return torch.mean((pred.view(-1) == label.view(-1)).type(torch.FloatTensor))

    def __get_class_span_dict__(self, label, is_string=False):
        '''
        return a dictionary of each class label/tag corresponding to the entity positions in the sentence
        {label:[(start_pos, end_pos), ...]}
        '''
        class_span = {}
        current_label = None
        i = 0
        if not is_string:
            # having labels in [0, num_of_class] 
            while i < len(label):
                if label[i] > 0:
                    start = i
                    current_label = label[i]
                    i += 1
                    while i < len(label) and label[i] == current_label:
                        i += 1
                    if current_label in class_span:
                        class_span[current_label].append((start, i))
                    else:
                        class_span[current_label] = [(start, i)]
                else:
                    assert label[i] == 0
                    i += 1
        else:
            # having tags in string format ['O', 'O', 'person-xxx', ..]
            while i < len(label):
                if label[i] != 'O':
                    start = i
                    current_label = label[i]
                    i += 1
                    while i < len(label) and label[i] == current_label:
                        i += 1
                    if current_label in class_span:
                        class_span[current_label].append((start, i))
                    else:
                        class_span[current_label] = [(start, i)]
                else:
                    i += 1
        return class_span

    def __get_class_span_dict_for_BIO__(self, label, is_string=False):
        '''
        return a dictionary of each class label/tag corresponding to the entity positions in the sentence
        {label:[(start_pos, end_pos), ...]}
        '''
        class_span = {}
        current_label = None
        i = 0
        if not is_string:
            # having labels in [0, num_of_class] 
            while i < len(label):
                if label[i] > 0:
                    start = i
                    current_label = label[i]
                    i += 1
                    while i < len(label) and label[i] in [current_label, current_label+self.args.N]:
                        if current_label == label[i]:
                            break
                        i += 1
                    if current_label in class_span:
                        class_span[current_label].append((start, i))
                    else:
                        class_span[current_label] = [(start, i)]
                else:
                    assert label[i] == 0
                    i += 1
        else:
            # having tags in string format ['O', 'O', 'person-xxx', ..]
            while i < len(label):
                if label[i] != 'O':
                    start = i
                    current_label = label[i]
                    i += 1
                    while i < len(label) and label[i] == current_label:
                        i += 1
                    if current_label in class_span:
                        class_span[current_label].append((start, i))
                    else:
                        class_span[current_label] = [(start, i)]
                else:
                    i += 1
        return class_span
    

    def __get_intersect_by_entity__(self, pred_class_span, label_class_span):
        '''
        return the count of correct entity
        '''
        cnt = 0
        for label in label_class_span:
            cnt += len(list(set(label_class_span[label]).intersection(set(pred_class_span.get(label,[])))))
        return cnt

    def __get_cnt__(self, label_class_span):
        '''
        return the count of entities
        '''
        cnt = 0
        for label in label_class_span:
            cnt += len(label_class_span[label])
        return cnt

    def __transform_label_to_tag__(self, pred, query):
        '''
        flatten labels and transform them to string tags
        '''
        pred_tag = []
        label_tag = []
        current_sent_idx = 0 # record sentence index in the batch data
        current_token_idx = 0 # record token index in the batch data
        assert len(query['sentence_num']) == len(query['label2tag'])
        # iterate by each query set
        for idx, num in enumerate(query['sentence_num']):
            true_label = torch.cat(query['label'][current_sent_idx:current_sent_idx+num], 0)
            # drop ignore index
            true_label = true_label[true_label!=self.ignore_index]
            
            true_label = true_label.cpu().numpy().tolist()
            set_token_length = len(true_label)
            # use the idx-th label2tag dict
            pred_tag += [query['label2tag'][idx][label] for label in pred[current_token_idx:current_token_idx + set_token_length]]
            label_tag += [query['label2tag'][idx][label] for label in true_label]
            # update sentence and token index
            current_sent_idx += num
            current_token_idx += set_token_length
        assert len(pred_tag) == len(label_tag)
        assert len(pred_tag) == len(pred)
        return pred_tag, label_tag

    def __get_correct_span__(self, pred_span, label_span):
        '''
        return count of correct entity spans
        '''
        pred_span_list = []
        label_span_list = []
        for pred in pred_span:
            pred_span_list += pred_span[pred]
        for label in label_span:
            label_span_list += label_span[label]
        return len(list(set(pred_span_list).intersection(set(label_span_list))))

    def __get_wrong_within_span__(self, pred_span, label_span):
        '''
        return count of entities with correct span, correct coarse type but wrong finegrained type
        '''
        cnt = 0
        for label in label_span:
            coarse = label.split('-')[0]
            within_pred_span = []
            for pred in pred_span:
                if pred != label and pred.split('-')[0] == coarse:
                    within_pred_span += pred_span[pred]
            cnt += len(list(set(label_span[label]).intersection(set(within_pred_span))))
        return cnt

    def __get_wrong_outer_span__(self, pred_span, label_span):
        '''
        return count of entities with correct span but wrong coarse type
        '''
        cnt = 0
        for label in label_span:
            coarse = label.split('-')[0]
            outer_pred_span = []
            for pred in pred_span:
                if pred != label and pred.split('-')[0] != coarse:
                    outer_pred_span += pred_span[pred]
            cnt += len(list(set(label_span[label]).intersection(set(outer_pred_span))))
        return cnt

    def __get_type_error__(self, pred, label, query):
        '''
        return finegrained type error cnt, coarse type error cnt and total correct span count
        '''
        pred_tag, label_tag = self.__transform_label_to_tag__(pred, query)
        pred_span = self.__get_class_span_dict__(pred_tag, is_string=True)
        label_span = self.__get_class_span_dict__(label_tag, is_string=True)
        total_correct_span = self.__get_correct_span__(pred_span, label_span) + 1e-6
        wrong_within_span = self.__get_wrong_within_span__(pred_span, label_span)
        wrong_outer_span = self.__get_wrong_outer_span__(pred_span, label_span)
        return wrong_within_span, wrong_outer_span, total_correct_span
                
    def metrics_by_entity(self, pred, label):
        '''
        return entity level count of total prediction, true labels, and correct prediction
        '''
        pred = pred.view(-1)
        label = label.view(-1)
        pred, label = self.__delete_ignore_index(pred, label)
        pred = pred.cpu().numpy().tolist()
        label = label.cpu().numpy().tolist()
        if self.args.dataset_mode == 'BIO':
            pred_class_span = self.__get_class_span_dict_for_BIO__(pred)
            label_class_span = self.__get_class_span_dict_for_BIO__(label)
        else:
            pred_class_span = self.__get_class_span_dict__(pred)
            label_class_span = self.__get_class_span_dict__(label)
        pred_cnt = self.__get_cnt__(pred_class_span)
        label_cnt = self.__get_cnt__(label_class_span)
        correct_cnt = self.__get_intersect_by_entity__(pred_class_span, label_class_span)
        return pred_cnt, label_cnt, correct_cnt

    def error_analysis(self, pred, label, query):
        '''
        return 
        token level false positive rate and false negative rate
        entity level within error and outer error 
        '''
        pred = pred.view(-1)
        label = label.view(-1)
        pred, label = self.__delete_ignore_index(pred, label)
        fp = torch.sum(((pred > 0) & (label == 0)).type(torch.FloatTensor))
        fn = torch.sum(((pred == 0) & (label > 0)).type(torch.FloatTensor))
        pred = pred.cpu().numpy().tolist()
        label = label.cpu().numpy().tolist()
        within, outer, total_span = self.__get_type_error__(pred, label, query)
        return fp, fn, len(pred), within, outer, total_span


class FewShotNERFramework:

    def __init__(self, args, tokenizer, train_data_loader, val_data_loader, test_data_loader, viterbi=False, N=None, train_fname=None, tau=0.05, use_sampled_data=True):
        '''
        train_data_loader: DataLoader for training.
        val_data_loader: DataLoader for validating.
        test_data_loader: DataLoader for testing.
        '''
        self.args = args
        self.tokenizer = tokenizer
        self.train_data_loader = train_data_loader
        self.val_data_loader = val_data_loader
        self.test_data_loader = test_data_loader
        self.viterbi = viterbi
        if viterbi:
            abstract_transitions = get_abstract_transitions(train_fname, use_sampled_data=use_sampled_data)
            self.viterbi_decoder = ViterbiDecoder(N+2, abstract_transitions, tau)
    
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

    def train(self,
              model,
              model_name,
              learning_rate=1e-1,
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
        parameters_to_optimize = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        parameters_to_optimize = [
            {'params': [p for n, p in parameters_to_optimize 
                if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in parameters_to_optimize
                if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        if use_sgd_for_bert:
            optimizer = torch.optim.SGD(parameters_to_optimize, lr=learning_rate)
        else:
            optimizer = AdamW(parameters_to_optimize, lr=learning_rate, correct_bias=False)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=train_iter) 
        
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

                # BIO data preprocess
                if self.args.dataset_mode == 'BIO':
                    support, query = self.__get_BIO_data__(support, query)

                if self.args.model == 'proto_multiOclass':
                    for i, l_lst in enumerate(support['label']):
                        numb = l_lst.shape[0]
                        if numb <= 3:
                            continue
                        for j, l in enumerate(l_lst):
                            if l == 0:
                                if j > int(numb/3) and j < int(numb/3*2):
                                    support['label'][i][j] = self.args.N+1
                                if j >= int(numb/3*2):
                                    support['label'][i][j] = self.args.N+2

                    for i, l_lst in enumerate(query['label']):
                        numb = l_lst.shape[0]
                        if numb <= 3:
                            continue
                        for j, l in enumerate(l_lst):
                            if l == 0:
                                if j > int(numb/3) and j < int(numb/3*2):
                                    query['label'][i][j] = self.args.N+1
                                if j >= int(numb/3*2):
                                    query['label'][i][j] = self.args.N+2

                # support, query = next(self.train_data_loader)
                if torch.cuda.is_available():
                    for k in support:
                        if k != 'label' and k != 'sentence_num':
                            support[k] = support[k].cuda()
                            query[k] = query[k].cuda()
                    label = torch.cat(query['label'], 0)
                    label = label.cuda()
                    support_label = torch.cat(support['label'], 0).cuda()
                
                
                if self.args.model == 'proto_maml':
                    # MAML
                    support_logits, support_pred = model(support, query)
                    support_loss = model.loss(support_logits, support_label)
                    
                    self.__zero_grad__(model.fc.parameters())
                    
                    grads_fc = autograd.grad(support_loss, model.fc.parameters(), allow_unused=True, retain_graph=True)
                    fast_weights_fc, orderd_params_fc = model.cloned_fc_dict(), OrderedDict()
                    for (key, val), grad in zip(model.fc.named_parameters(), grads_fc):
                        fast_weights_fc[key] = orderd_params_fc[key] = val - self.args.task_lr * grad
                    
                    fast_weights = {}
                    fast_weights['fc'] = fast_weights_fc
                    
                    train_support_acc = []
                    train_support_loss = []
                    # if (it+1) % 50 == 0:
                    #     self.args.task_lr /= 2
                    for i in range(self.args.train_support_iter - 1):
                        support_logits, support_pred = model(support, query, model_parameters=fast_weights)
                        support_loss = model.loss(support_logits, support_label)

                        support_acc = ((support_pred == support_label).float()).sum().item() / support_pred.shape[0]
                        train_support_acc.append(support_acc)
                        train_support_loss.append(support_loss.item())

                        self.__zero_grad__(orderd_params_fc.values())

                        grads_fc = torch.autograd.grad(support_loss, orderd_params_fc.values(), allow_unused=True, retain_graph=False)
                        for (key, val), grad in zip(orderd_params_fc.items(), grads_fc):
                            if grad is not None:
                                fast_weights['fc'][key] = orderd_params_fc[key] = val - self.args.task_lr * grad
                    logits, pred = model(support, query, query_flag=True, model_parameters=fast_weights)
                    # query_loss = model.loss(logits, label)
                else:
                    logits, pred = model(support, query)
                    
                assert logits.shape[0] == label.shape[0], print(logits.shape, label.shape)
                loss = model.loss(logits, label) / float(grad_iter)
                # print(loss)

                if fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                if it % grad_iter == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                if self.args.model == 'proto_multiOclass':
                    for i, pr in enumerate(pred):
                        if (pr == self.args.N+1) or (pr == self.args.N+2):
                            pred[i] = 0
                    for i, la in enumerate(label):
                        if (la == self.args.N+1) or (la == self.args.N+2):
                            label[i] = 0

                tmp_pred_cnt, tmp_label_cnt, correct = model.metrics_by_entity(pred, label)
                iter_loss += self.item(loss.data)
                #iter_right += self.item(right.data)
                pred_cnt += tmp_pred_cnt
                label_cnt += tmp_label_cnt
                correct_cnt += correct
                iter_sample += 1
                if (it + 1) % 100 == 0 or (it + 1) % val_step == 0:
                    precision = correct_cnt / pred_cnt
                    recall = correct_cnt / label_cnt
                    f1 = 2 * precision * recall / (precision + recall + 1e-10)
                    sys.stdout.write('step: {0:4} | loss: {1:2.6f} | [ENTITY] precision: {2:3.4f}, recall: {3:3.4f}, f1: {4:3.4f}'\
                        .format(it + 1, iter_loss/ iter_sample, precision, recall, f1) + '\r')
                sys.stdout.flush()

                if (it + 1) % val_step == 0:
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
    
    
    def __get_BIO_data__(self, support, query):
        # change label index
        for _ in [support, query]:
            for label_list in _['label']:
                for i, l in enumerate(label_list):
                    if l == -1:
                        if label_list[i-1] == 0:
                            label_list[i] = 0
                        elif label_list[i-1] in [*range(1, self.args.N+1)]:
                            label_list[i] = label_list[i-1] + self.args.N
                        else:
                            label_list[i] = label_list[i-1]
        
        # change label2tag
        for label2tag in query['label2tag']:
            for key in [*range(1, self.args.N+1)]:
                label2tag[key + self.args.N] = 'I-' + label2tag[key]
                label2tag[key] = 'B-' + label2tag[key]

        return support, query

    
    def __zero_grad__(self, params):
        for p in params:
            if p.grad is not None:
                p.grad.zero_()
    
    def __save_test_inference__(self, logits, pred, query):
        logits = F.softmax(logits, dim=-1)
        # 将word转为真实单词
        sentence_list = []  # 二维列表
        for words, mask in zip(query['word'], query['text_mask']):
            real_words = []
            for word in words[mask==1]:
                real_words.append(self.tokenizer.decode(word))
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
        logits_list = []
        sentence_len = []
        sentence_len.append(0)
        tmp = 0
        for _labels in query['label']:
            tmp += _labels.shape[0]
            sentence_len.append(tmp)
        for i in range(len(sentence_len)-1):
            tmp2 = pred[sentence_len[i]: sentence_len[i+1]]
            tmp3 = logits[sentence_len[i]: sentence_len[i+1]]
            pred_list.append(tmp2.cpu())
            logits_list.append(tmp3.cpu().detach().numpy().tolist())

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

    def __get_emmissions__(self, logits, tags_list):
        # split [num_of_query_tokens, num_class] into [[num_of_token_in_sent, num_class], ...]
        emmissions = []
        current_idx = 0
        for tags in tags_list:
            emmissions.append(logits[current_idx:current_idx+len(tags)])
            current_idx += len(tags)
        assert current_idx == logits.size()[0]
        return emmissions

    def viterbi_decode(self, logits, query_tags):
        emissions_list = self.__get_emmissions__(logits, query_tags)
        pred = []
        for i in range(len(query_tags)):
            sent_scores = emissions_list[i].cpu()
            sent_len, n_label = sent_scores.shape
            sent_probs = F.softmax(sent_scores, dim=1)
            start_probs = torch.zeros(sent_len) + 1e-6
            sent_probs = torch.cat((start_probs.view(sent_len, 1), sent_probs), 1)
            feats = self.viterbi_decoder.forward(torch.log(sent_probs).view(1, sent_len, n_label+1))
            vit_labels = self.viterbi_decoder.viterbi(feats)
            vit_labels = vit_labels.view(sent_len)
            vit_labels = vit_labels.detach().cpu().numpy().tolist()
            for label in vit_labels:
                pred.append(label-1)
        return torch.tensor(pred).cuda()

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
        print("")
        
        model.eval()
        if ckpt is None:
            print("Use val dataset")
            eval_dataset = self.val_data_loader
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

        pred_cnt = 0 # pred entity cnt
        label_cnt = 0 # true label entity cnt
        correct_cnt = 0 # correct predicted entity cnt

        fp_cnt = 0 # misclassify O as I-
        fn_cnt = 0 # misclassify I- as O
        total_token_cnt = 0 # total token cnt
        within_cnt = 0 # span correct but of wrong fine-grained type 
        outer_cnt = 0 # span correct but of wrong coarse-grained type
        total_span_cnt = 0 # span correct

        eval_iter = min(eval_iter, len(eval_dataset))
        
        it = 0

        if ckpt is not None:
            if self.args.save_test_inference is not 'none':
                # generate save path
                if self.args.dataset == 'fewnerd':
                    save_path = self.args.save_test_inference + '_' + self.args.dataset + '_' + self.args.mode + '_' + self.args.model + '_' + str(self.args.N) + '_' + str(self.args.K)
                else:
                    save_path = self.args.save_test_inference + '_' + self.args.dataset + '_' + self.args.model + '_' + str(self.args.N) + '_' + str(self.args.K)
                f_write = open(save_path + '.txt', 'a', encoding='utf-8')

        # for print test inference
        logits_list = []
        pred_list = []
        # label_list = []
        while it + 1 < eval_iter:
            for _, (support, query) in enumerate(eval_dataset):
                # print("begining...")
                # BIO data preprocess
                if self.args.dataset_mode == 'BIO':
                    support, query = self.__get_BIO_data__(support, query)

                if self.args.model == 'proto_multiOclass':
                    for i, l_lst in enumerate(support['label']):
                        numb = l_lst.shape[0]
                        if numb <= 3:
                            continue
                        for j, l in enumerate(l_lst):
                            if l == 0:
                                if j > int(numb/3) and j < int(numb/3*2):
                                    support['label'][i][j] = self.args.N+1
                                if j >= int(numb/3*2):
                                    support['label'][i][j] = self.args.N+2

                    for i, l_lst in enumerate(query['label']):
                        numb = l_lst.shape[0]
                        if numb <= 3:
                            continue
                        for j, l in enumerate(l_lst):
                            if l == 0:
                                if j > int(numb/3) and j < int(numb/3*2):
                                    query['label'][i][j] = self.args.N+1
                                if j >= int(numb/3*2):
                                    query['label'][i][j] = self.args.N+2

                if torch.cuda.is_available():
                    for k in support:
                        if k != 'label' and k != 'sentence_num':
                            support[k] = support[k].cuda()
                            query[k] = query[k].cuda()
                    label = torch.cat(query['label'], 0)
                    label = label.cuda()
                    support_label = torch.cat(support['label'], 0).cuda()
                
                if self.args.model == 'proto_maml':
                    # MAML
                    support_logits, support_pred = model(support, query)
                    support_loss = model.loss(support_logits, support_label)
                    
                    self.__zero_grad__(model.fc.parameters())
                    
                    grads_fc = autograd.grad(support_loss, model.fc.parameters(), allow_unused=True, retain_graph=True)
                    fast_weights_fc, orderd_params_fc = model.cloned_fc_dict(), OrderedDict()
                    for (key, val), grad in zip(model.fc.named_parameters(), grads_fc):
                        fast_weights_fc[key] = orderd_params_fc[key] = val - self.args.task_lr * grad
                    
                    fast_weights = {}
                    fast_weights['fc'] = fast_weights_fc
                    
                    train_support_acc = []
                    train_support_loss = []
                    # if (it+1) % 50 == 0:
                    #     self.args.task_lr /= 2
                    for i in range(self.args.train_support_iter - 1):
                        support_logits, support_pred = model(support, query, model_parameters=fast_weights)
                        support_loss = model.loss(support_logits, support_label)
                        support_acc = ((support_pred == support_label).float()).sum().item() / support_pred.shape[0]
                        train_support_acc.append(support_acc)
                        train_support_loss.append(support_loss.item())
                        self.__zero_grad__(orderd_params_fc.values())
                        grads_fc = torch.autograd.grad(support_loss, orderd_params_fc.values(), allow_unused=True, retain_graph=False)
                        for (key, val), grad in zip(orderd_params_fc.items(), grads_fc):
                            if grad is not None:
                                fast_weights['fc'][key] = orderd_params_fc[key] = val - self.args.task_lr * grad
                    logits, pred = model(support, query, query_flag=True, model_parameters=fast_weights)
                    # query_loss = model.loss(logits, label)
                else:
                    logits, pred = model(support, query)
                if self.viterbi:
                    pred = self.viterbi_decode(logits, query['label'])

                if self.args.model == 'proto_multiOclass':
                    for i, pr in enumerate(pred):
                        if (pr == self.args.N+1) or (pr == self.args.N+2):
                            pred[i] = 0
                    for i, la in enumerate(label):
                        if (la == self.args.N+1) or (la == self.args.N+2):
                            label[i] = 0
                    for i, l_lst in enumerate(query['label']):
                        for j, l in enumerate(l_lst):
                            if (l == self.args.N+1) or (l == self.args.N+2):
                                query['label'][i][j] = 0

                tmp_pred_cnt, tmp_label_cnt, correct = model.metrics_by_entity(pred, label)
                fp, fn, token_cnt, within, outer, total_span = model.error_analysis(pred, label, query)
                pred_cnt += tmp_pred_cnt
                label_cnt += tmp_label_cnt
                correct_cnt += correct

                fn_cnt += self.item(fn.data)
                fp_cnt += self.item(fp.data)
                total_token_cnt += token_cnt
                outer_cnt += outer
                within_cnt += within
                total_span_cnt += total_span

                # for print test inference
                # save test inference file
                if ckpt is not None:
                    if self.args.save_test_inference is not 'none':
                        sentence_list, real_label_list, pred_label_list, label_name_list = self.__save_test_inference__(logits, pred, query)
                        assert len(sentence_list) == len(real_label_list) == len(pred_label_list) == len(label_name_list)
                        for i in range(len(sentence_list)):
                            assert len(sentence_list[i]) == len(real_label_list[i]) == len(pred_label_list[i]) == len(label_name_list[i])
                            for j in range(len(sentence_list[i])):
                                f_write.write(sentence_list[i][j] + '\t' + real_label_list[i][j] + '\t' + pred_label_list[i][j] + '\n')
                                f_write.flush()
                            f_write.write('\n')
                            f_write.flush()


                # if ckpt is not None:
                #     if self.args.save_test_inference is not 'none':
                #         logits_list.append(logits.cpu().numpy().tolist())
                #         pred_list.append(pred.cpu().numpy().tolist())
                        # label_list.append(query['label2tag'])

                if it + 1 == eval_iter:
                    break
                it += 1

        precision = correct_cnt / pred_cnt
        recall = correct_cnt /label_cnt
        f1 = 2 * precision * recall / (precision + recall + 1e-10)
        fp_error = fp_cnt / total_token_cnt
        fn_error = fn_cnt / total_token_cnt
        within_error = within_cnt / total_span_cnt
        outer_error = outer_cnt / total_span_cnt
        sys.stdout.write('[EVAL] step: {0:4} | [ENTITY] precision: {1:3.4f}, recall: {2:3.4f}, f1: {3:3.4f}'.format(it + 1, precision, recall, f1) + '\r')
        sys.stdout.flush()
        print("")

        if ckpt is not None:
            if self.args.save_test_inference is not 'none':
                f_write.close()
                    

        return precision, recall, f1, fp_error, fn_error, within_error, outer_error


class FewShotNERFramework_MAML:

    def __init__(self, tokenizer, train_data_loader, val_data_loader, test_data_loader, args, viterbi=False, N=None, train_fname=None, tau=0.05, use_sampled_data=True):
        '''
        train_data_loader: DataLoader for training.
        val_data_loader: DataLoader for validating.
        test_data_loader: DataLoader for testing.
        '''
        self.tokenizer = tokenizer
        self.train_data_loader = train_data_loader
        self.val_data_loader = val_data_loader
        self.test_data_loader = test_data_loader
        self.args = args
        self.viterbi = viterbi
        if viterbi:
            abstract_transitions = get_abstract_transitions(train_fname, use_sampled_data=use_sampled_data)
            self.viterbi_decoder = ViterbiDecoder(N+2, abstract_transitions, tau)
    
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

    def train(self,
              model,
              model_name,
              learning_rate=1e-1,
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
    
        # # Init optimizer
        # print('Use bert optim!')
        # parameters_to_optimize = list(model.named_parameters())
        # no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        # parameters_to_optimize = [
        #     {'params': [p for n, p in parameters_to_optimize 
        #         if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        #     {'params': [p for n, p in parameters_to_optimize
        #         if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        # ]
        # if use_sgd_for_bert:
        #     optimizer = torch.optim.SGD(parameters_to_optimize, lr=self.args.meta_lr)
        # else:
        #     optimizer = AdamW(parameters_to_optimize, lr=self.args.meta_lr, correct_bias=False)
        # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=train_iter) 
        

        # optimizer = torch.optim.Adam(model.fc.parameters(), lr=self.args.meta_lr)

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

        wd_param = {}
        wo_wd_param = {}
        wd_p = []
        wo_wd_p = []
        for n, p in parameters_to_optimize:
            if not any(wo in n for wo in without):
                if not any(nd in n for nd in no_decay):
                    wd_p.append(p)
                else:
                    wo_wd_p.append(p)
        wd_param['params'] = wd_p
        wd_param['weight_decay'] = self.args.wobert_wd
        wo_wd_param['params'] = wo_wd_p
        wo_wd_param['weight_decay'] = 0.0
        parameters_to_optimize = []
        parameters_to_optimize.append(wd_param)
        parameters_to_optimize.append(wo_wd_param)


        # parameters_to_optimize = [
        #     {'params': [p for n, p in parameters_to_optimize 
        #         if not (any(nd in n for nd in no_decay) or any(wo in n for wo in without))], 'weight_decay': 0.01},
        #     {'params': [p for n, p in parameters_to_optimize
        #         if any(nd in n for nd in no_decay) and not any(wo in n for wo in without)], 'weight_decay': 0.0}
        # ]
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
                # support, query = next(self.train_data_loader)
                if torch.cuda.is_available():
                    for k in support:
                        if k != 'label' and k != 'sentence_num':
                            support[k] = support[k].cuda()
                            query[k] = query[k].cuda()
                    query_label = torch.cat(query['label'], 0)
                    query_label = query_label.cuda()


                # MAML
                support_label = torch.cat(support['label'], 0).cuda()

                if self.args.use_class_weights:
                    class_weight = []
                    count_class_no_sample = 0
                    for i in range(self.args.N+1):
                        if support_label[support_label==i].shape[0] == 0:
                            count_class_no_sample += 1
                            class_weight.append(1)
                        else:
                            class_weight.append(1 / support_label[support_label==i].shape[0])
                    class_weight = torch.Tensor(class_weight).cuda()
                    # print("count_class_no_sample:", count_class_no_sample)
                    loss_fun = nn.CrossEntropyLoss(ignore_index=self.args.ignore_index, weight=class_weight)

                support_logits, support_pred = model(support)
                if self.args.use_class_weights == True:
                    support_loss = loss_fun(support_logits, support_label)
                else:
                    support_loss = model.loss(support_logits, support_label)
                # print_info = 'train_support, ' + str(support_loss.item())
                # print('\033[0;31;40m{}\033[0m'.format(print_info))
                self.__zero_grad__(model.fc.parameters())
                # self.__zero_grad__(model.fc1.parameters())
                # self.__zero_grad__(model.fc2.parameters())
                # self.__zero_grad__(model.fc3.parameters())
                # self.__zero_grad__(model.fc4.parameters())

                grads_fc = autograd.grad(support_loss, model.fc.parameters(), allow_unused=True, retain_graph=True)
                fast_weights_fc, orderd_params_fc = model.cloned_fc_dict(), OrderedDict()
                for (key, val), grad in zip(model.fc.named_parameters(), grads_fc):
                    fast_weights_fc[key] = orderd_params_fc[key] = val - self.args.task_lr * grad
                # grads_fc1 = autograd.grad(support_loss, model.fc1.parameters(), allow_unused=True, retain_graph=True)
                # fast_weights_fc1, orderd_params_fc1 = model.cloned_fc1_dict(), OrderedDict()
                # for (key, val), grad in zip(model.fc1.named_parameters(), grads_fc1):
                #     fast_weights_fc1[key] = orderd_params_fc1[key] = val - self.args.task_lr * grad

                # grads_fc2 = autograd.grad(support_loss, model.fc2.parameters(), allow_unused=True, retain_graph=True)
                # fast_weights_fc2, orderd_params_fc2 = model.cloned_fc2_dict(), OrderedDict()
                # for (key, val), grad in zip(model.fc2.named_parameters(), grads_fc2):
                #     fast_weights_fc2[key] = orderd_params_fc2[key] = val - self.args.task_lr * grad

                # grads_fc3 = autograd.grad(support_loss, model.fc3.parameters(), allow_unused=True, retain_graph=True)
                # fast_weights_fc3, orderd_params_fc3 = model.cloned_fc3_dict(), OrderedDict()
                # for (key, val), grad in zip(model.fc3.named_parameters(), grads_fc3):
                #     fast_weights_fc3[key] = orderd_params_fc3[key] = val - self.args.task_lr * grad

                # grads_fc4 = autograd.grad(support_loss, model.fc4.parameters(), allow_unused=True, retain_graph=True)
                # fast_weights_fc4, orderd_params_fc4 = model.cloned_fc4_dict(), OrderedDict()
                # for (key, val), grad in zip(model.fc4.named_parameters(), grads_fc4):
                #     fast_weights_fc4[key] = orderd_params_fc4[key] = val - self.args.task_lr * grad

                fast_weights = {}
                fast_weights['fc'] = fast_weights_fc
                # fast_weights['fc1'] = fast_weights_fc1
                # fast_weights['fc2'] = fast_weights_fc2
                # fast_weights['fc3'] = fast_weights_fc3
                # fast_weights['fc4'] = fast_weights_fc4

                train_support_acc = []
                train_support_loss = []
                # if (it+1) % 50 == 0:
                #     self.args.task_lr /= 2
                for i in range(self.args.train_support_iter - 1):
                    support_logits, support_pred = model(support, fast_weights)
                    if self.args.use_class_weights == True:
                        support_loss = loss_fun(support_logits, support_label)
                    else:
                        support_loss = model.loss(support_logits, support_label)
                    # support_loss = model.loss(support_logits, support_label)
                    support_acc = ((support_pred == support_label).float()).sum().item() / support_pred.shape[0]
                    train_support_acc.append(support_acc)
                    train_support_loss.append(support_loss.item())
                    # print_info = 'train_support, ' + str(support_loss.item())
                    # print('\033[0;31;40m{}\033[0m'.format(print_info))
                    self.__zero_grad__(orderd_params_fc.values())
                    # self.__zero_grad__(model.fc1.parameters())
                    # self.__zero_grad__(model.fc2.parameters())
                    # self.__zero_grad__(model.fc3.parameters())
                    # self.__zero_grad__(model.fc4.parameters())

                    grads_fc = torch.autograd.grad(support_loss, orderd_params_fc.values(), allow_unused=True, retain_graph=False)
                    for (key, val), grad in zip(orderd_params_fc.items(), grads_fc):
                        if grad is not None:
                            fast_weights['fc'][key] = orderd_params_fc[key] = val - self.args.task_lr * grad
                    
                    # grads_fc1 = torch.autograd.grad(support_loss, orderd_params_fc1.values(), allow_unused=True, retain_graph=True)
                    # for (key, val), grad in zip(orderd_params_fc1.items(), grads_fc1):
                    #     if grad is not None:
                    #         fast_weights['fc1'][key] = orderd_params_fc1[key] = val - self.args.task_lr * grad
                    
                    # grads_fc2 = torch.autograd.grad(support_loss, orderd_params_fc2.values(), allow_unused=True, retain_graph=True)
                    # for (key, val), grad in zip(orderd_params_fc2.items(), grads_fc2):
                    #     if grad is not None:
                    #         fast_weights['fc2'][key] = orderd_params_fc2[key] = val - self.args.task_lr * grad

                    # grads_fc3 = torch.autograd.grad(support_loss, orderd_params_fc3.values(), allow_unused=True, retain_graph=True)
                    # for (key, val), grad in zip(orderd_params_fc3.items(), grads_fc3):
                    #     if grad is not None:
                    #         fast_weights['fc3'][key] = orderd_params_fc3[key] = val - self.args.task_lr * grad

                    # grads_fc4 = torch.autograd.grad(support_loss, orderd_params_fc4.values(), allow_unused=True)
                    # for (key, val), grad in zip(orderd_params_fc4.items(), grads_fc4):
                    #     if grad is not None:
                    #         fast_weights['fc4'][key] = orderd_params_fc4[key] = val - self.args.task_lr * grad
                
                query_logits, query_pred = model(query, fast_weights)
                if self.args.use_class_weights == True:
                    query_loss = loss_fun(query_logits, query_label)
                else:
                    query_loss = model.loss(query_logits, query_label)
                # print_info = 'train_query, ' + str(query_loss.item())
                # print('\033[0;32;40m{}\033[0m'.format(print_info))
                # query_acc = ((query_pred == query_label).float()).sum().item() / query_pred.shape[0]

                tmp_pred_cnt, tmp_label_cnt, correct = model.metrics_by_entity(query_pred, query_label)

                bert_optimizer.zero_grad()
                wobert_optimizer.zero_grad()
                # optimizer.zero_grad()
                    
                if fp16:
                    with amp.scale_loss(query_loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    query_loss.backward()
                
                if it % grad_iter == 0:
                    bert_optimizer.step()
                    wobert_optimizer.step()
                    bert_scheduler.step()
                    wobert_scheduler.step()

                iter_loss += self.item(query_loss.data)
                #iter_right += self.item(right.data)
                pred_cnt += tmp_pred_cnt
                label_cnt += tmp_label_cnt
                correct_cnt += correct
                iter_sample += 1
                if (it + 1) % 100 == 0 or (it + 1) % val_step == 0:
                    precision = correct_cnt / pred_cnt
                    recall = correct_cnt / label_cnt
                    f1 = 2 * precision * recall / (precision + recall + 1e-10)
                    print('step: {0:4} | loss: {1:2.6f} | [ENTITY] precision: {2:3.4f}, recall: {3:3.4f}, f1: {4:3.4f}'\
                        .format(it + 1, iter_loss/ iter_sample, precision, recall, f1) + '\r')
                    # iter_loss = 0.
                    # iter_sample = 0.
                    # pred_cnt = 0
                    # label_cnt = 0
                    # correct_cnt = 0
                    # sys.stdout.write('step: {0:4} | loss: {1:2.6f} | [ENTITY] precision: {2:3.4f}, recall: {3:3.4f}, f1: {4:3.4f}'\
                    #     .format(it + 1, iter_loss/ iter_sample, precision, recall, f1) + '\r')
                # sys.stdout.flush()

                if (it + 1) % val_step == 0:
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
        logits = F.softmax(logits, dim=-1)
        # 将word转为真实单词
        sentence_list = []  # 二维列表
        for words, mask in zip(query['word'], query['text_mask']):
            real_words = []
            for word in words[mask==1]:
                real_words.append(self.tokenizer.decode(word))
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
        logits_list = []
        sentence_len = []
        sentence_len.append(0)
        tmp = 0
        for _labels in query['label']:
            tmp += _labels.shape[0]
            sentence_len.append(tmp)
        for i in range(len(sentence_len)-1):
            tmp2 = pred[sentence_len[i]: sentence_len[i+1]]
            tmp3 = logits[sentence_len[i]: sentence_len[i+1]]
            pred_list.append(tmp2.cpu())
            logits_list.append(tmp3.cpu().detach().numpy().tolist())

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
    
    def __zero_grad__(self, params):
        for p in params:
            if p.grad is not None:
                p.grad.zero_()
    
    def __get_emmissions__(self, logits, tags_list):
        # split [num_of_query_tokens, num_class] into [[num_of_token_in_sent, num_class], ...]
        emmissions = []
        current_idx = 0
        for tags in tags_list:
            emmissions.append(logits[current_idx:current_idx+len(tags)])
            current_idx += len(tags)
        assert current_idx == logits.size()[0]
        return emmissions

    def viterbi_decode(self, logits, query_tags):
        emissions_list = self.__get_emmissions__(logits, query_tags)
        pred = []
        for i in range(len(query_tags)):
            sent_scores = emissions_list[i].cpu()
            sent_len, n_label = sent_scores.shape
            sent_probs = F.softmax(sent_scores, dim=1)
            start_probs = torch.zeros(sent_len) + 1e-6
            sent_probs = torch.cat((start_probs.view(sent_len, 1), sent_probs), 1)
            feats = self.viterbi_decoder.forward(torch.log(sent_probs).view(1, sent_len, n_label+1))
            vit_labels = self.viterbi_decoder.viterbi(feats)
            vit_labels = vit_labels.view(sent_len)
            vit_labels = vit_labels.detach().cpu().numpy().tolist()
            for label in vit_labels:
                pred.append(label-1)
        return torch.tensor(pred).cuda()

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
        print("")
        
        model.eval()
        if ckpt is None:
            print("Use val dataset")
            eval_dataset = self.val_data_loader
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

        pred_cnt = 0 # pred entity cnt
        label_cnt = 0 # true label entity cnt
        correct_cnt = 0 # correct predicted entity cnt

        fp_cnt = 0 # misclassify O as I-
        fn_cnt = 0 # misclassify I- as O
        total_token_cnt = 0 # total token cnt
        within_cnt = 0 # span correct but of wrong fine-grained type 
        outer_cnt = 0 # span correct but of wrong coarse-grained type
        total_span_cnt = 0 # span correct

        eval_iter = min(eval_iter, len(eval_dataset))

        # print test inference
        # if ckpt is not None:
        #     if self.args.save_test_inference is not 'none':
        #         # generate save path
        #         if self.args.dataset == 'fewnerd':
        #             save_path = '_'.join(self.args.save_test_inference, self.args.dataset, self.args.mode, self.args.model, str(self.args.N), str(self.args.K))
        #         else:
        #             save_path = '_'.join(self.args.save_test_inference, self.args.dataset, self.args.model, str(self.args.N), str(self.args.K))
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
                # logits, pred = model(support, query)
                # if self.viterbi:
                #     pred = self.viterbi_decode(logits, query['label'])

                support_label = torch.cat(support['label'], 0).cuda()
                if self.args.use_class_weights:
                    class_weight = []
                    count_class_no_sample = 0
                    for i in range(self.args.N+1):
                        if support_label[support_label==i].shape[0] == 0:
                            count_class_no_sample += 1
                            class_weight.append(1)
                        else:
                            class_weight.append(1 / support_label[support_label==i].shape[0])
                    class_weight = torch.Tensor(class_weight).cuda()
                    # print("count_class_no_sample:", count_class_no_sample)
                    loss_fun = nn.CrossEntropyLoss(ignore_index=self.args.ignore_index, weight=class_weight)

                # MAML
                support_label = torch.cat(support['label'], 0).cuda()
                support_logits, support_pred = model(support)
                if self.args.use_class_weights == True:
                    support_loss = loss_fun(support_logits, support_label)
                else:
                    support_loss = model.loss(support_logits, support_label)
                # support_loss = model.loss(support_logits, support_label)

                self.__zero_grad__(model.fc.parameters())
                # self.__zero_grad__(model.fc1.parameters())
                # self.__zero_grad__(model.fc2.parameters())
                # self.__zero_grad__(model.fc3.parameters())
                # self.__zero_grad__(model.fc4.parameters())

                grads_fc = autograd.grad(support_loss, model.fc.parameters(), allow_unused=True, retain_graph=True)
                fast_weights_fc, orderd_params_fc = model.cloned_fc_dict(), OrderedDict()
                for (key, val), grad in zip(model.fc.named_parameters(), grads_fc):
                    fast_weights_fc[key] = orderd_params_fc[key] = val - self.args.task_lr * grad

                # grads_fc1 = autograd.grad(support_loss, model.fc1.parameters(), allow_unused=True, retain_graph=True)
                # fast_weights_fc1, orderd_params_fc1 = model.cloned_fc1_dict(), OrderedDict()
                # for (key, val), grad in zip(model.fc1.named_parameters(), grads_fc1):
                #     fast_weights_fc1[key] = orderd_params_fc1[key] = val - self.args.task_lr * grad

                # grads_fc2 = autograd.grad(support_loss, model.fc2.parameters(), allow_unused=True, retain_graph=True)
                # fast_weights_fc2, orderd_params_fc2 = model.cloned_fc2_dict(), OrderedDict()
                # for (key, val), grad in zip(model.fc2.named_parameters(), grads_fc2):
                #     fast_weights_fc2[key] = orderd_params_fc2[key] = val - self.args.task_lr * grad

                # grads_fc3 = autograd.grad(support_loss, model.fc3.parameters(), allow_unused=True, retain_graph=True)
                # fast_weights_fc3, orderd_params_fc3 = model.cloned_fc3_dict(), OrderedDict()
                # for (key, val), grad in zip(model.fc3.named_parameters(), grads_fc3):
                #     fast_weights_fc3[key] = orderd_params_fc3[key] = val - self.args.task_lr * grad

                # grads_fc4 = autograd.grad(support_loss, model.fc4.parameters(), allow_unused=True, retain_graph=True)
                # fast_weights_fc4, orderd_params_fc4 = model.cloned_fc4_dict(), OrderedDict()
                # for (key, val), grad in zip(model.fc4.named_parameters(), grads_fc4):
                #     fast_weights_fc4[key] = orderd_params_fc4[key] = val - self.args.task_lr * grad

                fast_weights = {}
                fast_weights['fc'] = fast_weights_fc
                # fast_weights['fc1'] = fast_weights_fc1
                # fast_weights['fc2'] = fast_weights_fc2
                # fast_weights['fc3'] = fast_weights_fc3
                # fast_weights['fc4'] = fast_weights_fc4

                train_support_acc = []
                train_support_loss = []
                # if (it+1) % 50 == 0:
                #     self.args.task_lr /= 2
                for i in range(self.args.train_support_iter - 1):
                    support_logits, support_pred = model(support, fast_weights)
                    if self.args.use_class_weights == True:
                        support_loss = loss_fun(support_logits, support_label)
                    else:
                        support_loss = model.loss(support_logits, support_label)
                    # support_loss = model.loss(support_logits, support_label)
                    support_acc = ((support_pred == support_label).float()).sum().item() / support_pred.shape[0]
                    train_support_acc.append(support_acc)
                    train_support_loss.append(support_loss.item())
                    # print('train_support', support_loss)
                    self.__zero_grad__(orderd_params_fc.values())
                    # self.__zero_grad__(model.fc1.parameters())
                    # self.__zero_grad__(model.fc2.parameters())
                    # self.__zero_grad__(model.fc3.parameters())
                    # self.__zero_grad__(model.fc4.parameters())
                    grads_fc = torch.autograd.grad(support_loss, orderd_params_fc.values(), allow_unused=True, retain_graph=False)
                    for (key, val), grad in zip(orderd_params_fc.items(), grads_fc):
                        if grad is not None:
                            fast_weights['fc'][key] = orderd_params_fc[key] = val - self.args.task_lr * grad
                    
                    # grads_fc1 = torch.autograd.grad(support_loss, orderd_params_fc1.values(), allow_unused=True, retain_graph=True)
                    # for (key, val), grad in zip(orderd_params_fc1.items(), grads_fc1):
                    #     if grad is not None:
                    #         fast_weights['fc1'][key] = orderd_params_fc1[key] = val - self.args.task_lr * grad
                    
                    # grads_fc2 = torch.autograd.grad(support_loss, orderd_params_fc2.values(), allow_unused=True, retain_graph=True)
                    # for (key, val), grad in zip(orderd_params_fc2.items(), grads_fc2):
                    #     if grad is not None:
                    #         fast_weights['fc2'][key] = orderd_params_fc2[key] = val - self.args.task_lr * grad

                    # grads_fc3 = torch.autograd.grad(support_loss, orderd_params_fc3.values(), allow_unused=True, retain_graph=True)
                    # for (key, val), grad in zip(orderd_params_fc3.items(), grads_fc3):
                    #     if grad is not None:
                    #         fast_weights['fc3'][key] = orderd_params_fc3[key] = val - self.args.task_lr * grad

                    # grads_fc4 = torch.autograd.grad(support_loss, orderd_params_fc4.values(), allow_unused=True)
                    # for (key, val), grad in zip(orderd_params_fc4.items(), grads_fc4):
                    #     if grad is not None:
                    #         fast_weights['fc4'][key] = orderd_params_fc4[key] = val - self.args.task_lr * grad
                
                query_logits, query_pred = model(query, fast_weights)
                # query_loss = model.loss(query_logits, query_label)


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

                # if ckpt is not None:
                #     if self.args.save_test_inference is not 'none':
                #         sentence_list, real_label_list, pred_label_list, label_name_list = self.__save_test_inference__(query_logits, query_pred, query)
                #         assert len(sentence_list) == len(real_label_list) == len(pred_label_list) == len(label_name_list)
                #         for i in range(len(sentence_list)):
                #             assert len(sentence_list[i]) == len(real_label_list[i]) == len(pred_label_list[i]) == len(label_name_list[i])
                #             for j in range(len(sentence_list[i])):
                #                 f_write.write(sentence_list[i][j] + '\t' + real_label_list[i][j] + '\t' + pred_label_list[i][j] + '\n')
                #                 f_write.flush()
                #             f_write.write('\n')
                #             f_write.flush()

                if it + 1 == eval_iter:
                    break
                it += 1

        precision = correct_cnt / pred_cnt
        recall = correct_cnt / label_cnt
        f1 = 2 * precision * recall / (precision + recall + 1e-10)
        fp_error = fp_cnt / total_token_cnt
        fn_error = fn_cnt / total_token_cnt
        within_error = within_cnt / total_span_cnt
        outer_error = outer_cnt / total_span_cnt
        print('[EVAL] step: {0:4} | [ENTITY] precision: {1:3.4f}, recall: {2:3.4f}, f1: {3:3.4f}'.format(it + 1, precision, recall, f1) + '\r')

        # if ckpt is not None:
        #         if self.args.save_test_inference is not 'none':
        #             f_write.close()

        return precision, recall, f1, fp_error, fn_error, within_error, outer_error


class FewShotNERFramework_RelationNER:

    def __init__(self, train_data_loader, val_data_loader, test_data_loader, args, tokenizer, viterbi=False, N=None, train_fname=None, tau=0.05, use_sampled_data=True):
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
        # self.viterbi = viterbi
        # if viterbi:
        #     abstract_transitions = get_abstract_transitions(train_fname, use_sampled_data=use_sampled_data)
        #     self.viterbi_decoder = ViterbiDecoder(N+2, abstract_transitions, tau)

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

    def train(self,
              model,
              model_name,
              learning_rate=1e-1,
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
                # support, query = next(self.train_data_loader)
                if torch.cuda.is_available():
                    for k in support:
                        if k != 'label' and k != 'sentence_num':
                            support[k] = support[k].cuda()
                            query[k] = query[k].cuda()
                    query_label = torch.cat(query['label'], 0)
                    query_label = query_label.cuda()

                # RelationNER
                # 1. get label name id in tokenizer
                label_data = self.__generate_label_data__(query)

                support_logits, support_pair_label, support_pred = model(support, label_data)
                support_logits = support_logits.view(-1)
                support_pair_label = torch.Tensor(support_pair_label).long().cuda()

                # MAML

                if self.args.use_class_weights:
                    class_weight = []
                    for i in range(2):
                        class_weight.append(self.args.cs / support_pair_label[support_pair_label==i].shape[0])
                    class_weight_list = []
                    for label in support_pair_label:
                        class_weight_list.append(torch.tensor(class_weight[label]))
                    class_weight_list = torch.stack(class_weight_list, dim=0).cuda()
                    loss_fun = nn.BCEWithLogitsLoss(weight=class_weight_list)
                else:
                    loss_fun = nn.BCEWithLogitsLoss()

                support_pair_label = support_pair_label.float()

                support_loss = loss_fun(support_logits, support_pair_label)

                self.__zero_grad__(model.fc.parameters())

                grads_fc = autograd.grad(support_loss, model.fc.parameters(), allow_unused=True, retain_graph=False)
                fast_weights_fc, orderd_params_fc = model.cloned_fc_dict(), OrderedDict()
                for (key, val), grad in zip(model.fc.named_parameters(), grads_fc):
                    fast_weights_fc[key] = orderd_params_fc[key] = val - self.args.task_lr * grad

                fast_weights = {}
                fast_weights['fc'] = fast_weights_fc

                train_support_acc = []
                train_support_loss = []
                # if (it+1) % 50 == 0:
                #     self.args.task_lr /= 2
                for i in range(self.args.train_support_iter - 1):
                    support_logits, support_pair_label, support_pred = model(support, label_data, fast_weights)
                    support_logits = support_logits.view(-1)
                    support_pair_label = torch.Tensor(support_pair_label).cuda()
                    support_loss = loss_fun(support_logits, support_pair_label)
                    train_support_loss.append(support_loss.item())
                    # print('train_support', support_loss)
                    self.__zero_grad__(orderd_params_fc.values())

                    grads_fc = torch.autograd.grad(support_loss, orderd_params_fc.values(), allow_unused=True, retain_graph=True)
                    for (key, val), grad in zip(orderd_params_fc.items(), grads_fc):
                        if grad is not None:
                            fast_weights['fc'][key] = orderd_params_fc[key] = val - self.args.task_lr * grad
                
                query_logits, query_pair_label, query_pred = model(query, label_data, fast_weights)
                query_logits = query_logits.view(-1)
                query_pair_label = torch.Tensor(query_pair_label).long().cuda()

                if self.args.use_class_weights:
                    class_weight = []
                    for i in range(2):
                        class_weight.append(self.args.cs / query_pair_label[query_pair_label==i].shape[0])
                    class_weight_list = []
                    for label in query_pair_label:
                        class_weight_list.append(torch.tensor(class_weight[label]))
                    class_weight_list = torch.stack(class_weight_list, dim=0).cuda()
                    loss_fun_query = nn.BCEWithLogitsLoss(weight=class_weight_list)
                else:
                    loss_fun_query = nn.BCEWithLogitsLoss()
                
                query_pair_label = query_pair_label.float()
                query_loss = loss_fun_query(query_logits, query_pair_label)
                # print('query_support', query_pred)

                # query_label_no_neg = []
                # for ql in query_label:
                #     if ql != -1:
                #         query_label_no_neg.append(ql)
                # query_label_no_neg = torch.Tensor(query_label_no_neg).long().cuda()
                query_pred = torch.Tensor(query_pred).long().cuda()
                tmp_pred_cnt, tmp_label_cnt, correct = model.metrics_by_entity(query_pred, query_label)

                bert_optimizer.zero_grad()
                wobert_optimizer.zero_grad()
                    
                if fp16:
                    with amp.scale_loss(query_loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    query_loss.backward()
                
                if it % grad_iter == 0:
                    bert_optimizer.step()
                    wobert_optimizer.step()
                    bert_scheduler.step()
                    wobert_scheduler.step()

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
                    iter_loss = 0.
                    iter_sample = 0.
                    pred_cnt = 0
                    label_cnt = 0
                    correct_cnt = 0
                    # sys.stdout.write('step: {0:4} | loss: {1:2.6f} | [ENTITY] precision: {2:3.4f}, recall: {3:3.4f}, f1: {4:3.4f}'\
                    #     .format(it + 1, iter_loss/ iter_sample, precision, recall, f1) + '\r')
                # sys.stdout.flush()

                if (it + 1) % val_step == 0:
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
        print("")
        
        model.eval()
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

        pred_cnt = 0 # pred entity cnt
        label_cnt = 0 # true label entity cnt
        correct_cnt = 0 # correct predicted entity cnt

        fp_cnt = 0 # misclassify O as I-
        fn_cnt = 0 # misclassify I- as O
        total_token_cnt = 0 # total token cnt
        within_cnt = 0 # span correct but of wrong fine-grained type 
        outer_cnt = 0 # span correct but of wrong coarse-grained type
        total_span_cnt = 0 # span correct

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

                # RelationNER
                # 1. get label name id in tokenizer
                label_data = self.__generate_label_data__(query)

                support_logits, support_pair_label, support_pred = model(support, label_data)
                support_logits = support_logits.view(-1)
                support_pair_label = torch.Tensor(support_pair_label).long().cuda()

                # MAML

                if self.args.use_class_weights:
                    class_weight = []
                    for i in range(2):
                        class_weight.append(self.args.cs / support_pair_label[support_pair_label==i].shape[0])
                    class_weight_list = []
                    for label in support_pair_label:
                        class_weight_list.append(torch.tensor(class_weight[label]))
                    class_weight_list = torch.stack(class_weight_list, dim=0).cuda()
                    loss_fun = nn.BCEWithLogitsLoss(weight=class_weight_list)
                else:
                    loss_fun = nn.BCEWithLogitsLoss()

                support_pair_label = support_pair_label.float()

                support_loss = loss_fun(support_logits, support_pair_label)

                self.__zero_grad__(model.fc.parameters())

                grads_fc = autograd.grad(support_loss, model.fc.parameters(), allow_unused=True, retain_graph=False)
                fast_weights_fc, orderd_params_fc = model.cloned_fc_dict(), OrderedDict()
                for (key, val), grad in zip(model.fc.named_parameters(), grads_fc):
                    fast_weights_fc[key] = orderd_params_fc[key] = val - self.args.task_lr * grad

                fast_weights = {}
                fast_weights['fc'] = fast_weights_fc

                train_support_acc = []
                train_support_loss = []

                for i in range(self.args.train_support_iter - 1):
                    support_logits, support_pair_label, support_pred = model(support, label_data, fast_weights)
                    support_logits = support_logits.view(-1)
                    support_pair_label = torch.Tensor(support_pair_label).cuda()
                    support_loss = loss_fun(support_logits, support_pair_label)
                    train_support_loss.append(support_loss.item())
                    # print('train_support', support_loss)
                    self.__zero_grad__(orderd_params_fc.values())

                    grads_fc = torch.autograd.grad(support_loss, orderd_params_fc.values(), allow_unused=True, retain_graph=True)
                    for (key, val), grad in zip(orderd_params_fc.items(), grads_fc):
                        if grad is not None:
                            fast_weights['fc'][key] = orderd_params_fc[key] = val - self.args.task_lr * grad
                
                query_logits, query_pair_label, query_pred = model(query, label_data, fast_weights)
                query_logits = query_logits.view(-1)
                query_pair_label = torch.Tensor(query_pair_label).long().cuda()

                if self.args.use_class_weights:
                    class_weight = []
                    for i in range(2):
                        class_weight.append(self.args.cs / query_pair_label[query_pair_label==i].shape[0])
                    class_weight_list = []
                    for label in query_pair_label:
                        class_weight_list.append(torch.tensor(class_weight[label]))
                    class_weight_list = torch.stack(class_weight_list, dim=0).cuda()
                    loss_fun_query = nn.BCEWithLogitsLoss(weight=class_weight_list)
                else:
                    loss_fun_query = nn.BCEWithLogitsLoss()
                
                query_pair_label = query_pair_label.float()
                query_loss = loss_fun_query(query_logits, query_pair_label)

                # query_label_no_neg = []
                # for ql in query_label:
                #     if ql != -1:
                #         query_label_no_neg.append(ql)
                # query_label_no_neg = torch.Tensor(query_label_no_neg).long().cuda()
                query_pred = torch.Tensor(query_pred).long().cuda()
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

                # if ckpt is not None:
                #     if self.args.save_test_inference is not 'none':
                #         sentence_list, real_label_list, pred_label_list, label_name_list = self.__save_test_inference__(query_logits, query_label_no_neg, query)
                #         assert len(sentence_list) == len(real_label_list) == len(pred_label_list) == len(label_name_list)
                #         for i in range(len(sentence_list)):
                #             assert len(sentence_list[i]) == len(real_label_list[i]) == len(pred_label_list[i]) == len(label_name_list[i])
                #             for j in range(len(sentence_list[i])):
                #                 f_write.write(sentence_list[i][j] + '\t' + real_label_list[i][j] + '\t' + pred_label_list[i][j] + '\n')
                #                 f_write.flush()
                #             f_write.write('\n')
                #             f_write.flush()

                if it + 1 == eval_iter:
                    break
                it += 1

        precision = correct_cnt / pred_cnt
        recall = correct_cnt / label_cnt
        f1 = 2 * precision * recall / (precision + recall + 1e-10)
        fp_error = fp_cnt / total_token_cnt
        fn_error = fn_cnt / total_token_cnt
        within_error = within_cnt / total_span_cnt
        outer_error = outer_cnt / total_span_cnt
        print('[EVAL] step: {0:4} | [ENTITY] precision: {1:3.4f}, recall: {2:3.4f}, f1: {3:3.4f}'.format(it + 1, precision, recall, f1) + '\r')

        # if ckpt is not None:
        #         if self.args.save_test_inference is not 'none':
        #             f_write.close()

        return precision, recall, f1, fp_error, fn_error, within_error, outer_error


class FewShotNERFramework_Siamese:

    def __init__(self, train_data_loader, val_data_loader, test_data_loader, args, tokenizer, viterbi=False, N=None, train_fname=None, tau=0.05, use_sampled_data=True):
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
        # self.viterbi = viterbi
        # if viterbi:
        #     abstract_transitions = get_abstract_transitions(train_fname, use_sampled_data=use_sampled_data)
        #     self.viterbi_decoder = ViterbiDecoder(N+2, abstract_transitions, tau)

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
        data['word'] = data['word'][data['text_mask']==1]
        data['label'] = torch.cat(data['label'], dim=0)
        data_1['word'] = data['word'][[l in [*range(1, self.args.N+1)] for l in data['label']]]
        data_1['label'] = data['label'][[l in [*range(1, self.args.N+1)] for l in data['label']]]
        data_2['word'] = data['word'][[l in [*range(0, self.args.N+1)] for l in data['label']]]
        data_2['label'] = data['label'][[l in [*range(0, self.args.N+1)] for l in data['label']]]

        return data_1, data_2
    

    def train(self,
              model,
              model_name,
              learning_rate=1e-1,
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
        loss_func = ContrastiveLoss(args=self.args)

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
                if torch.cuda.is_available():
                    for k in support:
                        if k != 'label' and k != 'sentence_num':
                            support[k] = support[k].cuda()
                            query[k] = query[k].cuda()
                    query_label = torch.cat(query['label'], 0)
                    query_label = query_label.cuda()

                support_dis, support_pair_label = model(support, query)
                support_loss, margin = loss_func(support_dis, support_pair_label)

                bert_optimizer.zero_grad()
                wobert_optimizer.zero_grad()
                    
                if fp16:
                    with amp.scale_loss(support_loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    support_loss.backward()
                
                if it % grad_iter == 0:
                    bert_optimizer.step()
                    wobert_optimizer.step()
                    bert_scheduler.step()
                    wobert_scheduler.step()
                
                query_pred = model(support, query, query_flag=True, margin=margin)
                tmp_pred_cnt, tmp_label_cnt, correct = model.metrics_by_entity(query_pred, query_label)

                iter_loss += self.item(support_loss.data)
                pred_cnt += tmp_pred_cnt
                label_cnt += tmp_label_cnt
                correct_cnt += correct
                iter_sample += 1

                '''
                # # MAML

                # if self.args.use_class_weights:
                #     class_weight = []
                #     for i in range(2):
                #         class_weight.append(self.args.cs / support_pair_label[support_pair_label==i].shape[0])
                #     class_weight_list = []
                #     for label in support_pair_label:
                #         class_weight_list.append(torch.tensor(class_weight[label]))
                #     class_weight_list = torch.stack(class_weight_list, dim=0).cuda()
                #     loss_fun = nn.BCEWithLogitsLoss(weight=class_weight_list)
                # else:
                #     loss_fun = nn.BCEWithLogitsLoss()

                # support_pair_label = support_pair_label.float()

                # support_loss = loss_fun(support_logits, support_pair_label)

                # self.__zero_grad__(model.fc.parameters())

                # grads_fc = autograd.grad(support_loss, model.fc.parameters(), allow_unused=True, retain_graph=False)
                # fast_weights_fc, orderd_params_fc = model.cloned_fc_dict(), OrderedDict()
                # for (key, val), grad in zip(model.fc.named_parameters(), grads_fc):
                #     fast_weights_fc[key] = orderd_params_fc[key] = val - self.args.task_lr * grad

                # fast_weights = {}
                # fast_weights['fc'] = fast_weights_fc

                # train_support_acc = []
                # train_support_loss = []
                # # if (it+1) % 50 == 0:
                # #     self.args.task_lr /= 2
                # for i in range(self.args.train_support_iter - 1):
                #     support_logits, support_pair_label, support_pred = model(support, label_data, fast_weights)
                #     support_logits = support_logits.view(-1)
                #     support_pair_label = torch.Tensor(support_pair_label).cuda()
                #     support_loss = loss_fun(support_logits, support_pair_label)
                #     train_support_loss.append(support_loss.item())
                #     # print('train_support', support_loss)
                #     self.__zero_grad__(model.fc.parameters())

                #     grads_fc = torch.autograd.grad(support_loss, orderd_params_fc.values(), allow_unused=True, retain_graph=True)
                #     for (key, val), grad in zip(orderd_params_fc.items(), grads_fc):
                #         if grad is not None:
                #             fast_weights['fc'][key] = orderd_params_fc[key] = val - self.args.task_lr * grad
                
                # query_logits, query_pair_label, query_pred = model(query, label_data, fast_weights)
                # query_logits = query_logits.view(-1)
                # query_pair_label = torch.Tensor(query_pair_label).long().cuda()

                # if self.args.use_class_weights:
                #     class_weight = []
                #     for i in range(2):
                #         class_weight.append(self.args.cs / query_pair_label[query_pair_label==i].shape[0])
                #     class_weight_list = []
                #     for label in query_pair_label:
                #         class_weight_list.append(torch.tensor(class_weight[label]))
                #     class_weight_list = torch.stack(class_weight_list, dim=0).cuda()
                #     loss_fun_query = nn.BCEWithLogitsLoss(weight=class_weight_list)
                # else:
                #     loss_fun_query = nn.BCEWithLogitsLoss()
                
                # query_pair_label = query_pair_label.float()
                # query_loss = loss_fun_query(query_logits, query_pair_label)
                # print('query_support', query_pred)

                # query_label_no_neg = []
                # for ql in query_label:
                #     if ql != -1:
                #         query_label_no_neg.append(ql)
                # query_label_no_neg = torch.Tensor(query_label_no_neg).long().cuda()
                # query_pred = torch.Tensor(query_pred).long().cuda()
                # tmp_pred_cnt, tmp_label_cnt, correct = model.metrics_by_entity(query_pred, query_label)

                # bert_optimizer.zero_grad()
                # wobert_optimizer.zero_grad()
                    
                # if fp16:
                #     with amp.scale_loss(query_loss, optimizer) as scaled_loss:
                #         scaled_loss.backward()
                # else:
                #     query_loss.backward()
                
                # if it % grad_iter == 0:
                #     bert_optimizer.step()
                #     wobert_optimizer.step()
                #     bert_scheduler.step()
                #     wobert_scheduler.step()

                # iter_loss += self.item(query_loss.data)
                # pred_cnt += tmp_pred_cnt
                # label_cnt += tmp_label_cnt
                # correct_cnt += correct
                # iter_sample += 1
                '''
                if (it + 1) % 100 == 0 or (it + 1) % val_step == 0:
                    precision = correct_cnt / (pred_cnt + 1e-10)
                    recall = correct_cnt / (label_cnt + 1e-10)
                    f1 = 2 * precision * recall / (precision + recall + 1e-10)
                    print('step: {0:4} | loss: {1:2.6f} | [ENTITY] precision: {2:3.4f}, recall: {3:3.4f}, f1: {4:3.4f}'\
                        .format(it + 1, iter_loss/ iter_sample, precision, recall, f1) + '\r')
                    iter_loss = 0.
                    iter_sample = 0.
                    pred_cnt = 0
                    label_cnt = 0
                    correct_cnt = 0

                if (it + 1) % val_step == 0:
                    torch.save({'state_dict': model.state_dict()}, 'current_siamese.ckpt')
                    _, _, f1, _, _, _, _ = self.eval(model, val_iter, ckpt='current_siamese.ckpt')
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

        # set Bert learning rate
        parameters_to_optimize = list(model.word_encoder.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        parameters_to_optimize = [
            {'params': [p for n, p in parameters_to_optimize 
                if not any(nd in n for nd in no_decay)], 'weight_decay': self.args.bert_wd},
            {'params': [p for n, p in parameters_to_optimize
                if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        
        bert_optimizer = AdamW(parameters_to_optimize, lr=self.args.bert_lr, correct_bias=False)
        bert_scheduler = get_linear_schedule_with_warmup(bert_optimizer, num_warmup_steps=300, num_training_steps=30000) 

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
        wobert_scheduler = get_linear_schedule_with_warmup(wobert_optimizer, num_warmup_steps=300, num_training_steps=30000) 

        loss_func = ContrastiveLoss(args=self.args)
        if ckpt is None:
            # print("Use val dataset")
            # eval_dataset = self.val_data_loader
            print("Use test dataset")
            eval_dataset = self.test_data_loader
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

        pred_cnt = 0 # pred entity cnt
        label_cnt = 0 # true label entity cnt
        correct_cnt = 0 # correct predicted entity cnt

        fp_cnt = 0 # misclassify O as I-
        fn_cnt = 0 # misclassify I- as O
        total_token_cnt = 0 # total token cnt
        within_cnt = 0 # span correct but of wrong fine-grained type 
        outer_cnt = 0 # span correct but of wrong coarse-grained type
        total_span_cnt = 0 # span correct

        eval_iter = min(eval_iter, len(eval_dataset))

        # print test inference
        if ckpt is not None:
            if self.args.save_test_inference is not 'none':
                # generate save path
                if self.args.dataset == 'fewnerd':
                    save_path = '_'.join([self.args.save_test_inference, self.args.dataset, self.args.mode, self.args.model, str(self.args.N), str(self.args.K)])
                else:
                    save_path = '_'.join([self.args.save_test_inference, self.args.dataset, self.args.model, str(self.args.N), str(self.args.K)])
                f_write = open(save_path + '.txt', 'a', encoding='utf-8')


        it = 0
        while it + 1 < eval_iter:
            for _, (support, query) in enumerate(eval_dataset):
                # 每轮重读模型参数
                state_dict = self.__load_model__(ckpt)['state_dict']
                own_state = model.state_dict()
                for name, param in state_dict.items():
                    if name not in own_state:
                        continue
                    own_state[name].copy_(param)

                if torch.cuda.is_available():
                    for k in support:
                        if k != 'label' and k != 'sentence_num':
                            support[k] = support[k].cuda()
                            query[k] = query[k].cuda()
                    query_label = torch.cat(query['label'], 0)
                    query_label = query_label.cuda()

                label_data = self.__generate_label_data__(query)

                support_dis, support_pair_label = model(support, query)
                support_loss, margin = loss_func(support_dis, support_pair_label)

                bert_optimizer.zero_grad()
                wobert_optimizer.zero_grad()

                support_loss.backward()

                bert_optimizer.step()
                wobert_optimizer.step()
                bert_scheduler.step()
                wobert_scheduler.step()

                query_pred = model(support, query, query_flag=True, margin=margin)
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

        precision = correct_cnt / pred_cnt
        recall = correct_cnt / label_cnt
        f1 = 2 * precision * recall / (precision + recall + 1e-10)
        fp_error = fp_cnt / total_token_cnt
        fn_error = fn_cnt / total_token_cnt
        within_error = within_cnt / total_span_cnt
        outer_error = outer_cnt / total_span_cnt
        print('[EVAL] step: {0:4} | [ENTITY] precision: {1:3.4f}, recall: {2:3.4f}, f1: {3:3.4f}'.format(it + 1, precision, recall, f1) + '\r')

        # sys.stdout.write('[EVAL] step: {0:4} | [ENTITY] precision: {1:3.4f}, recall: {2:3.4f}, f1: {3:3.4f}'.format(it + 1, precision, recall, f1) + '\r')
        # sys.stdout.flush()
        # print("")
        if ckpt is not None:
                if self.args.save_test_inference is not 'none':
                    f_write.close()

        return precision, recall, f1, fp_error, fn_error, within_error, outer_error


class FewShotNERFramework_SiameseMAML:

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
        loss_func = ContrastiveLoss(args=self.args)

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
                if torch.cuda.is_available():
                    for k in support:
                        if k != 'label' and k != 'sentence_num':
                            support[k] = support[k].cuda()
                            query[k] = query[k].cuda()
                    query_label = torch.cat(query['label'], 0)
                    query_label = query_label.cuda()


                support_emb = model.word_encoder(support['word'], support['mask'])
                support_emb = model.drop(support_emb)
                support['word_emb'] = support_emb
                support['word_emb'] = support['word_emb'][support['text_mask']==1]
                support['label'] = torch.cat(support['label'], dim=0)
                support_1, support_2 = self.__get_sample_pairs__(support)
                support_pair_label = self.__generate_pair_label__(support_1['label'], support_2['label'])

                support_dis = model(support_1, support_2)
                if self.args.margin_num == 0:
                    margin = torch.mean(support_dis)
                elif self.args.margin_num == -1:
                    margin = model.param
                else:
                    sorted, index = torch.sort(support_dis)
                    margin = sorted[self.args.N*self.args.K*self.args.margin_num]

                support_loss = loss_func(support_dis, support_pair_label, margin)

                self.__zero_grad__(model.fc.parameters())  # 或许是model.parameters()--即Bert的梯度也清了，应该问题不大

                grads_fc = autograd.grad(support_loss, model.fc.parameters(), allow_unused=True, retain_graph=True)
                fast_weights_fc, orderd_params_fc = model.cloned_fc_dict(), OrderedDict()
                for (key, val), grad in zip(model.fc.named_parameters(), grads_fc):
                    fast_weights_fc[key] = orderd_params_fc[key] = val - self.args.task_lr * grad  # grad中weight数量级是1e-4，bias是1e-11，有点太小了？
                
                fast_weights = {}
                fast_weights['fc'] = fast_weights_fc

                train_support_loss = []
                for _ in range(self.args.train_support_iter - 1):
                    support_dis = model(support_1, support_2, model_parameters=fast_weights)
                    support_loss = loss_func(support_dis, support_pair_label, margin)
                    train_support_loss.append(support_loss.item())
                    # print_info = 'train_support, ' + str(support_loss.item())
                    # print('\033[0;31;40m{}\033[0m'.format(print_info))
                    self.__zero_grad__(orderd_params_fc.values())

                    grads_fc = torch.autograd.grad(support_loss, orderd_params_fc.values(), allow_unused=True, retain_graph=True)
                    for (key, val), grad in zip(orderd_params_fc.items(), grads_fc):
                        if grad is not None:
                            fast_weights['fc'][key] = orderd_params_fc[key] = val - self.args.task_lr * grad
                
                query_emb = model.word_encoder(query['word'], query['mask'])
                query_emb = model.drop(query_emb)
                query['word_emb'] = query_emb
                query['word_emb'] = query['word_emb'][query['text_mask']==1]
                query['label'] = query_label
                # 1. 得到query set对应的预测值
                
                # [x, 768] -> [x, 256]
                query['word_emb'] = model.__forward_once_with_param__(query['word_emb'], fast_weights)
                # 得到proto
                support_out = model.__forward_once_with_param__(support['word_emb'], fast_weights)
                proto = []
                assert support_out.shape[0] == support['label'].shape[0]
                for label in range(1, self.args.N+1):
                    proto.append(torch.mean(support_out[support['label']==label], dim=0))
                proto = torch.stack(proto)
                # 计算query->proto距离
                query_dis = model.__pos_dist__(query['word_emb'], proto).view(-1)  # [x, N]
                query_dis = F.layer_norm(query_dis, normalized_shape=[query_dis.shape[0]], bias=torch.full((query_dis.shape[0],), self.args.ln_bias).cuda())
                query_dis = query_dis.view(-1, proto.shape[0])
                # 得到预测值
                query_pred = []
                for tmp in query_dis:
                    if any(t < margin for t in tmp):
                        query_pred.append(torch.min(tmp, dim=0)[1].item()+1)
                    else:
                        query_pred.append(0)
                query_pred = torch.Tensor(query_pred).cuda()
                assert query_pred.shape[0] == query_label.shape[0]

                # 2. 得到query set对应的loss
                query_dis, query_pair_label = self.__generate_query_pair_label__(query_dis, query_label)
                query_loss = loss_func(query_dis, query_pair_label, margin)

                bert_optimizer.zero_grad()
                wobert_optimizer.zero_grad()
                    
                if fp16:
                    with amp.scale_loss(query_loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    query_loss.backward()
                
                if it % grad_iter == 0:
                    bert_optimizer.step()
                    wobert_optimizer.step()
                    bert_scheduler.step()
                    wobert_scheduler.step()
                    # print('margin:', margin)
                
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
                    print('margin:', margin.item())
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
        loss_func = ContrastiveLoss(args=self.args)
        
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

        pred_cnt = 0 # pred entity cnt
        label_cnt = 0 # true label entity cnt
        correct_cnt = 0 # correct predicted entity cnt

        fp_cnt = 0 # misclassify O as I-
        fn_cnt = 0 # misclassify I- as O
        total_token_cnt = 0 # total token cnt
        within_cnt = 0 # span correct but of wrong fine-grained type 
        outer_cnt = 0 # span correct but of wrong coarse-grained type
        total_span_cnt = 0 # span correct

        eval_iter = min(eval_iter, len(eval_dataset))

        # print test inference
        if ckpt is not None:
            if self.args.save_test_inference is not 'none':
                # generate save path
                if self.args.dataset == 'fewnerd':
                    save_path = '_'.join([self.args.save_test_inference, self.args.dataset, self.args.mode, self.args.model, str(self.args.N), str(self.args.K)])
                else:
                    save_path = '_'.join([self.args.save_test_inference, self.args.dataset, self.args.model, str(self.args.N), str(self.args.K)])
                f_write = open(save_path + '.txt', 'a', encoding='utf-8')


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

                support_emb = model.word_encoder(support['word'], support['mask'])
                support_emb = model.drop(support_emb)
                support['word_emb'] = support_emb
                support['word_emb'] = support['word_emb'][support['text_mask']==1]
                support['label'] = torch.cat(support['label'], dim=0)
                support_1, support_2 = self.__get_sample_pairs__(support)
                support_pair_label = self.__generate_pair_label__(support_1['label'], support_2['label'])

                support_dis = model(support_1, support_2)
                if self.args.margin_num == 0:
                    margin = torch.mean(support_dis)
                elif self.args.margin_num == -1:
                    margin = model.param
                else:
                    sorted, index = torch.sort(support_dis)
                    margin = sorted[self.args.N*self.args.K*self.args.margin_num]

                support_loss = loss_func(support_dis, support_pair_label, margin)

                self.__zero_grad__(model.fc.parameters())  # 或许是model.parameters()--即Bert的梯度也清了，应该问题不大

                grads_fc = autograd.grad(support_loss, model.fc.parameters(), allow_unused=True, retain_graph=True)
                fast_weights_fc, orderd_params_fc = model.cloned_fc_dict(), OrderedDict()
                for (key, val), grad in zip(model.fc.named_parameters(), grads_fc):
                    fast_weights_fc[key] = orderd_params_fc[key] = val - self.args.task_lr * grad  # grad中weight数量级是1e-4，bias是1e-11，有点太小了？
                
                fast_weights = {}
                fast_weights['fc'] = fast_weights_fc

                train_support_loss = []
                for _ in range(self.args.train_support_iter - 1):
                    support_dis = model(support_1, support_2, model_parameters=fast_weights)
                    support_loss = loss_func(support_dis, support_pair_label, margin)
                    train_support_loss.append(support_loss.item())
                    # print_info = 'train_support, ' + str(support_loss.item())
                    # print('\033[0;31;40m{}\033[0m'.format(print_info))
                    self.__zero_grad__(orderd_params_fc.values())

                    grads_fc = torch.autograd.grad(support_loss, orderd_params_fc.values(), allow_unused=True, retain_graph=True)
                    for (key, val), grad in zip(orderd_params_fc.items(), grads_fc):
                        if grad is not None:
                            fast_weights['fc'][key] = orderd_params_fc[key] = val - self.args.task_lr * grad
                
                query_emb = model.word_encoder(query['word'], query['mask'])
                query_emb = model.drop(query_emb)
                query['word_emb'] = query_emb
                query['word_emb'] = query['word_emb'][query['text_mask']==1]
                # query['label'] = query_label
                # 1. 得到query set对应的预测值
                
                # [x, 768] -> [x, 256]
                query['word_emb'] = model.__forward_once_with_param__(query['word_emb'], fast_weights)
                # 得到proto
                support_out = model.__forward_once_with_param__(support['word_emb'], fast_weights)
                proto = []
                assert support_out.shape[0] == support['label'].shape[0]
                for label in range(1, self.args.N+1):
                    proto.append(torch.mean(support_out[support['label']==label], dim=0))
                proto = torch.stack(proto)
                # 计算query->proto距离
                query_dis = model.__pos_dist__(query['word_emb'], proto).view(-1)  # [x, N]
                query_dis = F.layer_norm(query_dis, normalized_shape=[query_dis.shape[0]], bias=torch.full((query_dis.shape[0],), self.args.ln_bias).cuda())
                query_dis = query_dis.view(-1, proto.shape[0])
                # 得到预测值
                query_pred = []
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

        precision = correct_cnt / pred_cnt
        recall = correct_cnt / label_cnt
        f1 = 2 * precision * recall / (precision + recall + 1e-10)
        fp_error = fp_cnt / total_token_cnt
        fn_error = fn_cnt / total_token_cnt
        within_error = within_cnt / total_span_cnt
        outer_error = outer_cnt / total_span_cnt
        print('[EVAL] step: {0:4} | [ENTITY] precision: {1:3.4f}, recall: {2:3.4f}, f1: {3:3.4f}'.format(it + 1, precision, recall, f1) + '\r')

        # sys.stdout.write('[EVAL] step: {0:4} | [ENTITY] precision: {1:3.4f}, recall: {2:3.4f}, f1: {3:3.4f}'.format(it + 1, precision, recall, f1) + '\r')
        # sys.stdout.flush()
        # print("")
        if ckpt is not None:
                if self.args.save_test_inference is not 'none':
                    f_write.close()

        return precision, recall, f1, fp_error, fn_error, within_error, outer_error


      