import argparse
import numpy as np
import random

import torch

def config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='fewnerd', help='[fewcomm, fewnerd]')
    parser.add_argument('--dataset_mode', default='IO', help='[BIO, IO] only for fewcomm')
    parser.add_argument('--mode', default='inter', help='training mode [inter, intra, supervised], only use for fewnerd')
    parser.add_argument('--trainN', default=5, type=int, help='N in train')
    parser.add_argument('--N', default=5, type=int, help='N way')
    parser.add_argument('--K', default=1, type=int, help='K shot')
    parser.add_argument('--Q', default=1, type=int, help='Num of queries per class')
    parser.add_argument('--batch_size', default=1, type=int, help='batch size')
    parser.add_argument('--train_iter', default=6000, type=int, help='num of iters in training, default=6000')
    parser.add_argument('--val_iter', default=100, type=int, help='num of iters in validation')
    parser.add_argument('--test_iter', default=500, type=int, help='num of iters in testing, default=500')
    parser.add_argument('--val_step', default=200, type=int, help='val after training how many iters, default=200')
    parser.add_argument('--model', default='proto_multiOclass', help='model name [proto, nnshot, structshot, MTNet, MAML, relation_ner, Siamese, SiameseMAML, proto_multiOclass]')
    parser.add_argument('--max_length', default=100, type=int, help='max length')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate for proto, nnshot, structshot')
    parser.add_argument('--grad_iter', default=1, type=int, help='accumulate gradient every x iterations')
    # checkpoint/proto-fewnerd-inter-IO-5-1-seed0-1649991137204.pth.tar
    # checkpoint/MTNet-fewnerd-inter-IO-5-1-seed0-1649354243516.pth.tar
    parser.add_argument('--load_ckpt', default=None, help='load ckpt')
    parser.add_argument('--save_ckpt', default=None, help='save ckpt')
    parser.add_argument('--fp16', action='store_true', help='use nvidia apex fp16')
    parser.add_argument('--only_test', action='store_true', help='only test')
    parser.add_argument('--ckpt_name', type=str, default='', help='checkpoint name.')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--ignore_index', type=int, default=-1, help='label index to ignore when calculating loss and metrics')
    parser.add_argument('--use_sampled_data', action='store_true', help='use released sampled data, the data should be stored at "data/episode-data/" ')
    # experiment
    parser.add_argument('--use_sgd_for_bert', action='store_true', help='use SGD instead of AdamW for BERT.')
    # only for bert / roberta
    parser.add_argument('--pretrain_ckpt', default=None, help='bert / roberta pre-trained checkpoint')
    # for print inference
    parser.add_argument('--save_test_inference', default='none', help='test inference profile, default=test_inference')


    # only for prototypical networks
    parser.add_argument('--dot', action='store_true', help='use dot instead of L2 distance for proto')
    parser.add_argument('--mlp', action='store_true', help='use a mlp for proto')
    # only for structshot
    parser.add_argument('--tau', default=0.05, type=float, help='StructShot parameter to re-normalizes the transition probabilities')
    # RelationNER
    parser.add_argument("--use_class_weights", type=bool, default=True, help="use class weights for MAML, SiameseMAML")
    parser.add_argument("--cs", type=float, default=100, help="class weight hyper-param")
    parser.add_argument("--alpha", type=float, default=0.5, help="pred O when all preds are smaller than it")
    # Siamese
    parser.add_argument("--margin_num", type=int, default=-1, help="control margin(*N*K) by the number of distance, default=8")
    parser.add_argument("--margin", type=float, default=-1, help="control margin of distance, default=8")
    parser.add_argument('--only_use_test', action='store_true', help='eval use test set.')
    # parser.add_argument('--alpha_is_trainable', action='store_true', help='use trainable alpha or not.')
    parser.add_argument("--trainable_alpha_init", type=float, default=1.0, help="alpha init.default=0.5")


    # MTNet
    parser.add_argument("--bert_lr", type=float, default=2e-5, help="learning rate of bert")
    parser.add_argument("--meta_lr", type=float, default=5e-4, help="learning rate of meta(out)")
    parser.add_argument("--task_lr", type=float, default=1e-1, help="learning rate of task(in)")
    parser.add_argument("--train_support_iter", type=int, default=3, help="Number of iterations of training(in)")
    parser.add_argument("--neg_num", type=int, default=1, help="the bias of layer norm")
    parser.add_argument("--use_proto_as_neg", action='store_true', help="use proto as neg")
    parser.add_argument("--use_diff_threshold", action='store_true', help="use different thresholds")
    parser.add_argument('--threshold_mode', default='max', help='mean or max')
    parser.add_argument("--ln_bias", type=float, default=10.0, help="the bias of layer norm for SiameseMAML and MTNet")
    parser.add_argument("--trainable_margin_init", type=float, default=6.0, help="use trainable margin, it's the init of it.default=8.5")
    parser.add_argument("--dropout", type=float, default=0.0, help="dropout")
    parser.add_argument("--bert_wd", type=float, default=1e-5, help="bert weight decay")
    parser.add_argument("--wobert_wd", type=float, default=1e-5, help="weight decay of param without bert")
    parser.add_argument('--multi_margin', action='store_true', help='multi adaptive margin')

    # Ablation study
    parser.add_argument('--label_name_mode', default='LnAsQKV', help='[mean, LnAsQ, LnAsQKV]')
    parser.add_argument('--tripletloss_mode', default='sig+dp+dn', help='[tl, tl+dp, sig+dp+dn]')
    parser.add_argument('--have_otherO', action='store_true', help='[for Ablation study of MTNet]')

    # for visualization
    parser.add_argument('--save_query_ebd', action='store_true', help='[save query ebd]')
    parser.add_argument('--load_ckpt_proto', default=None, help='load ckpt')
    parser.add_argument('--load_ckpt_metnet', default=None, help='load ckpt')

    
    opt = parser.parse_args()

    return opt


def setSeed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def print_args(args):
    """
        Print arguments (only show the relevant arguments)
    """

    print("\nParameters:")
    for attr, value in sorted(args.__dict__.items()):
        print("\t{}={}".format(attr.upper(), value))
    print("""                          
        MTNet -> Go!Go!Go!                           
    """)

    return