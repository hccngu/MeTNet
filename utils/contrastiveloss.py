import itertools
import torch

import torch.nn.functional as F



# 自定义ContrastiveLoss
class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, args):
        super(ContrastiveLoss, self).__init__()
        self.args = args

    def forward(self, dis, label, margin):

        tmp1 = (label) * torch.pow(dis, 2).squeeze(-1)
        # mean_val = torch.mean(euclidean_distance)
        tmp2 = (1 - label) * torch.pow(torch.clamp(margin - dis, min=0.0),
                                       2).squeeze(-1)
        loss_contrastive = torch.mean(tmp1 + tmp2)

        # print("**********************************************************************")
        return loss_contrastive

