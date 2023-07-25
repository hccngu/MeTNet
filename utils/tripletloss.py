import itertools
import torch

import torch.nn.functional as F



# 自定义ContrastiveLoss
class TripletLoss(torch.nn.Module):
    """
    Triplet loss function.
    """

    def __init__(self, args):
        super(TripletLoss, self).__init__()
        self.args = args

    def __tripletloss__(self, dp_one, dn_one, margin, alpha):
        # alpha = torch.sigmoid(alpha)
        # return alpha*torch.clamp(dp_one-dn_one+margin, min=0.0) + (1-alpha)*dp_one
        return torch.clamp(dp_one-dn_one+margin, min=0.0)
    
    def __tripletloss_dp__(self, dp_one, dn_one, margin, alpha):
        # alpha = torch.sigmoid(alpha)
        # return alpha*torch.clamp(dp_one-dn_one+margin, min=0.0) + (1-alpha)*dp_one
        return 0.5 * torch.clamp(dp_one-dn_one+margin, min=0.0) + 0.5 * dp_one
    
    def __sigmoid_tripletloss__(self, dp_one, dn_one, margin, alpha):
        pos_weight = torch.sigmoid(dp_one - margin)
        neg_weight = torch.sigmoid(margin - dn_one)
        return pos_weight*dp_one + neg_weight*torch.clamp(margin - dn_one, min=0.0)


    def forward(self, dis, label, margin, alpha):
        '''
        dis: [N, N*K]
        label: [N*K]
        margin: a trainable param
        '''
        loss = 0
        for i, ps_dis in enumerate(dis):
            temp_loss = 0
            if self.args.have_otherO is True:
                dp = ps_dis[label==i]
                dn = ps_dis[label!=i]
            else:
                dp = ps_dis[label==i+1]
                dn = ps_dis[label!=i+1]
            dn, index = torch.sort(dn)
            if dn.shape[0] > self.args.neg_num:
                dn = dn[:self.args.neg_num]
            for dp_one in dp:
                for dn_one in dn:
                    if self.args.tripletloss_mode == 'tl':
                        if self.args.multi_margin is True:
                            temp_loss += self.__tripletloss__(dp_one, dn_one, margin[i], alpha)
                        else:
                            temp_loss += self.__tripletloss__(dp_one, dn_one, margin, alpha)
                    elif self.args.tripletloss_mode == 'tl+dp':
                        if self.args.multi_margin is True:
                            temp_loss += self.__tripletloss_dp__(dp_one, dn_one, margin[i], alpha)
                        else:
                            temp_loss += self.__tripletloss_dp__(dp_one, dn_one, margin, alpha)
                    elif self.args.tripletloss_mode == 'sig+dp+dn':
                        if self.args.multi_margin is True:
                            temp_loss += self.__sigmoid_tripletloss__(dp_one, dn_one, margin[i], alpha)
                        else:
                            temp_loss += self.__sigmoid_tripletloss__(dp_one, dn_one, margin, alpha)
                    else:
                        raise NotImplementedError
            temp_loss = temp_loss/dp.shape[0]/dn.shape[0]
            loss += temp_loss
        loss = loss/dis.shape[0]
        return loss
            
        # tmp1 = (label) * torch.pow(dis, 2).squeeze(-1)
        # # mean_val = torch.mean(euclidean_distance)
        # tmp2 = (1 - label) * torch.pow(torch.clamp(margin - dis, min=0.0),
        #                                2).squeeze(-1)
        # loss_contrastive = torch.mean(tmp1 + tmp2)

        # # print("**********************************************************************")
        # return loss_contrastive

