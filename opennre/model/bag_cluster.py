import torch
from torch import nn, optim
import math
from .base_model import BagRE
import random
import math
import numpy as np

def convert_index(bag_num, labels, sbag_num, is_random=True):
    """
    初始化超包和包的对应索引
    Args:
    bag_num: 包的数量
    label: 包对应的标签
    sbag_num: 超包的数量
    is_random: 是否随机初始化，否则为使用标签初始化。
    return：
    cor_map: b->sb的关系，包和超包的关联数组，索引i的值为j，表示第i个包属于第j个超包
    sbag_map: sb->b的关系，(sbag_num, )
    cluster_labels: sb对应的label
    """
    cor_map = [-1] * bag_num
    labels = labels.cpu().numpy().tolist()
    if is_random:
        cluster_labels = labels
        for i in range(bag_num):
            if i < sbag_num:
                cor_map[i] = i
            else:
                cor_map[i] = math.floor(random.random() * sbag_num)
    else:
        # 先将不重合的超包标签定下来
        superbag_labels = list(set(labels))
        for (idx, label) in enumerate(superbag_labels):
            l_idx = labels.index(label)
            cor_map[l_idx] = idx
        # 当前超包的数量小于需要的数量，加入重合的超包标签
        iter_num = sbag_num - len(superbag_labels)
        for _ in range(iter_num):
            # 将第一个出现的未标记的包的标签标记为新的超包标签
            empty_idx = cor_map.index(-1)
            cor_map[empty_idx] = len(superbag_labels)
            superbag_labels.append(labels[empty_idx])
        for (idx, label) in enumerate(cor_map):
            if label < 0:
                cur_label = labels[idx]
                superbag_idxs = [i for (i, l) in enumerate(superbag_labels) if l == cur_label]
                selected = random.choice(superbag_idxs)
                cor_map[idx] = selected
    sbag_map = [[] for _ in range(sbag_num)]
    for idx, sbag_index in enumerate(cor_map):
        try:
            sbag_map[sbag_index].append(idx)
        except:
            print(sbag_index, cor_map)
    return cor_map, sbag_map


def SbagFeature(bag_feat, label, num_sbag, is_random=True):
    """
    通过平均值获取超包的特征
    Args: 
    bag_feat: 包特征(B,3H)
    label: 包对应的label
    num_sbag: 超包的个数
    return 
    超包的特征
    """
    bag_num, hidden_size = bag_feat.shape
    cor_map, sbag_map = convert_index(bag_num, label, num_sbag, is_random=is_random)
    ave_feat = []
    for sbags_idx in sbag_map:
        sbags_feat = []
        for idx in sbags_idx:
            try:
                sbags_feat.append(bag_feat[idx])
            except:
                print(idx)
        sbags_feat = torch.stack(sbags_feat)
#         print(sbags_feat.size(), sbags_feat.shape)
        ave_feat.append(torch.mul(sbags_feat.sum(0), 1/len(sbags_idx)))
    return torch.stack(ave_feat), cor_map, sbag_map

def Passoc(bag_feat, sbag_feat, sb2b_index, scale_value=1):
    '''
    calculate the distance between bag with each superbag. each iteration spixel_init is fixed,
    only change the feature and association.
    :param bag_feat: (B,3H)
    :param sbag_feat: (D, 3H) D is the number of surpixels
    :param p2sp_index_: (D)
    :param scale_value:
    :return:
    distance (B*D*3H)
    '''
    b, h = bag_feat.shape
    sb_num = len(sb2b_index)
#     if len(p2sp_index_.shape) == 3:
#         p2sp_index_ = torch.from_numpy(p2sp_index_).unsqueeze(0)
#         invisible_ = torch.from_numpy(invisible_).unsqueeze(0)
    bag_feat = bag_feat.repeat(1, sb_num).reshape(b, sb_num, h) # (B*D*3H)
    sbag_feat = sbag_feat.repeat(b, 1).reshape(b, -1, h ) #(B*D*3H)

    distance = torch.pow(sbag_feat - bag_feat, 2.0)  # 9*B*C*H*W  (occupy storage 440M)
    distance = distance * scale_value  # B*D*3H
    return distance

def compute_assignments(sbag_feature, bag_rep, sb2b_index):

    pixel_spixel_neg_dist = Passoc(bag_rep, sbag_feature, sb2b_index)
    pixel_spixel_assoc = (pixel_spixel_neg_dist - pixel_spixel_neg_dist.max(1, keepdim=True)[0]).exp()
    pixel_spixel_assoc = pixel_spixel_assoc / (pixel_spixel_assoc.sum(1, keepdim=True))
    
    return pixel_spixel_assoc

def SpixelFeature2(bag_feature, weight, num_spixels):
    '''
    calculate spixel feature according to the similarity matrix between pixel and spixel
    :param bag_feature: B*3H
    :param weight:  B*D*3H
    :return:D*3H
    '''
    b, h = bag_feature.shape
    
    feat = bag_feature.reshape(b, 1, h) # B*1*3H
    
    s_feat = feat * weight  #B*D*3H
    
#     s_feat = s_feat.reshape(b, 1, num_spixels, -1)  #B*D*(n/D)
#     weight = weight.reshape(b, 1, num_spixels, -1)  #B*D*(n/D)
    
    weight = weight.sum(0)  #D*H
    s_feat = s_feat.sum(0)  #D*H

    S_feat = s_feat / (weight + 1e-5)
    S_feat = S_feat * (weight > 0.001).float()
    return S_feat

def exec_iter(sbag_feature, bag_rep, sb2b_index):

    # Compute pixel-superpixel assignments
    # t3 = time.time()
    # print(f't2-t1:{t2-t1:.3f}, t3-t2:{t3-t2:.3f}')
    pixel_assoc = compute_assignments(sbag_feature, bag_rep, sb2b_index)
    sbag_feat = SpixelFeature2(bag_rep, pixel_assoc, len(sb2b_index))
    return sbag_feat, pixel_assoc


# 解码，用于计算紧致度损失
def decode_features(pixel_spixel_assoc, spixel_feat):
    """
    :param pixel_spixel_assoc: B*D*3H the distance of each bag and each sbag
    :param spixel_feat: B*D*3H sbag feature
    :return:
    """
    b, d, h, = pixel_spixel_assoc.shape
    recon_feat = spixel_feat.sum(1) + 1e-10  # B*3H

    # norm
    try:
        assert recon_feat.min() >= 0., 'fails'
    except:
        import pdb
        pdb.set_trace()
    #
    recon_feat = recon_feat / recon_feat.sum(1, keepdim=True)


    return recon_feat

def compute_final_bag_rep(bag_rep, sbag_feat, b2sb_index):
    res = []
    for (idx,sb_idx) in enumerate(b2sb_index):
        res.append((bag_rep[idx] + sbag_feat[sb_idx]) / 2.0)
    return torch.stack(res)

def compute_final_superbag_rep(bag_rep, sbag_feat, sb2b_index):
    res = []
    for (idx, idxs) in enumerate(sb2b_index):
        cur_bag_feat = bag_rep[idxs]
        ave_feat = torch.mul(cur_bag_feat.sum(0), 1/len(idxs))
        res.append((ave_feat + sbag_feat[idx]) / 2.0)
    return torch.stack(res)

def get_cluster_labels(labels, sb2b_index, train_as_bag):
    if train_as_bag:
        return sb2b_index
    else:
        cluster_labels = []
        for sb in sb2b_index:
            sb_labels = []
            for b_idx in sb:
                sb_labels.append(labels[b_idx])
            # 求众数
            major_label = np.argmax(np.bincount(sb_labels))
            cluster_labels.append(major_label)
        return cluster_labels
def compute_final_assignment(bag_feat, superbag_feat):
    fianl_assignments = []
    for bag in bag_feat:
        distances = torch.pow(bag - superbag_feat, 2).sum(1)
        fianl_assignments.append(distances.argmin().item())
    return fianl_assignments


class BagCluster(BagRE):
    """
    Instance attention for bag-level relation extraction.
    """

    def __init__(self, bag_encoder, num_class, rel2id, cluster_num, num_iter):
        """
        Args:
            bag_encoder: encoder for bag
            num_class: number of classes
            id2rel: dictionary of id -> relation name mapping
            cluster_num: number of cluster
            num_iter: number of iteration
        """
        super().__init__()
        self.bag_encoder = bag_encoder
        self.num_class = num_class
        self.cluster_num = cluster_num
        self.num_iter = num_iter
        self.fc = nn.Linear(self.bag_encoder.sentence_encoder.hidden_size, num_class)
        self.softmax = nn.Softmax(-1)
        self.rel2id = rel2id
        self.id2rel = {}
        self.drop = nn.Dropout()
        self.task_criterion = nn.CrossEntropyLoss()
        self.recon_loss = nn.MSELoss()
        for rel, id in rel2id.items():
            self.id2rel[id] = rel



    
    def forward(self, labels, scope, args, train=True, bag_size=0, train_as_bag=True):
        """
        Args:
            bag_reps: (B, 3H) bag representations of a batch
            num_sbags: (int) A number of superbags 
        Return:
            cluster, (C, , 3H) C is the number of cluster, the cluster of bag.
        """
        bag_rep = self.bag_encoder(labels, scope, *args, bag_size=bag_size)
        sbag_feature, b2sb_index, sb2b_index = SbagFeature(bag_rep, labels, self.cluster_num, train_as_bag)
        origin_rep = sbag_feature.clone()
        for i in range(self.num_iter):
            sbag_feature, _ = exec_iter(sbag_feature, bag_rep, sb2b_index)
        final_bag_assoc = compute_assignments(sbag_feature, bag_rep, sb2b_index)
        cluster_labels = get_cluster_labels(labels, sb2b_index, train_as_bag)
        if train:
            new_sbag_feat = SpixelFeature2(bag_rep, final_bag_assoc, len(sb2b_index))
                # loss1: 计算重建损失（reconstruction loss)
            loss1 = self.recon_loss(origin_rep, new_sbag_feat)
                # # loss2: 计算紧致度损失
                
                # recon_feat = decode_features(final_bag_assoc, new_sbag_feat, b2sb_index)

                # compact_loss = CompactLoss()
                # loss2 = compact_loss(lable, )
                # loss3: 将超包特征赋予给原来的包，获取最后的bag_logits
            final_bag_feat = compute_final_bag_rep(bag_rep, new_sbag_feat, b2sb_index)
            final_superbag_feat = compute_final_superbag_rep(bag_rep, new_sbag_feat, sb2b_index)
            final_assignment = compute_final_assignment(final_bag_feat, final_superbag_feat)
            if train_as_bag:
                bag_logits = self.fc(final_bag_feat)
            else:
                bag_logits = self.fc(final_superbag_feat)
            
            return loss1, cluster_labels, bag_logits, final_assignment
                
        else:
            print('=== Testing ===')
            new_sbag_feat = SpixelFeature2(bag_rep, final_bag_assoc, len(sb2b_index))
            final_bag_feat = compute_final_bag_rep(bag_rep, new_sbag_feat, b2sb_index)
            bag_logits = self.fc(final_bag_feat)
            return bag_logits


            

