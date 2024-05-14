import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# import jax
# import jax.numpy as jnp
from timm.loss import SoftTargetCrossEntropy

def compute_sdm(image_fetures, text_fetures, pid, logit_scale, image_id=None, factor=0.3, epsilon=1e-8):
    """
    Similarity Distribution Matching
    """
    batch_size = image_fetures.shape[0]
    pid = pid.reshape((batch_size, 1))  # make sure pid size is [batch_size, 1]
    pid_dist = pid - pid.t()
    labels = (pid_dist == 0).float()

    if image_id != None:
        # print("Mix PID and ImageID to create soft label.")
        image_id = image_id.reshape((-1, 1))
        image_id_dist = image_id - image_id.t()
        image_id_mask = (image_id_dist == 0).float()
        labels = (labels - image_id_mask) * factor + image_id_mask
        # labels = (labels + image_id_mask) / 2

    image_norm = image_fetures / image_fetures.norm(dim=-1, keepdim=True)
    text_norm = text_fetures / text_fetures.norm(dim=1, keepdim=True)

    t2i_cosine_theta = text_norm @ image_norm.t()
    # t2i_cosine_theta = text_norm @ image_norm.transpose(0,1)
    i2t_cosine_theta = t2i_cosine_theta.t()

    text_proj_image = logit_scale * t2i_cosine_theta
    image_proj_text = logit_scale * i2t_cosine_theta

    # normalize the true matching distribution
    labels_distribute = labels / labels.sum(dim=1)

    i2t_pred = F.softmax(image_proj_text, dim=1)
    i2t_loss = i2t_pred * (F.log_softmax(image_proj_text.cuda(), dim=1) - torch.log(labels_distribute.cuda() + epsilon))
    t2i_pred = F.softmax(text_proj_image, dim=1)
    t2i_loss = t2i_pred * (F.log_softmax(text_proj_image.cuda(), dim=1) - torch.log(labels_distribute.cuda() + epsilon))

    loss = torch.mean(torch.sum(i2t_loss, dim=1)) + torch.mean(torch.sum(t2i_loss, dim=1))

    return loss


def compute_mlm(scores, labels):
    ce = nn.CrossEntropyLoss(ignore_index=0)
    return ce(scores, labels)


def compute_itc(i_feats_clsures, t_feats_clsures, logit_scale):
    """
    image-text contrastive (ITC) loss, InfoNCE
    """
    batch_size = i_feats_clsures.shape[0]
    labels = torch.arange(start=0, end=batch_size, dtype=torch.int64)
    labels = labels.to(i_feats_clsures.device)

    # normalized features
    image_norm = i_feats_clsures / i_feats_clsures.norm(dim=-1, keepdim=True)
    text_norm = t_feats_clsures / t_feats_clsures.norm(dim=-1, keepdim=True)

    # cosine similarity as logits
    logits_per_image = logit_scale * image_norm @ text_norm.t()
    # logits_per_image = logit_scale * image_norm @ text_norm.t()
    logits_per_text = logits_per_image.t()

    loss_i = F.cross_entropy(logits_per_image, labels)
    loss_t = F.cross_entropy(logits_per_text, labels)
    loss = (loss_i + loss_t) / 2

    return loss


def compute_id(image_logits, text_logits, labels):
    """
    Instance loss proposed at http://arxiv.org/abs/1711.05535
    """
    criterion = nn.CrossEntropyLoss(reduction="mean")

    loss = criterion(image_logits, labels) + criterion(text_logits, labels)

    return loss / 2


def compute_cmpm(image_embeddings, text_embeddings, labels, epsilon=1e-8):

    batch_size = image_embeddings.shape[0]
    labels_reshape = torch.reshape(labels, (batch_size, 1))
    labels_dist = labels_reshape - labels_reshape.t()
    labels_mask = (labels_dist == 0).float()

    image_norm = image_embeddings / image_embeddings.norm(dim=1, keepdim=True)
    text_norm = text_embeddings / text_embeddings.norm(dim=1, keepdim=True)
    image_proj_text = torch.matmul(image_embeddings, text_norm.t())
    # text_proj_image = torch.matmul(text_embeddings, image_norm.t())
    text_proj_image = torch.matmul(text_embeddings, image_norm.transpose(0, 1))

    # normalize the true matching distribution
    labels_mask_norm = labels_mask / labels_mask.norm(dim=1)

    i2t_pred = F.softmax(image_proj_text, dim=1)
    i2t_loss = i2t_pred * (F.log_softmax(image_proj_text, dim=1) - torch.log(labels_mask_norm + epsilon))
    t2i_pred = F.softmax(text_proj_image, dim=1)
    t2i_loss = t2i_pred * (F.log_softmax(text_proj_image, dim=1) - torch.log(labels_mask_norm + epsilon))

    cmpm_loss = torch.mean(torch.sum(i2t_loss, dim=1)) + torch.mean(torch.sum(t2i_loss, dim=1))

    return cmpm_loss


######################   compute_rap  ######################################
def compute_rap(self, i_feats, i_feats_cls, i_feats_patch, t_feats,
                t_feats_cls, t_feats_token, alpha, logit_scale, t_mask=None):
    with torch.no_grad():
        self.temp_Rap.clamp_(0.001, 0.5)
    rap_bt = self.rap_bt  # 4
    rap_lambd = self.rap_lambd  # 1
    i_feat_all = torch.cat([i_feats_cls.t(), self.image_queue.clone().detach()], dim=1)
    t_feat_all = torch.cat([t_feats_cls.t(), self.text_queue.clone().detach()], dim=1)

    if self.args.plan_f == "all":

        i_feat_all = i_feats.permute(2,0,1).reshape(512,-1)
        t_feat_all = t_feats.permute(2,0,1).reshape(512,-1)

        expand_dim_i = 14000-i_feat_all.size()[1]
        expand_dim_t = 14000-t_feat_all.size()[1]
        i_feat_all = torch.cat([i_feat_all, torch.zeros(512, expand_dim_i).cuda()], dim=1)
        t_feat_all = torch.cat([t_feat_all, torch.zeros(512, expand_dim_t).cuda()], dim=1)

    if self.args.softmax == "false":
        sim_i2i = i_feats_cls @ i_feat_all / self.temp_Rap
        sim_t2t = t_feats_cls @ t_feat_all / self.temp_Rap
    else:
        sim_i2i = F.softmax(i_feats_cls, dim=1) @ F.softmax(i_feat_all, dim=1) / self.temp_Rap
        sim_t2t = F.softmax(t_feats_cls, dim=1) @ F.softmax(t_feat_all, dim=1) / self.temp_Rap

        i_feats_cls = i_feats_cls.cuda()
        i_feat_all = i_feat_all.cuda()
        t_feats_cls = t_feats_cls.cuda()
        t_feat_all = t_feat_all.cuda()

        i_feats_cls = torch.mul(i_feats_cls, F.softmax(i_feats_cls, dim=1))
        i_feat_all = torch.mul(i_feat_all, F.softmax(i_feat_all, dim=1))
        t_feats_cls = torch.mul(t_feats_cls, F.softmax(t_feats_cls, dim=1))
        t_feat_all = torch.mul(t_feat_all, F.softmax(t_feat_all, dim=1))

    sim_i2t_m = i_feats_cls @ t_feat_all / self.temp_Rap
    sim_t2i_m = t_feats_cls @ i_feat_all / self.temp_Rap

    sim_targets = torch.zeros(sim_i2t_m.size()).to(i_feats_cls.device)
    sim_targets.fill_diagonal_(1)

    sim_i2t_targets = alpha * F.softmax(sim_i2t_m, dim=1) + (1 - alpha) * sim_targets
    sim_t2i_targets = alpha * F.softmax(sim_t2i_m, dim=1) + (1 - alpha) * sim_targets
    sim_i2t = i_feats_cls @ t_feat_all / self.temp_Rap
    sim_t2i = t_feats_cls @ i_feat_all / self.temp_Rap
    loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1) * sim_i2t_targets, dim=1).mean()
    loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1) * sim_t2i_targets, dim=1).mean()



    sim_local_t2i = torch.bmm(t_feats_token, i_feats_patch.permute(0, 2, 1)) / self.temp_Rap
    sim_local_i2t = torch.bmm(i_feats_patch, t_feats_token.permute(0, 2, 1)) / self.temp_Rap

    sim_local_t2i_max = torch.nn.Softmax(dim=-1)(torch.max(sim_local_t2i, dim=-1).values)
    sim_local_i2t_max = torch.nn.Softmax(dim=-1)(torch.max(sim_local_i2t, dim=-1).values)

    loss_t2i_crosMod_l = redundancy_weight(t_feats_token, i_feats_cls, self.temp_Rap, attention_mask=t_mask,
                                           sim=sim_local_t2i_max)

    loss_i2t_crosMod_l = redundancy_weight(i_feats_patch, t_feats_cls, self.temp_Rap, attention_mask=None,
                                           sim=sim_local_i2t_max)

    loss_i2i = -torch.sum(F.log_softmax(sim_i2i, dim=1) * sim_targets, dim=1).mean()  # 模态内
    loss_t2t = -torch.sum(F.log_softmax(sim_t2t, dim=1) * sim_targets, dim=1).mean()

    if rap_bt != 0:
        if self.args.xiaorong == "IREM":
            loss_ita = (loss_i2i + loss_t2t) / rap_bt
        elif self.args.xiaorong == "CREM":
            loss_ita = (loss_t2i_crosMod_l + loss_i2t_crosMod_l) * rap_lambd
        else:
            loss_ita = (loss_i2i + loss_t2t) / rap_bt + (
                    loss_t2i_crosMod_l + loss_i2t_crosMod_l) * rap_lambd
    else:
        loss_ita = (loss_t2i_crosMod_l + loss_i2t_crosMod_l) * rap_lambd
    return loss_ita


def redundancy_weight(l, m, temp, attention_mask=None, sim=None):
    m = m.unsqueeze(1)
    N, n_locals, dim = l.size()
    u_p = torch.matmul(l, m.permute(0, 2, 1)).unsqueeze(2) / temp
    if attention_mask is not None:
        temp_mask = attention_mask.unsqueeze(2).unsqueeze(3).to(l.device)
        u_p = (temp_mask * u_p) + (10000. * (1 - temp_mask))

    l_n = l.reshape(-1, dim)
    m_n = m.reshape(-1, dim)
    u_n = torch.mm(m_n, l_n.t()) / temp
    u_n = u_n.reshape(N, 1, N, n_locals).permute(0, 2, 3, 1)
    mask = torch.eye(N)[:, :, None, None].to(l.device)
    n_mask = 1 - mask
    u_n = (n_mask * u_n) - (10000. * (1 - n_mask))
    if attention_mask is not None:
        temp_mask = attention_mask.unsqueeze(0).unsqueeze(3).expand(N, -1, -1, -1).to(l.device)
        u_n = (temp_mask * u_n) - (10000. * (1 - temp_mask)).to(l.device)

    u_n = u_n.reshape(N, N * n_locals, 1).unsqueeze(dim=1).expand(-1, n_locals, -1, -1)
    pred_lgt = torch.cat([u_p, u_n], dim=2)
    pred_log = F.log_softmax(pred_lgt, dim=2)
    if attention_mask is not None:
        loss = (torch.sum(-(pred_log[:, :, 0].squeeze() * sim), dim=1) / torch.sum(attention_mask.to(l.device),
                                                                                   dim=1)).mean()
    else:
        loss = -(pred_log[:, :, 0].squeeze() * sim).mean()

    return loss


def compute_NT_XentLoss(z1, z2, temperature=0.5):
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    N, Z = z1.shape
    device = z1.device
    representations = torch.cat([z1, z2], dim=0)
    similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=-1)
    l_pos = torch.diag(similarity_matrix, N)
    r_pos = torch.diag(similarity_matrix, -N)
    positives = torch.cat([l_pos, r_pos]).view(2 * N, 1)
    diag = torch.eye(2 * N, dtype=torch.bool, device=device)
    diag[N:, :N] = diag[:N, N:] = diag[:N, :N]

    negatives = similarity_matrix[~diag].view(2 * N, -1)

    logits = torch.cat([positives, negatives], dim=1)
    logits /= temperature

    labels = torch.zeros(2 * N, device=device, dtype=torch.int64)

    loss = F.cross_entropy(logits, labels, reduction='sum')
    return loss / (2 * N)

