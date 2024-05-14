from collections import OrderedDict

import torch
import torch.nn as nn
from torchinfo import summary

from datasets.ReID.make_model_clipreid import PromptLearner, TextEncoder
from model import objectives
from .clip_model import Transformer, QuickGELU, LayerNorm, build_CLIP_from_openai_pretrained, convert_weights
from .supcontrast import SupConLoss
import numpy as np

class IRRA(nn.Module):
    def __init__(self, args, num_classes=11003):
        super().__init__()
        self.args = args
        self.num_classes = num_classes
        self._set_task()

        self.base_model, base_cfg = build_CLIP_from_openai_pretrained(args.pretrain_choice, args.img_size,
                                                                      args.stride_size)

        if 'reid' in args.loss_names:
            """----------------------------------------------ReID---------------------------------------------------------------------------"""
            self.prompt_learner = PromptLearner(num_classes, args.dataset_name, self.base_model.dtype,
                                                self.base_model.token_embedding)
            self.text_encoder = TextEncoder(self.base_model)
            """----------------------------------------------------------------------------------------------------------------------------"""

        self.embed_dim = base_cfg['embed_dim']

        self.logit_scale = torch.ones([]) * (1 / args.temperature)
        self.logit_scale_sdm = torch.ones([]) * (1 / 0.02)

        '''CYCLIP'''
        self.logit_scale_CY = nn.Parameter(torch.ones([]))
        nn.init.constant_(self.logit_scale_CY, np.log(1 / 0.07))
        logit_scale_CY = self.logit_scale_CY.exp()
        logit_scale_CY.data = torch.clamp(logit_scale_CY.data, max=100)

        '''uni-loss'''
        self.logit_scale_uni = nn.Parameter(torch.ones([]))



        if 'id' in args.loss_names:
            self.classifier = nn.Linear(self.embed_dim, self.num_classes)
            nn.init.normal_(self.classifier.weight.data, std=0.001)
            nn.init.constant_(self.classifier.bias.data, val=0.0)

        if 'mlm' in args.loss_names:
            self.cross_attn = nn.MultiheadAttention(self.embed_dim,
                                                    self.embed_dim // 64,
                                                    batch_first=True)
            self.cross_modal_transformer = Transformer(width=self.embed_dim,
                                                       layers=args.cmt_depth,
                                                       heads=self.embed_dim //
                                                             64)
            scale = self.cross_modal_transformer.width ** -0.5

            self.ln_pre_t = LayerNorm(self.embed_dim)
            self.ln_pre_i = LayerNorm(self.embed_dim)
            self.ln_post = LayerNorm(self.embed_dim)

            proj_std = scale * ((2 * self.cross_modal_transformer.layers) ** -0.5)
            attn_std = scale
            fc_std = (2 * self.cross_modal_transformer.width) ** -0.5
            for block in self.cross_modal_transformer.resblocks:
                nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
                nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
                nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
                nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

            # init cross attn
            nn.init.normal_(self.cross_attn.in_proj_weight, std=attn_std)
            nn.init.normal_(self.cross_attn.out_proj.weight, std=proj_std)

            self.mlm_head = nn.Sequential(
                OrderedDict([('dense', nn.Linear(self.embed_dim, self.embed_dim)),
                             ('gelu', QuickGELU()),
                             ('ln', LayerNorm(self.embed_dim)),
                             ('fc', nn.Linear(self.embed_dim, args.vocab_size))]))
            # init mlm head
            nn.init.normal_(self.mlm_head.dense.weight, std=fc_std)
            nn.init.normal_(self.mlm_head.fc.weight, std=proj_std)
        convert_weights(self.base_model)

        if 'rap' in args.loss_names:
            self.register_buffer("image_queue", torch.randn(self.embed_dim, self.args.queue_size))
            self.register_buffer("text_queue", torch.randn(self.embed_dim, self.args.queue_size))
            self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
            self.image_queue = nn.functional.normalize(self.image_queue, dim=0)
            self.text_queue = nn.functional.normalize(self.text_queue, dim=0)
            self.temp_Rap = nn.Parameter(args.temp_Rap * torch.ones([]))
            self.rap_lambd = args.rap_lambd
            self.rap_bt = args.rap_bt

    def _set_task(self):
        loss_names = self.args.loss_names
        self.current_task = [l.strip() for l in loss_names.split('+')]
        print(f'Training Model with {self.current_task} tasks')

    def cross_former(self, q, k, v):
        x = self.cross_attn(
            self.ln_pre_t(q),
            self.ln_pre_i(k),
            self.ln_pre_i(v),
            need_weights=False)[0]
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.cross_modal_transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x)
        return x

    def encode_image(self, image):
        x = self.base_model.encode_image(image)
        return x[:, 0, :].float()


    def encode_text(self, text):
        x = self.base_model.encode_text(text)
        return x[torch.arange(x.shape[0]), text.argmax(dim=-1)].float()

    def forward(self, batch, target, alpha):
        device = "cuda"

        ret = dict()

        images = batch['images'].to(device)
        """caption_id : token"""
        caption_ids = batch['caption_ids'].to(device)

        if 'rap' in self.current_task:
            i_features_all, i_features_cls, i_features_patch = self.base_model.encode_image_Rap(images)
            t_features_all, t_features_cls, t_features_token,t_mask = self.base_model.encode_text_Rap(self.args.token_type,caption_ids)
            i_features_all.float().to(device),
            i_features_cls.float().to(device),
            i_features_patch.float().to(device),
            t_features_all.float().to(device),
            t_features_cls.float().to(device),
            t_features_token.float().to(device)
            i_feats = i_features_cls.float().to(device)
            t_feats = t_features_cls.float().to(device)
            t_mask.float().to(device)
        else:
            image_feats = self.base_model.encode_image(images)
            i_feats = image_feats[:, 0, :].float()
            text_feats = self.base_model.encode_text(caption_ids)
            t_feats = text_feats[torch.arange(text_feats.shape[0]), caption_ids.argmax(dim=-1)].float()


        logit_scale = self.logit_scale
        logit_scale_sdm = self.logit_scale_sdm
        ret.update({'temperature': 1 / logit_scale})

        '''CYCLIP'''
        logit_scale_CY = self.logit_scale_CY

        '''uniCL-loss'''
        T = self.logit_scale_uni.exp()

        "--------------------------------------------------------------------------------------------------------------------------"
        if 'reid' in self.current_task:
            """ReID"""
            xent = SupConLoss(device)
            prompts = self.prompt_learner(target)
            tokenized_prompts = self.prompt_learner.tokenized_prompts.cuda()
            tokenized_prompts = torch.tensor(tokenized_prompts, dtype=torch.float16)
            rt_feats = self.text_encoder(prompts, tokenized_prompts)
            loss_i2t = xent(i_feats, rt_feats, target, target)
            loss_t2i = xent(rt_feats, i_feats, target, target)
            reid_loss = loss_i2t + loss_t2i
            ret.update({'reid_loss': reid_loss * 1})
        "--------------------------------------------------------------------------------------------------------------------------"

        if 'itc' in self.current_task:
            ret.update({'itc_loss': objectives.compute_itc(i_feats, t_feats, logit_scale)*self.args.itc_loss_weight})

        if 'sdm' in self.current_task:
            ret.update({'sdm_loss': objectives.compute_sdm(i_feats, t_feats, batch['pids'], logit_scale_sdm)})

        if 'cmpm' in self.current_task:
            ret.update({'cmpm_loss': objectives.compute_cmpm(i_feats, t_feats, batch['pids'])})

        if 'id' in self.current_task:

            image_logits = self.classifier(i_feats).float()
            text_logits = self.classifier(t_feats).float()
            ret.update(
                {'id_loss': objectives.compute_id(image_logits, text_logits,
                                                  batch['pids'].to(device)) * self.args.id_loss_weight})

            image_pred = torch.argmax(image_logits, dim=1)
            text_pred = torch.argmax(text_logits, dim=1)

            image_precision = (image_pred == batch['pids'].to(device)).float().mean()
            text_precision = (text_pred == batch['pids'].to(device)).float().mean()
            ret.update({'img_acc': image_precision})
            ret.update({'txt_acc': text_precision})

        if 'mlm' in self.current_task:
            mlm_ids = batch['mlm_ids'].to(device)

            mlm_feats = self.base_model.encode_text(mlm_ids)

            x = self.cross_former(mlm_feats, image_feats, image_feats)

            x = self.mlm_head(x)  # [batch_size, text_len, num_colors]

            scores = x.float().reshape(-1, self.args.vocab_size)
            mlm_labels = batch['mlm_labels'].reshape(-1)
            ret.update({'mlm_loss': objectives.compute_mlm(scores, mlm_labels) * self.args.mlm_loss_weight})

            pred = scores.max(1)[1]
            mlm_label_idx = torch.nonzero(mlm_labels)
            acc = (pred[mlm_label_idx] == mlm_labels[mlm_label_idx]).float().mean()
            ret.update({'mlm_acc': acc})

        if 'xent' in self.current_task:
            ret.update({'xEnt_loss': objectives.compute_NT_XentLoss(i_feats, t_feats,
                                                      temperature=self.args.temperature) * self.args.xEnt_loss_weight})


        if 'rap' in self.current_task:

            if self.args.mask:
                ret.update({'rap_loss': objectives.compute_rap(self, i_features_all.float().to(device),
                                                               i_features_cls.float().to(device),
                                                               i_features_patch.float().to(device),
                                                               t_features_all.float().to(device),
                                                               t_features_cls.float().to(device),
                                                               t_features_token.float().to(device), alpha,
                                                               logit_scale,t_mask) * self.args.rap_loss_weight})
            else:
                ret.update({'rap_loss': objectives.compute_rap(self, i_features_all.float().to(device),
                                                               i_features_cls.float().to(device),
                                                               i_features_patch.float().to(device),
                                                               t_features_all.float().to(device),
                                                               t_features_cls.float().to(device),
                                                               t_features_token.float().to(device), alpha,
                                                               logit_scale) * self.args.rap_loss_weight})


        return ret


def build_model(args, num_classes=11003):
    model = IRRA(args, num_classes)
    summary(model)
    # covert model to fp16
    return model
