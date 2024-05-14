def get_args():
    parser = argparse.ArgumentParser(description="TransTextReID")

    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument("--name", default="baseline", help="experiment name to save")
    parser.add_argument("--output_dir", default="E:\\Savemodel\\CMRP")
    parser.add_argument("--log_period", default=100)
    parser.add_argument("--eval_period", default=1)
    parser.add_argument("--val_dataset", default="test")
    parser.add_argument("--resume", default=False, action='store_true')
    parser.add_argument("--resume_ckpt_file", default="E:\\Savemodel\best.pth", help='resume from ...')

    parser.add_argument("--pretrain_choice", default='ViT-B/16')
    parser.add_argument("--temperature", type=float, default=0.03,
                        help="initial temperature value, if 0, don't use temperature,best 0.02")
    parser.add_argument("--temp_Rap", type=float, default=0.01,
                        help="Rap temp")
    parser.add_argument("--img_aug", default=False, action='store_true')

    parser.add_argument("--cmt_depth", type=int, default=4, help="cross modal transformer self attn layers")
    parser.add_argument("--masked_token_rate", type=float, default=0.8, help="masked token rate for mlm task")
    parser.add_argument("--masked_token_unchanged_rate", type=float, default=0.1, help="masked token unchanged rate")
    parser.add_argument("--lr_factor", type=float, default=5.0, help="lr factor for random init self implement module")
    parser.add_argument("--MLM", default=False, action='store_true',
                        help="whether to use Mask Language Modeling dataset")

    parser.add_argument("--loss_names", default='itc',
                        help="which loss to use ['mlm', 'cmpm', 'id', 'itc', 'sdm']")
    parser.add_argument("--mlm_loss_weight", type=float, default=1.0, help="mlm loss weight")
    parser.add_argument("--id_loss_weight", type=float, default=1.0, help="id loss weight")

    parser.add_argument("--itc_loss_weight", type=float, default=1, help="itc loss weight")
    parser.add_argument("--xEnt_loss_weight", type=float, default=1.4, help="xEnt loss weight")
    parser.add_argument("--rap_loss_weight", type=float, default=1.2, help="Rap loss weight")

    parser.add_argument("--img_size", type=tuple, default=(384, 128))
    parser.add_argument("--stride_size", type=int, default=16)

    parser.add_argument("--text_length", type=int, default=77)
    parser.add_argument("--vocab_size", type=int, default=49408)
    parser.add_argument("--queue_size", type=int, default=49408)

    parser.add_argument("--optimizer", type=str, default="Adam", help="[SGD, Adam, AdamW,LAMB]")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--bias_lr_factor", type=float, default=2.)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=4e-5)
    parser.add_argument("--weight_decay_bias", type=float, default=0.)
    parser.add_argument("--alpha", type=float, default=0.9)
    parser.add_argument("--beta", type=float, default=0.999)

    parser.add_argument("--num_epoch", type=int, default=60)
    parser.add_argument("--milestones", type=int, nargs='+', default=(20, 50))
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--warmup_factor", type=float, default=0.1)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--warmup_method", type=str, default="linear")
    parser.add_argument("--lrscheduler", type=str, default="cosine")
    parser.add_argument("--target_lr", type=float, default=0)
    parser.add_argument("--power", type=float, default=0.9)

    parser.add_argument("--dataset_name", default="CUHK-PEDES", help="[CUHK-PEDES, ICFG-PEDES, RSTPReid]")
    parser.add_argument("--sampler", default="random", help="choose sampler from [idtentity, random]")
    parser.add_argument("--root_dir", default="./data")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--test_batch_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--test", dest='training', default=True, action='store_false')

    parser.add_argument("--alpha_Rap", type=float, default=0.4)
    parser.add_argument("--rap_lambd", type=float, default=1)
    parser.add_argument("--rap_bt", type=float, default=4)
    parser.add_argument("--plan_f", default="na", help="all,na")
    parser.add_argument("--mask", default=True, action='store_true')
    parser.add_argument("--token_type",  default="tt", help="text,tt")
    parser.add_argument("--xiaorong",  default="I+C", help="I，C，I+C")
    parser.add_argument("--softmax",  default="true", help="true-softmax，false-softmax")

    parser.add_argument('--augc', default=0, type=int,
                        metavar='aug', help='use channel aug 1 or not')
    parser.add_argument('--rande', default=0, type=float,
                        metavar='ra', help='use random erasing 0.5 or not and the probability')

    parser.add_argument("--l2", default=False, action='store_true')




    args = parser.parse_args()

    return args









import argparse
