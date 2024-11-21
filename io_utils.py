import argparse
#import backbone

#model_dict = dict(ResNet10 = backbone.ResNet10)
#model_dict = dict(ResNet18 = backbone.ResNet18)

def str2bool(v):
    """
    Converts string to bool type; enables command line 
    arguments in the format of '--arg1 true --arg2 false'
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args(script):
    parser = argparse.ArgumentParser(description= 'few-shot script %s' %(script))
    parser.add_argument('--model'       , default='ResNet10',      help='backbone architecture')
    #parser.add_argument('--train_aug'   , action='store_true',  help='perform data augmentation or not during training ')
    #parser.add_argument('--freeze_backbone'   , action='store_true', help='Freeze the backbone network for finetuning')
    #parser.add_argument('--use_saved', action='store_true', help='Use the saved resources')
    parser.add_argument('--lamda', default=0.001, type=float)
    parser.add_argument('--beta', default=0.1, type=float)
    parser.add_argument('--k_lp', default=2, type=int)
    parser.add_argument('--delta', default=0.2, type=float)
    parser.add_argument('--alpha', default=0.5, type=float)
    #parser.add_argument("--model_type", choices=["ViT-B_16", "ViT-B_32", "ViT-L_16",
    #                                             "ViT-L_32", "ViT-H_14", "R50-ViT-B_16"],
    #                                             default="ViT-B_32",
    #                    help="Which variant to use.")
    #parser.add_argument("--root", type=str, default="", help="path to dataset")
    #parser.add_argument("--output-dir", type=str, default="", help="output directory")
    parser.add_argument("--resume", type=str, default="", help="checkpoint directory (from which the training resumes)",)
    parser.add_argument("--seed", type=int, default=-1, help="only positive value enables a fixed seed")
    #parser.add_argument("--source-domains", type=str, nargs="+", help="source domains for DA/DG")
    #parser.add_argument("--target-domains", type=str, nargs="+", help="target domains for DA/DG")
    parser.add_argument("--transforms", type=str, nargs="+", help="data augmentation methods")
    #parser.add_argument("--config-file", type=str, default="", help="path to config file")
    #parser.add_argument("--dataset-config-file", type=str, default="", help="path to config file for dataset setup", )
    #parser.add_argument("--trainer", type=str, default="", help="name of trainer")
    #parser.add_argument("--backbone", type=str, default="vit_b32", help="name of CNN backbone")
    #parser.add_argument("--head", type=str, default="", help="name of head")
    #parser.add_argument("--eval-only", action="store_true", help="evaluation only")
    #parser.add_argument("--model-dir", type=str, default="", help="load model from this directory for eval-only mode",    )
    #parser.add_argument("--load-epoch", type=int, help="load model weights at this epoch for evaluation"    )
    #parser.add_argument("--no-train", action="store_true", help="do not call trainer.train()"    )
    #parser.add_argument("opts", default=None, nargs=argparse.REMAINDER,
    #                    help="modify config options using the command-line",
    #                    )
                        
    #parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
    #parser.add_argument(
    #    "--opts",
    #    help="Modify config options by adding 'KEY VALUE' pairs. ",
    #    default=None,
    #    nargs='+',
    #)

    # easy config modification
    '''
    parser.add_argument('--batch-size', type=int, help="batch size for single GPU")
    parser.add_argument('--data-path', type=str, help='path to dataset')
    parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                        help='no: no cache, '
                             'full: cache all data, '
                             'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
    parser.add_argument('--pretrained',
                        help='pretrained weight from checkpoint, could be imagenet22k pretrained weight')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--disable_amp', action='store_true', help='Disable pytorch amp')
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    #parser.add_argument('--tag', default=time.strftime("%Y%m%d%H%M%S", time.localtime()), help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')

    parser.add_argument('--fused_layernorm', action='store_true', help='Use fused layernorm.')
    parser.add_argument('--optim', type=str, help='overwrite optimizer if provided, can be adamw/sgd.')
    '''

    # EMA related parameters
    #parser.add_argument('--model_ema', type=str2bool, default=True)
    #parser.add_argument('--model_ema_decay', type=float, default=0.9999, help='')
    #parser.add_argument('--model_ema_force_cpu', type=str2bool, default=False, help='')

    #parser.add_argument('--memory_limit_rate', type=float, default=-1, help='limitation of gpu memory use')

    parser.add_argument('--dtarget', default='CropDisease', choices=['CropDisease', 'EuroSAT', 'ISIC', 'ChestX', 'miniImageNet', 'Pattern'])
    parser.add_argument('--test_n_way', default=5, type=int, help='class num to classify for testing')
    parser.add_argument('--n_shot', default=5, type=int,
                        help='number of labeled data in each class, same as n_support')
    
    '''
    if script == 'train':
        parser.add_argument('--num_classes' , default=100, type=int, help='total number of classes in softmax, only used in baseline') #make it larger than the maximum label value in base class
        parser.add_argument('--save_freq'   , default=50, type=int, help='Save frequency')
        parser.add_argument('--start_epoch' , default=0, type=int,help ='Starting epoch')
        parser.add_argument('--stop_epoch'  , default=4000, type=int, help ='Stopping epoch')
    elif script == 'finetune':
        parser.add_argument('--dtarget', default='CropDisease', choices=['CropDisease', 'EuroSAT', 'ISIC', 'ChestX', 'miniImageNet', 'Pattern'])
        parser.add_argument('--test_n_way', default=5, type=int, help='class num to classify for testing')
        parser.add_argument('--n_shot', default=5, type=int,
                            help='number of labeled data in each class, same as n_support')
        
    else:
       raise ValueError('Unknown script')
    '''
        
    return parser.parse_args()
