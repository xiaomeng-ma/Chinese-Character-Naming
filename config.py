import os
import argparse


def get_args():
    parser = argparse.ArgumentParser(description='Text Classification: sentiment analysis')

    # random seed settings
    parser.add_argument('-seed', type=int, default=1,
                        help='random seed choices for reproducibility')

    # data path
    parser.add_argument('-data_path', type=str, default='Data/',
                        help='data path')
    # choose add pinyin in the input
    parser.add_argument('-pinyin', type = str, default = 'no', choices = ['no', 'add'],
                        help = "no: Model[-pinyin], add:Model[+pinyin]")
    # add label/no_label
    parser.add_argument('-label_spec', type=str, default='base', choices=['base', 's', 'sboth', 'm', 'mboth'],
                        help = 'base: no lable, s: label_s, sboth: label_sr, m: lable_m, mboth: label_mr')

    # output tone or no tone
    parser.add_argument('-tone_spec', type=str, default = 'notone', choices = ['notone','tone'])

    # shuffle order of consonant and vowel
    parser.add_argument('-shuffle_spec', type=str, default = 'noshuffle', choices = ['noshuffle','shuffle'])

    # use another training & dev data filtered by frequency
    parser.add_argument('-freq_range', type=str, default='all', choices=['all', 'high', 'mid'],
                        help='train with high frequency data')
    # add feature
    parser.add_argument('-feature_spec', type = str, default = 'no', choices = ['no', 'add_freq'])

    # select top k
    parser.add_argument('-vk', type=int, default=3,
                        help='set k value')

    # model setting
    parser.add_argument('-model_path', type=str, default='Model/',
                        help='data path')
    parser.add_argument('-num_heads', type=int, default=4, metavar='N',
                        help='number of heads')
    parser.add_argument('-d_model', type=int, default=128, metavar='N',
                        help='the number of expected features in the encoder/decoder inputs')
    parser.add_argument('-dff', type=int, default=256, metavar='N',
                        help='the dimension of the feedforward network model')

    
    # learning
    parser.add_argument('-nlayers', type=int, default=2, help='number of layers')
    parser.add_argument('-dropout', type=float, default=0.1,
                        help='dropout applied to layers (0 = no dropout)')
    return parser.parse_args()


def process_args(args):
    suffix_dir = str(args.seed) + args.pinyin + '_' + args.freq_range + '_' + args.label_spec + '_' + args.tone_spec + '_' + args.shuffle_spec + '_' + args.feature_spec
    args.model_path = os.path.join(args.model_path, suffix_dir)
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path, exist_ok=True)
    return args