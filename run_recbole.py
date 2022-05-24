# @Time   : 2020/7/20
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn

# UPDATE
# @Time   : 2020/10/3, 2020/10/1
# @Author : Yupeng Hou, Zihan Lin
# @Email  : houyupeng@ruc.edu.cn, zhlin@ruc.edu.cn


import argparse
import time
from recbole.quick_start import run_recbole


if __name__ == '__main__':
    begin = time.time()
    parameter_dict = {
        'neg_sampling': None
        # 'gpu_id':3,
        # 'attribute_predictor':'not',
        # 'attribute_hidden_size':"[256]",
        # 'fusion_type':'gate',
        # 'seed':212,
        # 'n_layers':4,
        # 'n_heads':1
    }
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='SASRecD', help='name of models')
    parser.add_argument('--dataset', '-d', type=str, default='Amazon_Beauty', help='name of datasets')
    parser.add_argument('--config_files', type=str, default='configs/Amazon_Beauty.yaml', help='config files')

    args, _ = parser.parse_known_args()

    config_file_list = args.config_files.strip().split(' ') if args.config_files else None
    run_recbole(model=args.model, dataset=args.dataset, config_file_list=config_file_list,config_dict=parameter_dict)
    end=time.time()
    print(end-begin)
