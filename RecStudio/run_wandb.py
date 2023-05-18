import os
import time
from recstudio.utils import *
from recstudio import quickstart
import wandb 
from argparse import ArgumentParser
from functools import partial

def wandb_run(args, model_conf):

    wandb.init(project=model_conf['train']['wandb_project'])
    model_conf['train']['wandb_project'] = model_conf['train']['wandb_project']
    model_conf['train']['use_wandb'] = True 
    model_conf = deep_update(model_conf, wandb.config)

    quickstart.kdd_cup_run(args.model, args.dataset, args, model_config=model_conf, data_config_path=args.data_config_path, 
                           do_prediction=args.do_prediction, do_evaluation=args.do_evaluation, model_path=args.model_path)

if __name__ == '__main__':
    
    # quickstart.run(args.model, args.dataset, model_config=model_conf, data_config_path=args.data_config_path)
    wandb.login()

    parser = get_default_parser()
    group = parser.add_argument_group('main')
    # group.add_argument('--wandb_project', type=str, default='tune_BPR', help='wandb project name')
    # group.add_argument('--wandb_count', type=int, default=2, help='the maximum number of tries in wandb')

    args, command_line_args = parser.parse_known_args()
    parser = add_model_arguments(parser, args.model)
    command_line_conf = parser2nested_dict(parser, command_line_args)

    model_class, model_conf = get_model(args.model)
    model_conf = deep_update(model_conf, command_line_conf)


    sweep_configuration = {
        'method': 'bayes',
        'metric': {
            'goal': 'maximize', 
            'name': 'mrr@100.max'
        },
        'parameters': 
        {   
            'train': {
                'parameters':{
                    'learning_rate': {'max' : 0.001, 'min' : 0.0001},
                    'scheduler_patience': {'values': [1, 2]}
                }
            },
            'model': {
                'parameters':{
                    'layer_num': {'values': [3, 4, 2]},
                    'head_num': {'values': [2, 4, 6, 1, 3, 8, 10]},
                    'hidden_size': {'values': [128, 256, 384, 512, 64]},
                    'embed_dim': {'values': [120, 128, 180, 192, 250, 256, 360]},
                    'dropout_rate': {'values': [0.1, 0.25, 0.3, 0.5, 0.6, 0.75, 0.85]},
                    
                }
            }   
        },
        'early_terminate': {
            'type' : 'hyperband', 
            'min_iter' : 50,
            'eta': 2
        }
    }

    sweep_id = wandb.sweep(
        sweep=sweep_configuration, 
        project=model_conf['train']['wandb_project']
    )
    
    wandb.agent(sweep_id, function=partial(wandb_run, args, model_conf), count=model_conf['train']['wandb_count'])

