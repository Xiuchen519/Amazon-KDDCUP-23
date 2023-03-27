import os
import time
from recstudio.utils import *
from recstudio import quickstart


if __name__ == '__main__':
    parser = get_default_parser()
    args, command_line_args = parser.parse_known_args()
    parser = add_model_arguments(parser, args.model)
    command_line_conf = parser2nested_dict(parser, command_line_args)

    model_class, model_conf = get_model(args.model)
    model_conf = deep_update(model_conf, command_line_conf)

    quickstart.kdd_cup_run(args.model, args.dataset, model_config=model_conf, data_config_path=args.data_config_path, 
                           do_prediction=args.do_prediction, model_path=args.model_path)
    
    # quickstart.run(args.model, args.dataset, model_config=model_conf, data_config_path=args.data_config_path)
