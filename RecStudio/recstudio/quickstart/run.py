import os, time
from typing import *
from recstudio.utils import *
import logging 
import pandas as pd 

def run(model: str, dataset: str, model_config: Dict=None, data_config: Dict=None, model_config_path: str=None, data_config_path: str=None, verbose=True, **kwargs):
    model_class, model_conf = get_model(model)

    if model_config_path is not None:
        if isinstance(model_config_path, str):
            model_conf = deep_update(model_conf, parser_yaml(model_config_path))
        else:
            raise TypeError(f"expecting `model_config_path` to be str, while get {type(model_config_path)} instead.")

    if model_config is not None:
        if isinstance(model_config, Dict):
            model_conf = deep_update(model_conf, model_config)
        else:
            raise TypeError(f"expecting `model_config` to be Dict, while get {type(model_config)} instead.")

    if kwargs is not None:
        model_conf = deep_update(model_conf, kwargs)

    log_path = time.strftime(f"{model}/{dataset}/%Y-%m-%d-%H-%M-%S.log", time.localtime())
    logger = get_logger(log_path)

    if not verbose:
        import logging
        logger.setLevel(logging.ERROR)

    logger.info("Log saved in {}.".format(os.path.abspath(log_path)))
    model = model_class(model_conf)
    dataset_class = model_class._get_dataset_class()

    data_conf = {}
    if data_config_path is not None:
        if isinstance(data_config_path, str):
            # load dataset config from file
            conf = parser_yaml(data_config)
            data_conf.update(conf)
        else:
            raise TypeError(f"expecting `data_config_path` to be str, while get {type(data_config_path)} instead.")

    if data_config is not None:
        if isinstance(data_config, dict):
            # update config with given dict
            data_conf.update(data_config)
        else:
            raise TypeError(f"expecting `data_config` to be Dict, while get {type(data_config)} instead.")

    data_conf.update(model_conf['data'])    # update model-specified config

    datasets = dataset_class(name=dataset, config=data_conf).build(**model_conf['data'])
    logger.info(f"{datasets[0]}")
    logger.info(f"\n{set_color('Model Config', 'green')}: \n\n" + color_dict_normal(model_conf, False))
    model.fit(*datasets[:2], run_mode='light')
    model.evaluate(datasets[-1])


def kdd_cup_run(model: str, dataset: str, model_config: Dict=None, data_config: Dict=None, model_config_path: str=None, data_config_path: str=None, verbose=True,
                do_prediction=False, model_path=None, **kwargs):
    model_class, model_conf = get_model(model)

    if model_config_path is not None:
        if isinstance(model_config_path, str):
            model_conf = deep_update(model_conf, parser_yaml(model_config_path))
        else:
            raise TypeError(f"expecting `model_config_path` to be str, while get {type(model_config_path)} instead.")

    if model_config is not None:
        if isinstance(model_config, Dict):
            model_conf = deep_update(model_conf, model_config)
        else:
            raise TypeError(f"expecting `model_config` to be Dict, while get {type(model_config)} instead.")

    if kwargs is not None:
        model_conf = deep_update(model_conf, kwargs)

    model_name = model
    dataset_name = dataset
    log_path = time.strftime(f"{model}/{dataset}/%Y-%m-%d-%H-%M-%S.log", time.localtime())
    logger = get_logger(log_path)

    # if not verbose:
        # import logging
        # logger.setLevel(logging.ERROR)

    logger.info("Log saved in {}.".format(os.path.abspath(log_path)))
    model = model_class(model_conf)
    dataset_class = model_class._get_dataset_class()

    data_conf = {}
    if data_config_path is not None:
        if isinstance(data_config_path, str):
            # load dataset config from file
            conf = parser_yaml(data_config)
            data_conf.update(conf)
        else:
            raise TypeError(f"expecting `data_config_path` to be str, while get {type(data_config_path)} instead.")

    if data_config is not None:
        if isinstance(data_config, dict):
            # update config with given dict
            data_conf.update(data_config)
        else:
            raise TypeError(f"expecting `data_config` to be Dict, while get {type(data_config)} instead.")

    data_conf.update(model_conf['data'])    # update model-specified config

    dataset = dataset_class(name=dataset, config=data_conf)
    datasets = dataset.build(**model_conf['data'])
    logger.info(f"{datasets[0]}")
    logger.info(f"\n{set_color('Model Config', 'green')}: \n\n" + color_dict_normal(model_conf, False))
    
    prediction_inter_feat_DE_path = './data_for_recstudio/test_inter_feat_task1_DE.csv'
    prediction_inter_feat_JP_path = './data_for_recstudio/test_inter_feat_task1_JP.csv'
    prediction_inter_feat_UK_path = './data_for_recstudio/test_inter_feat_task1_UK.csv'
    task1_prediction_inter_feat_list = [prediction_inter_feat_DE_path, prediction_inter_feat_JP_path, prediction_inter_feat_UK_path]
    if do_prediction == False:
        model.fit(*datasets[:2], run_mode='light')
        # model.evaluate(datasets[-1])

        prediction_path = os.path.join('./predictions', time.strftime(f"{model_name}/{dataset_name}/%Y-%m-%d-%H-%M-%S.parquet", time.localtime()))
        res_dfs = []
        for pred_path in task1_prediction_inter_feat_list:
            predict_dataset = datasets[0].build_test_dataset(pred_path)
            res_df = model.predict(predict_dataset)
            res_dfs.append(res_df)
        res_df = pd.concat(res_dfs, axis=0)
        res_df = res_df.reset_index(drop=True)

        # save results 
        if not os.path.exists(os.path.dirname(prediction_path)):
            os.makedirs(os.path.dirname(prediction_path))
        res_df.to_parquet(prediction_path, engine='pyarrow')
        model.logger.info(f'Prediction is finished, results are saved in {prediction_path}.')

        # predict_dataset = datasets[0].build_test_dataset(prediction_inter_feat_path)
        # model.predict(predict_dataset, prediction_path)
    else:
        prediction_path = os.path.join('./predictions', time.strftime(f"{model_name}/{dataset_name}/%Y-%m-%d-%H-%M-%S.parquet", time.localtime()))
        # init model 
        model.logger = logging.getLogger('recstudio')
        model._init_model(datasets[0])

        res_dfs = []
        for pred_path in task1_prediction_inter_feat_list:
            predict_dataset = datasets[0].build_test_dataset(pred_path)
            res_df = model.predict(predict_dataset, model_path=model_path)
            res_dfs.append(res_df)
        res_df = pd.concat(res_dfs, axis=0)
        res_df = res_df.reset_index(drop=True)

        # save results 
        if not os.path.exists(os.path.dirname(prediction_path)):
            os.makedirs(os.path.dirname(prediction_path))
        res_df.to_parquet(prediction_path, engine='pyarrow')
        model.logger.info(f'Prediction is finished, results are saved in {prediction_path}.')
