import qlib
import pandas as pd
from qlib.constant import REG_CN
from qlib.utils import init_instance_by_config, flatten_dict
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord, SigAnaRecord
from qlib.tests.data import GetData
from qlib.contrib.report import analysis_model, analysis_position
from qlib.data import D
import os,sys,site
import ruamel.yaml as yaml
from pathlib import Path
os.chdir(sys.path[0])
sys.path.append("..")
from qlib_tools.tests.config import CSI300_BENCH, CSI300_GBDT_TASK, CSI300_MULTI_TRANSFORMER_TASK

def get_path_list(path):
    if isinstance(path, str):
        return [path]
    else:
        return list(path)

def sys_config(config, config_path):
    """
    Configure the `sys` section

    Parameters
    ----------
    config : dict
        configuration of the workflow.
    config_path : str
        path of the configuration
    """
    sys_config = config.get("sys", {})

    # abspath
    for p in get_path_list(sys_config.get("path", [])):
        sys.path.append(p)

    # relative path to config path
    for p in get_path_list(sys_config.get("rel_path", [])):
        sys.path.append(str(Path(config_path).parent.resolve().absolute() / p))

if __name__ == "__main__":

    # use default data
    provider_uri = "~/.qlib/qlib_data/cn_data"  # target_dir
    qlib.init(provider_uri=provider_uri, region=REG_CN)

    market = "csi300"
    benchmark = "SH000300"

    config_path = '/home/jxfeng/code/fintech_qlib/examples/benchmarks/LightGBM/workflow_config_lightgbm_Alpha360.yaml'

    with open(config_path) as fp:
        config = yaml.safe_load(fp)

    # config the `sys` section
    sys_config(config, config_path)

    task = config['task']
    data_handler_config = config['data_handler_config']

    # model initiaiton
    model = init_instance_by_config(task["model"])
    dataset = init_instance_by_config(task["dataset"])
    

    # NOTE: This line is optional
    # It demonstrates that the dataset can be used standalone.
    # example_df = dataset.prepare("train")
    # print(example_df.head())
    # start exp
    with R.start(experiment_name="train_model"):
        R.log_params(**flatten_dict(task))
        model.fit(dataset)
        R.save_objects(trained_model=model)
        rid = R.get_recorder().id

    ###################################
    # prediction, backtest & analysis
    ###################################
    port_analysis_config = {
        "executor": {
            "class": "SimulatorExecutor",
            "module_path": "qlib.backtest.executor",
            "kwargs": {
                "time_per_step": "day",
                "generate_portfolio_metrics": True,
            },
        },
        "strategy": {
            "class": "TopkDropoutStrategy",
            "module_path": "qlib.contrib.strategy.signal_strategy",
            "kwargs": {
                "model": model,
                "dataset": dataset,
                "topk": 50,
                "n_drop": 5,
            },
        },
        "backtest": {
            "start_time": "2017-01-01",
            "end_time": "2020-08-01",
            "account": 100000000,
            "benchmark": benchmark,
            "exchange_kwargs": {
                "freq": "day",
                "limit_threshold": 0.095,
                "deal_price": "close",
                "open_cost": 0.0005,
                "close_cost": 0.0015,
                "min_cost": 5,
            },
        },
    }

    # backtest and analysis
    with R.start(experiment_name="backtest_analysis"):
        recorder = R.get_recorder(recorder_id=rid, experiment_name="train_model")
        model = recorder.load_object("trained_model")

        # prediction
        recorder = R.get_recorder()
        ba_rid = recorder.id
        sr = SignalRecord(model, dataset, recorder)
        sr.generate()

        # backtest & analysis
        par = PortAnaRecord(recorder, port_analysis_config, "day")
        par.generate()

    
    
    recorder = R.get_recorder(recorder_id=ba_rid, experiment_name="backtest_analysis")
    print(recorder)
    pred_df = recorder.load_object("pred.pkl")
    report_normal_df = recorder.load_object("portfolio_analysis/report_normal_1day.pkl")
    positions = recorder.load_object("portfolio_analysis/positions_normal_1day.pkl")
    analysis_df = recorder.load_object("portfolio_analysis/port_analysis_1day.pkl")

    analysis_position.report_graph(report_normal_df, show_notebook=False)
    analysis_position.risk_analysis_graph(analysis_df, report_normal_df, show_notebook=False)
    label_df = dataset.prepare("test", col_set="label")
    label_df.columns = ['label']
    pred_label = pd.concat([label_df, pred_df], axis=1, sort=True).reindex(label_df.index)
    analysis_position.score_ic_graph(pred_label, show_notebook=False)
    analysis_model.model_performance_graph(pred_label, show_notebook=False)


