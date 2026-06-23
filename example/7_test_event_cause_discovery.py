import argparse
import os
import os.path as osp
import sys
import json
sys.path.append('.')
import joblib
import yaml
sys.modules['sklearn.externals.joblib'] = joblib
import random

try:
    from base import *
except ModuleNotFoundError:
    from example.base import *

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


import pdb
# pdb.set_trace()
from cause.event.tasks import train
from cause.event.tasks.train import get_parser, osp, get_hparam_str, get_model, get_device, \
    train_nn_models, eval_nll, predict_next_event, get_infectivity_matrix

from cause.event.pkg.models.rnn import EventSeqDataset
from cause.event.pkg.utils.evaluation import eval_fns
from cause.event.pkg.utils.logging import get_logger, init_logging
from cause.event.pkg.utils.misc import (
    Timer,
    export_csv,
    export_json,
    makedirs,
    set_rand_seed,
)
from cause.event.pkg.utils.pp import (
    event_seq_to_counting_proc,
)
from utils import find_dir_until, dict2obj, PROJECT_ROOT_PATH


if __name__ == "__main__":
    args = get_parser().parse_args()
    train.args = args
    assert args.model is not None, "`model` needs to be specified."

    output_path = osp.join(
        'checkpoints',
        args.dataset,
        args.kind,
        f"split_id={args.split_id}",
        args.model,
        get_hparam_str(args),
    )
    makedirs([output_path])
    train.output_path = output_path

    # initialization
    set_rand_seed(args.rand_seed, args.cuda)
    init_logging(output_path)
    logger = get_logger(__file__)
    train.logger = logger



    # load data
    input_path = osp.join('cache', args.dataset, 'dataset', args.kind)
    if not osp.exists(input_path):
        print('%s does not exist.' % input_path)
        exit(0)

    import pickle
    with open(osp.join(input_path, "data.pkl"), "rb") as f:
        data = pickle.load(f)
    n_types = int(data["n_types"])
    # 冷启动数据集：训练时只建已知类型的嵌入
    train_n_types = n_types
    if 'coldstart' in args.kind:
        train_idx = [int(i) for i in data["train_test_splits"][0][0]]
        train_n_types = int(max(t for s in [data["event_seqs"][i] for i in train_idx]
                               for t in s[:, 1].astype(int)) + 1)
    setattr(args, 'full_n_types', n_types)

    # Loading prior probability of each event type occurring as root alarms
    with open(os.path.join(PROJECT_ROOT_PATH, 'config', 'cause.yaml'), 'r') as f:
        prior_root_alarm_prob = yaml.load(f.read(), Loader=yaml.FullLoader)[args.kind]['root_alarm_ratio']

        root_prob = []
        for event_type in range(0, int(n_types)):
            if event_type not in prior_root_alarm_prob:
                root_prob.append(prior_root_alarm_prob['default'])
            else:
                root_prob.append(prior_root_alarm_prob[int(event_type)])

    setattr(args, 'root_prob', root_prob)
    setattr(args, 'full_n_types', n_types)
    setattr(args, 'n_types', train_n_types)
    export_json(vars(args), osp.join(output_path, "config.json"))
    args.n_types = n_types  # 恢复完整类型数供后续使用

    logger.info(args)

    event_seqs = data["event_seqs"]
    event_seqs = np.array([np.array(es)[:, :-1] for es in event_seqs], dtype='object')
    train_event_seqs = event_seqs[data["train_test_splits"][args.split_id][0]]
    test_event_seqs = event_seqs[data["train_test_splits"][args.split_id][1]]
    # sorted test_event_seqs by their length
    test_event_seqs = sorted(test_event_seqs, key=lambda seq: len(seq))

    if osp.exists(osp.join(input_path, "infectivity.txt")):
        A_true = np.loadtxt(osp.join(input_path, "infectivity.txt"))
    else:
        A_true = None

    with Timer("Training model"):
        # define model
        if args.previous_model_dir is None or \
                find_dir_until(
                    osp.join(PROJECT_ROOT_PATH, 'checkpoints', args.previous_model_dir, args.kind), 'model.pt'
                ) is None:
            model = get_model(args, train_n_types)
        else:
            ckpt_path = find_dir_until(osp.join(PROJECT_ROOT_PATH, 'checkpoints', args.previous_model_dir, args.kind), 'model.pt')

            with open(os.path.join(ckpt_path, 'config.json')) as f:
                config_json = json.load(f)

            old_n_types = config_json['n_types']

            model = get_model(args, old_n_types)

            if os.path.exists(os.path.join(ckpt_path, 'model.pt')):
                model.load_state_dict(
                    torch.load(os.path.join(ckpt_path, 'model.pt'))
                )
                print('Loading parameters from %s successfully.' % ckpt_path)

            model.extend_type_num(train_n_types)

        if args.model in ["ODE-RNN", "SPNPP", "RME", "ERPP", "ERPP-TTF", "ERPP-SVD", "RPPN"]:
            train.dataloader_args = {
                "batch_size": args.batch_size,
                "collate_fn": EventSeqDataset.collate_fn,
                "num_workers": args.num_workers,
            }
            device = get_device(args.cuda)
            train.device = device

            model = model.to(device)
            model = train_nn_models(model, train_event_seqs, args)

        else:
            # NOTE: may change to weighted sampling (by seq length)
            if "max_seqs" in args and args.max_seqs > 0:
                train_event_seqs = random.sample(
                    list(train_event_seqs), args.max_seqs
                )

            train_cps = [
                event_seq_to_counting_proc(seq, n_types, to_numpy=True)
                for seq in tqdm(train_event_seqs)
            ]
            model.fit(train_cps)
            # TODO: many tick models can't be easily pickled. Probabily need to
            # write a wrapper class.
            # with open(osp.join(output_path, "model.pkl"), "wb") as f:
            # pickle.dump(model, f)


    # evaluate nll
    results = {}
    with Timer("Evaluate negative log-likelihood"):
        results["nll"] = eval_nll(model, test_event_seqs, args)
        print("nll", results["nll"])

    # export evaluation results
    time = pd.Timestamp.now()
    df = pd.DataFrame(
        columns=[
            "timestamp",
            "dataset",
            "split_id",
            "model",
            "metric",
            "value",
            "config",
        ]
    )

    for metric_name, val in results.items():

        df.loc[len(df)] = (
            time,
            args.dataset,
            args.split_id,
            args.model,
            metric_name,
            val,
            vars(args),
        )

    logger.info(df)
    export_csv(df, osp.join('checkpoints', args.dataset, "results.csv"), append=True)
