import os

import pandas as pd
from tqdm import tqdm

try:
    from base import *
except ModuleNotFoundError:
    from example.base import *

import random
import shutil
from utils import find_dir_until, reset_dir, add_label_cols

from cause.event.tasks.train import get_parser, osp

parser = argparse.ArgumentParser()

# parser.add_argument(
#     '--ckpt_path', type=str,
#     default='checkpoints/temp/Events_SAAS_compressed/split_id=0/ERPP/max_mean=100,n_bases=7,hidden_size=64,lr=0.001')

parser.add_argument(
    '--ckpt_path', type=str,
    default=None)

parser.add_argument('--alarm_type_index_path', type=str, default='alarm_type_index.csv')

parser.add_argument('--dataset', type=str, default='merge_alarm')
parser.add_argument('--kind', type=str, default='HCS')
parser.add_argument('--model', type=str, default='ERPP')
parser.add_argument('--algorithm', type=str, default='event_cause', choices=['event_cause', 'nearest'])
parser.add_argument('--steps', type=int, default=50)
parser.add_argument('--suffix', type=str, default='')
# parser.add_argument('--sub_type_index_path', type=str, default='cache/Events_SAAS_compressed/index_map.csv')
# parser.add_argument('--samples_path', type=str, default='cache/Events_SAAS_compressed/excels')
# parser.add_argument('--eval_num', type=int, default=100)
# parser.add_argument('--single', type=bool, default=False, help='True: 采样单个文件进行检测, False: 检测大文件。')
# parser.add_argument('--single_max_length', type=int, default=6000)
parser.add_argument('--do_compress', type=bool, default=True)
parser.add_argument('--manual_rule', action='store_true', help='是否添加手工规则以矫正因果得分和根告警识别概率')
# parser.add_argument('--fill_missing', action='store_true', help='通过命令行中加入--fill_missing开启，只适用于评估频繁项挖掘算法的结果')
parser.add_argument('--max_seqs', type=int, default=0,
                    help='max test sequences to process (0=all)')
parser.add_argument('--save_all', type=bool, default=False)
parser.add_argument('--output_dir', type=str, default='root_alarms_identification')
parser.add_argument('--add_label_cols', type=bool, default=False)


# parser.add_argument('--max_len', type=int, default=100)
# parser.add_argument('--max_hour', type=int, default=48)
args = parser.parse_args()


dataset = os.path.join('cache', '%s' % args.dataset, 'dataset', '%s' %args.kind)
# dataset = os.path.join('cache', '%s' % args.dataset)


alarm_type_index_path = os.path.join('cache', args.dataset, 'alarm_type_index.csv')
sub_type_index_path = os.path.join(dataset, 'index_map.csv')

algorithm_name = args.algorithm
if algorithm_name == 'event_cause':
    algorithm_name += '-' + args.model

if args.suffix != '':
    algorithm_name += '_' + args.suffix
output_path = os.path.join('outputs', args.dataset, args.output_dir, args.kind, algorithm_name)
reset_dir(output_path)

ckpt_path = args.ckpt_path

if ckpt_path == 'None' or ckpt_path is None:
    ckpt_model = 'ERPP' if args.model == 'ERPP-TTF' else args.model
    father_dir = "checkpoints/%s/%s/split_id=0/%s" % (args.dataset, args.kind, ckpt_model)
    ckpt_path = find_dir_until(father_dir, 'model.pt')

else:
    ckpt_path = args.ckpt_path


if args.algorithm == 'event_cause':
    from detect.attribution_rca import EventCauseDetection
    event_rca = EventCauseDetection(
        ckpt_path,
        args.kind,
        device='cpu',
        top_k=5,
        steps=args.steps,
        updated_args_dict={}
    )
elif args.algorithm == 'nearest':

    from detect.nearest import NearestEventCauseDetection
    event_rca = NearestEventCauseDetection(
        top_k=5,
        updated_args_dict={}
    )
else:
    raise NotImplementedError


# if args.single:
#
#     samples_path = os.path.join(dataset, 'excels')
#     data_samples_paths = [os.path.join(args.samples_path, x) for x in os.listdir(args.samples_path)]
#     for file in random.sample(data_samples_paths, k=min(len(data_samples_paths), args.eval_num)):
#         if str(file).endswith('csv'):
#             df = pd.read_csv(file)
#         else:
#             df = pd.read_excel(file)
#
#         rca_result = event_rca.detect(df.set_index('_id'), group_col='compress_group')
#         rca_result.to_excel(
#             os.path.join(output_path, file.split(os.sep)[-1])
#         )
# else:
import pickle
data_path = os.path.join('cache', args.dataset, 'dataset', args.kind, 'data.pkl')
with open(data_path, "rb") as f:
    data = pickle.load(f)
test_idx = data["train_test_splits"][0][1]
test_event_seqs = [data["event_seqs"][i] for i in test_idx]

# print的内容不仅输出到终端，还能打印到某个文件
from utils.tee import Tee
logfile = open(os.path.join(output_path, 'rca.log'), 'w')
sys.stdout = Tee(sys.stdout, logfile)


max_seqs = getattr(args, 'max_seqs', 0)
if max_seqs > 0:
    test_event_seqs = test_event_seqs[:max_seqs]

for i, test_event_seq in enumerate(tqdm(test_event_seqs, desc="RCA")):
    df = pd.DataFrame(test_event_seq, columns=['t', 'type', 'label-root-1'])
    # setting the value of column root_alarm as True if the label-root-1 is -1, else False
    df['label_root_alarm'] = df['label-root-1'].apply(lambda x: True if x == -1 else False)
    rca_result = event_rca.detect(df)

    rca_result.to_excel(
        os.path.join(output_path, args.kind+'%d.xlsx' % i), index_label='index'
    )


# event_seqs = data["event_seqs"]
# event_seqs = np.array([np.array(es)[:, :-1] for es in event_seqs])
# train_event_seqs = event_seqs[data["train_test_splits"][args.split_id][0]]
#
#
# alarm_path = os.path.join('cache', args.dataset, 'alarm_ci.csv')
# print('Loading alarm df from %s' % alarm_path)
# df = pd.read_csv(alarm_path)
# df = df.sort_values(['incident_open_time', '_id'])
# df = df.drop_duplicates('_id', keep='first')
# df = df.loc[df['alarm_type_group'] == args.kind]
# df = df.set_index('_id')


# print('Start detecting for kind %s' % args.kind)
# if args.add_label_cols:
#     # rca_result
#     rca_result = add_label_cols(rca_result)
#
# compressed_rca_result = rca_result.loc[~rca_result['repeated']]
#
# compressed_rca_result.to_excel(
#     os.path.join(output_path, 'compressed_' + args.kind+'.xlsx'), index_label='index'
# )

if args.save_all:
    rca_result.to_excel(
        os.path.join(output_path, args.kind+'.xlsx'), index_label='index'
    )
print('Detected results are saved to %s.' % output_path)
# for begin in range(0, len(df), args.single_max_length):
#     end = min(len(df), begin + args.single_max_length)
#     print('Detecting %d-%d' % (begin, end))
#
#     rca_result = event_rca.detect(
#         df.iloc[begin: end],
#         group_col='compress_group', do_compress=args.do_compress
#     )
#
#     rca_result.to_excel(
#         os.path.join(output_path, args.kind+'_%d-%d.xlsx' %(begin, end))
#     )
#     compressed_rca_result = rca_result.loc[~rca_result['repeated']]
#
#     compressed_rca_result.to_excel(
#         os.path.join(output_path, 'compressed_' + args.kind+'_%d-%d.xlsx' % (begin, end))
#     )
logfile.close()
