import os

import pandas as pd

try:
    from base import *
except ModuleNotFoundError:
    from example.base import *

from detect.attribution_rca import EventCauseDetection
from utils import file2df, move_column_to_front
from utils.metric import root_identification_acc, acc_k, effective_intersection
from sklearn.metrics import RocCurveDisplay, confusion_matrix, accuracy_score, recall_score, roc_auc_score, roc_curve
from matplotlib import pyplot as plt

import random
import shutil

from cause.event.tasks.train import get_parser, osp

parser = argparse.ArgumentParser()

# parser.add_argument(
#     '--ckpt_path', type=str,
#     default='checkpoints/temp/Events_SAAS_compressed/split_id=0/ERPP/max_mean=100,n_bases=7,hidden_size=64,lr=0.001')
#
# parser.add_argument('--alarm_type_index_path', type=str, default='cache/alarm_type_index.csv')

parser.add_argument('--dataset', type=str, default='toy')
parser.add_argument('--algorithm', type=str, default=None, help='If not None, only evaluate the specified algorithm. '
                                                                'Otherwise, evaluate all algorithms in the output directory.')
parser.add_argument('--output_dir', type=str, default='root_alarms_identification')
parser.add_argument('--kind', type=str, default='ggem-1K-5')
parser.add_argument('--eval_cnt', type=int, default=100)

parser.add_argument('--fill_missing', action='store_true', help='通过命令行中加入--fill_missing开启，只适用于评估频繁项挖掘算法的结果')

# parser.add_argument('--dataset', type=str, default='SAAS')
# # parser.add_argument('--sub_type_index_path', type=str, default='cache/Events_SAAS_compressed/index_map.csv')
# # parser.add_argument('--samples_path', type=str, default='cache/Events_SAAS_compressed/excels')
# parser.add_argument('--eval_num', type=int, default=100)
# parser.add_argument('--single', type=bool, default=False)
# parser.add_argument('--single_max_length', type=int, default=2000)

# parser.add_argument('--max_len', type=int, default=100)
# parser.add_argument('--max_hour', type=int, default=48)
args = parser.parse_args()

from cause.dataset.temporal_events_dataset import alarm_df2event_dataset

from utils import PROJECT_ROOT_PATH, mergedir_files2df

outputs_dir = os.path.join(PROJECT_ROOT_PATH, 'outputs', args.dataset, args.output_dir, args.kind)
# outputs_dir = os.path.join(PROJECT_ROOT_PATH, 'outputs', args.dataset)


gt_cols = ['label-root-%d' % i for i in range(1, 5)] + ['alarm_type_group', '_id']


root_1_col = 'label-root-1'

all_results = []
for algorithm in os.listdir(outputs_dir):
    if not os.path.isdir(os.path.join(outputs_dir, algorithm)):
        continue

    if args.algorithm is not None and algorithm != args.algorithm:
        continue

    print('#' * 10, algorithm, '#' * 10)
    dir_path = os.path.join(outputs_dir, str(algorithm))
    detected_df_list = []
    for i, file in zip(range(int(args.eval_cnt)), filter(lambda f: f.startswith(args.kind), os.listdir(dir_path))):
        df = file2df(os.path.join(dir_path, file))
        # add prefix i to the index
        detected_df_list.append(df)

    detected_df = pd.concat(detected_df_list)
    # reset the index of detected_df as 0, 1, 2, ...
    detected_df.reset_index(drop=True, inplace=True)

    ground_truth_df = detected_df

    # not_null_index = ground_truth_df[~(ground_truth_df[root_1_col].isnull())].index

    # 有效的标注样本数
    gt_not_null_cnt = (~ground_truth_df['label-root-1'].isnull()).sum()

    # region 评估根告警分类准确度

    # 预处理
    reserved_gt_df, reserved_detected_df = ground_truth_df, detected_df

    is_root_col = 'root_alarm'
    is_root_col_gt = 'label_root_alarm'
    root_prob_col = 'modified_root_prob'

    gt_root_alarm_cnt = reserved_gt_df[is_root_col_gt].sum()

    root_efficient = reserved_detected_df[root_prob_col].nlargest(gt_root_alarm_cnt).iloc[-1]
    print('Recommended root alarm classification efficient is %.2f for algorithm %s.' % (root_efficient, algorithm))
    print('%d/%d detected root alarms in detected result.' %
          ((reserved_detected_df[is_root_col]).sum(), len(reserved_detected_df)))
    # reserved_detected_df.loc[reserved_detected_df['root_prob'] >= root_efficient, is_root_col] = True

    # 存储excel
    merge_rid_df = detected_df
    merge_rid_df = move_column_to_front(merge_rid_df, [is_root_col_gt, is_root_col, 'root_prob', root_prob_col])
    merge_rid_df.to_excel(os.path.join(outputs_dir, str(algorithm), 'root_alarm.xlsx'))

    # 生成混淆矩阵并保存
    # Confusion matrix whose i-th row and j-th column entry indicates
    # the number of samples with true label being i-th class and predicted label being j-th class.
    cm = confusion_matrix(np.array(merge_rid_df[is_root_col_gt]), np.array(merge_rid_df[is_root_col]))
    cm_df = pd.DataFrame(cm, columns=['detected-False', 'detected-True'], index=['gt-False', 'gt-True'])
    print(cm_df)
    # np.savetxt(os.path.join(outputs_dir, str(algorithm), 'confusion_matrix.txt'), cm, fmt='%d')
    cm_df.to_excel(os.path.join(outputs_dir, str(algorithm), 'confusion_matrix.xlsx'))

    # 计算指标：AUC、ACC、RECALL

    is_root_alarm_gt = merge_rid_df[is_root_col_gt].tolist()
    is_root_alarm_detected = merge_rid_df[is_root_col].tolist()
    accuracy = accuracy_score(is_root_alarm_gt, is_root_alarm_detected)
    recall = recall_score(is_root_alarm_gt, is_root_alarm_detected)

    detected_score = (merge_rid_df[root_prob_col]).tolist()
    roc_auc = roc_auc_score(is_root_alarm_gt, detected_score)

    print('Root alarm identification\nAUC: %.2f, Acc: %.2f, Recall: %.2f\n' % (roc_auc, accuracy, recall))

    # 绘制roc曲线
    fpr, tpr, _ = roc_curve(is_root_alarm_gt, detected_score)
    roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr).plot()
    roc_display.figure_.set_size_inches(5, 5)
    plt.plot([0, 1], [0, 1], color='g')
    plt.savefig(os.path.join(outputs_dir, str(algorithm), 'roc.pdf'), dpi=300, bbox_inches="tight")

    # gt_df, detected_df = effective_intersection(gt_df, root_1_col, detected_df, 'root-1')
    # is_root_alarm_gt = (gt_df.index == gt_df[root_1_col]).tolist()
    # is_root_alarm_detected = (detected_df[is_root_col]).tolist()
    # accuracy = accuracy_score(is_root_alarm_gt, is_root_alarm_detected)
    # recall = recall_score(is_root_alarm_gt, is_root_alarm_detected)
    #
    # if root_prob_col not in detected_df.columns:
    #     detected_df.loc[detected_df[is_root_col], root_prob_col] = 1.0
    #     detected_df.loc[~detected_df[is_root_col], root_prob_col] = 0.0
    #
    # detected_score = (detected_df[root_prob_col]).tolist()
    #
    # roc_auc = roc_auc_score(is_root_alarm_gt, detected_score)
    #
    # fpr, tpr, _ = roc_curve(is_root_alarm_gt, detected_score)

    result = {'algorithm': algorithm, 'cnt': gt_not_null_cnt, 'auc': roc_auc, 'acc': accuracy, 'recall': recall}

    # acc = {}

    acc_derivative = acc_k(ground_truth_df, detected_df, k=5, only_derivative=True)

    def func(acc, kind, updated_result):
        for i in range(1, 6):
            # acc[i] = acc_k(ground_truth_df, detected_df, k=5, only_derivative=True)
            print('%s alarms: AC@%d of direct: %.2f' % (kind, i, acc[i-1]))
            v = np.array([acc[_] for _ in range(i)]).mean(axis=0)
            print('%s alarms: AVG@%d of direct: %.2f\n' % (kind, i, v))

            updated_result['%s_direct_acc@%d' % (kind, i)] = acc[i-1]
            updated_result['%s_direct_avg@%d' % (kind, i)] = v
            # result['derivative_root_acc@%d' % i] = acc[i][1]

    func(acc_derivative, 'Derivative', result)

    acc_all = acc_k(ground_truth_df, detected_df, k=5, only_derivative=False)
    func(acc_all, 'All', result)

    # ── 按类型分组评估 ──
    if 'type' in ground_truth_df.columns:
        type_col = ground_truth_df['type']
        all_types = sorted(type_col.unique())

        if args.kind == 'coldstart-7':
            # coldstart-7: 0-4 is known, 5-6 is cold
            known_types = [t for t in all_types if t <= 4]
            cold_types  = [t for t in all_types if t >= 5]
            groups = [('Known(A~E)', known_types), ('Cold(F,G)', cold_types)]
        else:
            # ggem-1K-5 / other: all types are known
            groups = [('All types', all_types)]

        for group_name, type_list in groups:
            mask = type_col.isin(type_list)
            gt_df_k = ground_truth_df[mask]
            det_df_k = detected_df[mask]
            if len(gt_df_k) == 0:
                continue

            print(f'\n--- {group_name} (types {list(type_list)}) ---')
            root_col = 'label_root_alarm'
            prob_col = 'modified_root_prob'
            if root_col in gt_df_k.columns and prob_col in det_df_k.columns:
                auc_k = roc_auc_score(gt_df_k[root_col], det_df_k[prob_col])
                print(f'Root AUC: {auc_k:.3f}')
            acc_k_types = acc_k(gt_df_k, det_df_k, k=5, only_derivative=True)
            for i in range(1, 6):
                print(f'AC@{i}: {acc_k_types[i-1]:.3f}', end='  ')
            print()

            # 每个类型单独看
            for t in type_list:
                mask_t = type_col == t
                gt_df_t = ground_truth_df[mask_t]
                det_df_t = detected_df[mask_t]
                if len(gt_df_t) == 0:
                    continue
                names = {0:'A',1:'B',2:'C',3:'D',4:'E',5:'F(root)',6:'G(deriv)'}
                label = names.get(int(t), f'type{t}')
                print(f'  Type {int(t)} {label}:')
                if root_col in gt_df_t.columns and prob_col in det_df_t.columns:
                    try:
                        auc_t = roc_auc_score(gt_df_t[root_col], det_df_t[prob_col])
                        print(f'    Root AUC: {auc_t:.3f}')
                    except ValueError:
                        print(f'    Root AUC: nan')
                acc_t = acc_k(gt_df_t, det_df_t, k=5, only_derivative=True)
                for i in range(1, 6):
                    print(f'    AC@{i}: {acc_t[i-1]:.3f}', end='  ')
                print()

    print('-' * 21)
    all_results.append(result)

cnt_sum = sum([r['cnt'] for r in all_results])

weight_mean = {
    k: sum([r[k] * r['cnt'] for r in all_results])/cnt_sum for k in all_results[0].keys() if k != 'algorithm' and k != 'cnt'
}
weight_mean['algorithm'] = 'Weighted Mean'
weight_mean['cnt'] = int(cnt_sum)
all_results.append(weight_mean)

result_df = pd.DataFrame(all_results)

# weighted_mean = result_df.dot(result_df['cnt']/float(result_df['cnt'].sum()))
# result_df[result_df.columns[1:]].to_numpy() / (result_df['cnt']/float(result_df['cnt'].sum())).to_numpy().unsqueeze(dim=0)

result_df.to_excel(os.path.join(outputs_dir, 'eval.xlsx'))










