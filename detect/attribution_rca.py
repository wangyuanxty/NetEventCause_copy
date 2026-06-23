
import json
import os.path
import pandas as pd
import yaml
import torch
from utils import PROJECT_ROOT_PATH, TimeRecorder
# from cause.event.pkg.models

# from cause.event.pkg.models.ode_rnn import ODERecurrentPointProcess
# from cause.event.pkg.models.rnn import ExplainableRecurrentPointProcess
from cause.event.pkg.models.rnn import ExplainableRecurrentPointProcess
# from cause.event.pkg.models.spnpp import SemiParametricPointProcess
# from cause.compress.rules import CompressRule
from .root_cause_analysys_base import RootCauseDetectionBase
import numpy as np
from cause.dataset.temporal_events_dataset import df2events_seq


class EventCauseDetection(RootCauseDetectionBase):
    def __init__(self,
                 ckpt_path: str,
                 alarm_group: str,
                 device='cpu',
                 steps=50,
                 updated_args_dict=None,
                 # root_alarm_threshold=2.0,
                 **kwargs
                 ):

        super(EventCauseDetection, self).__init__(algorithm='event cause', **kwargs)

        self.steps = steps

        with open(os.path.join(ckpt_path, 'config.json')) as f:
            config_json = json.load(f)

        for k, v in (updated_args_dict if updated_args_dict else {}).items():
            config_json[k] = v

        from cause.event.pkg.models.coldstart_erpp import ColdStartTTF, ColdStartLoRA

        if config_json['model'] == 'ERPP':
            model = ExplainableRecurrentPointProcess(**config_json)

        elif config_json['model'] == 'ERPP-TTF':
            model = ColdStartTTF(**config_json)
        elif config_json['model'] == 'ERPP-LoRA':
            model = ColdStartLoRA(**config_json)

        elif config_json['model'] == 'SPNPP':
            model = SemiParametricPointProcess(**config_json)

        elif config_json['model'] == 'ODE':
            model = ODERecurrentPointProcess(**config_json)
        else:
            raise NotImplementedError()

        if os.path.exists(os.path.join(ckpt_path, 'model.pt')):
            ckpt = torch.load(os.path.join(ckpt_path, 'model.pt'), weights_only=True)
            model.load_state_dict(ckpt, strict=False)
            full_n = config_json.get('full_n_types', config_json['n_types'])
            model.extend_type_num(full_n)
            if hasattr(model, '_seen'):
                model._seen = set(range(config_json['n_types']))
                model._ttf_enabled = True  # 开启 RCA 推理时的 TTF
            print('Loading parameters successfully (%d -> %d types).' %
                  (config_json['n_types'], full_n))
        else:
            print('There is no model.pt in %s' % ckpt_path)
        self.model = model.to(torch.device(device))
        self.device = device

        # with open(os.path.join(PROJECT_ROOT_PATH, 'config', 'cause.yaml'), 'r') as f:
        #     prior_root_alarm_prob = yaml.load(f.read(), Loader=yaml.FullLoader)
        #     self.root_alarm_ratio = {}
        #     try:
        #         self.root_alarm_ratio = float(prior_root_alarm_prob['root_alarm_ratio'][alarm_group])
        #     except KeyError as e:
        #         self.root_alarm_ratio = prior_root_alarm_prob['default']

    def root_alarm_prob(self, root_intensity, conditional_intensity):
        # return (prior_prob * prior_intensity) / (prior_prob * prior_intensity + (1-prior_prob) * conditional_intensity)
        return root_intensity / conditional_intensity

    def _detect(self, event_df: pd.DataFrame,  group_col='compress_group'):

        # if not compressed:
        #     raise NotImplementedError('df should be ')

        # df = df.set_index('_id')

        result = self.init_result_df(event_df)

        rc_cols = self.get_root_cause_cols()
        # result['detect_type'] = None

        # region 处理事件模型无法识别的告警事件，将此类告警认为是根告警，根告警概率为self.root_alarm_ratio
        not_identified_rows = event_df['type'].isnull()
        result.loc[not_identified_rows, rc_cols[0]] = result.loc[not_identified_rows].index
        result.loc[not_identified_rows, rc_cols[1]] = np.inf
        result.loc[not_identified_rows, 'identified'] = False
        result.loc[not_identified_rows, 'root_alarm'] = True
        result.loc[not_identified_rows, 'root_prob'] = 1.0
        result.loc[not_identified_rows, 'modified_root_prob'] = 1.0

        # endregion

        event_df = event_df.loc[~not_identified_rows]

        event_df['depth'] = 0

        data_batch = self.events_df2tensor_batch(event_df)

        # log_intensities_events.detach().cpu(), prior_log_intensities_events.detach().cpu()

        tr = TimeRecorder()
        with tr('attribution'):
            contribution_mat, log_intensities_events, base_log_intensities_events, prior_log_intensities_events = [x.squeeze(dim=0) for x in self.model.get_seq_contribution(
                data_batch.to(self.device), steps=self.steps
            )]

        print('Attribution time: %.2f s' % (tr['attribution']))
        for i in range(contribution_mat.size()[0]):

            cols = []
            res = []

            root_alarm_prob = self.root_alarm_prob(
                float(prior_log_intensities_events[i].exp()),
                float(log_intensities_events[i].exp())
            )

            cols += ['root_prob']
            res += [root_alarm_prob]

            # result.loc[compressed_df.iloc[i].name, 'root_prob'] = root_alarm_prob
            # result.loc[compressed_df.iloc[i].name, 'modified_root_prob'] = root_alarm_prob
            # if i == 0:
            # result.loc[compressed_df.iloc[i].name, 'root_alarm'] = True

            contribution_i = contribution_mat[i].clone()
            contribution_i[contribution_i < 0] = 0

            # root_cause_rank = [
            #     (self.cause_score_modify(float(contribution_i[k]), compressed_df.iloc[k], compressed_df.iloc[i]), k)
            #     for k in range(i)
            # ]
            root_cause_rank = [
                (self.cause_score_modify(float(contribution_i[k]), event_df.iloc[k], event_df.iloc[i]), k)
                for k in range(i)
            ]
            root_cause_rank = filter(lambda x: x[0] > 0, root_cause_rank)

            root_cause_rank = sorted(root_cause_rank, reverse=True)

            causative_alarms = [event_df.iloc[i] for _, i in root_cause_rank[:5]]

            if len(causative_alarms) == 0 or i == 0:
                modified_root_prob = 1.0
            else:
                modified_root_prob = root_alarm_prob

            # result.loc[compressed_df.iloc[i].name, 'modified_root_prob'] = modified_root_prob

            cols += ['modified_root_prob']
            res += [modified_root_prob]

            cols += ['root_alarm', 'depth']

            if modified_root_prob > 0.5:
                # 识别 i 为根告警
                # result.loc[compressed_df.iloc[i].name, 'root_alarm'] = True
                # root_cause_rank = [(np.inf, i)] + root_cause_rank
                event_df.loc[event_df.iloc[i].name, 'depth'] = 0
                res += [True, 0]
            else:
                event_df.loc[event_df.iloc[i].name, 'depth'] = causative_alarms[0]['depth'] + 1
                res += [False, causative_alarms[0]['depth'] + 1]

            for j, cause in zip(range(self.top_k), root_cause_rank):

                cause_id_col = rc_cols[j * self.single_cause_col_num]
                cause_score_col = rc_cols[j * self.single_cause_col_num + 1]
                cols += [cause_id_col, cause_score_col]
                res += [event_df.iloc[cause[1]].name, float(cause[0])]

            result.loc[event_df.iloc[i].name, cols] = res

        return result

    def cause_score_modify(self, score, cause_alarm, current_alarm, T=3600, ratio=5.0):
        """
        return : [0,inf] 如果返回值为0，将不作为因告警集中的候选告警
        """
        score = max(score, 1e-2)
        if current_alarm['type']==1 and cause_alarm['type']!=0:
            score = 0
        elif current_alarm['type']==2 and cause_alarm['type']!=0:
            score = 0
        elif current_alarm['type']==3 and cause_alarm['type']!=2:
            score = 0
        elif current_alarm['type']==4 or current_alarm['type']==0:
            score = 0

        return score
    def events_df2tensor_batch(self, df):
        event_seq = df2events_seq(df, alarm_type_col='type', time_col='t')
        data_batch = torch.FloatTensor(np.array(event_seq)).unsqueeze(dim=0)
        return data_batch
