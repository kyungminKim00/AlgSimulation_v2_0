from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 14:21:21 2018

@author: kim KyungMin
"""

import header.index_forecasting.RUNHEADER as RUNHEADER
if RUNHEADER.release:
    from libs import index_forecasting_adhoc
else:
    import index_forecasting_adhoc
from datasets.index_forecasting_protobuf2pickle import DataSet
# import util

import index_forecasting_test
from multiprocessing.managers import BaseManager
# import numpy as np
import pandas as pd


# from sklearn.metrics import classification_report
# from sklearn.metrics import f1_score
# from sklearn.metrics import mean_squared_error
import os
# import pickle
# import sys
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('agg')
import shutil
import argparse
# from mpl_finance import candlestick_ohlc
# from matplotlib.dates import date2num
import datetime
# import cloudpickle
# import plot_util
# import copy


def get_f_model_from_base(model_results, base_f_model):
    items = [item for item in os.listdir(model_results) if '.csv' in item]
    for item in items:
        if base_f_model.split('_sub_epo_')[1].split('_')[0] == item.split('_sub_epo_')[1].split('_')[0]:
            return item


if __name__ == '__main__':
    try:
        time_now = str(datetime.datetime.now())[:-10].replace(':', '-').replace('-', '').replace(' ', '_')
        """configuration
        """
        parser = argparse.ArgumentParser('')
        # init args
        parser.add_argument('--process_id', type=int, default=None)
        parser.add_argument('--m_target_index', type=int, default=None)
        parser.add_argument('--forward_ndx', type=int, default=None)
        parser.add_argument('--operation_mode', type=int, default=None)
        parser.add_argument('--dataset_version', type=str, default=None)
        # # For Demo
        # parser.add_argument('--process_id', type=int, default=11)
        # parser.add_argument('--m_target_index', type=int, default=None)
        # parser.add_argument('--forward_ndx', type=int, default=None)
        # parser.add_argument('--operation_mode', type=int, default=0)
        # parser.add_argument('--dataset_version', type=str, default=None)
        args = parser.parse_args()

        assert args.operation_mode is not None, 'check argument'
        if bool(args.operation_mode):
            assert args.forward_ndx is not None and args.m_target_index is not None and \
                   args.process_id is None and args.dataset_version is not None, 'check argument'
        else:
            assert args.forward_ndx is None and args.m_target_index is None and \
                   args.process_id is not None and args.dataset_version is None, 'check argument'

        # re-write RUNHEADER
        index_forecasting_test.configure_header(args)
        pickable_header = index_forecasting_test.convert_pickable(RUNHEADER)

        m_name = RUNHEADER.m_name
        m_inference_buffer = RUNHEADER.m_inference_buffer
        _model_location = './save/model/rllearn/' + m_name
        _tensorboard_log = './save/tensorlog/index_forecasting/' + m_name
        _dataset_dir = RUNHEADER.m_dataset_dir
        _full_tensorboard_log = False

        target_list = None
        if bool(args.operation_mode):
            _result_sub1 = './save/result/{}'.format(time_now)
            _result_sub2 = './save/result/{}/{}_T{}'.format(time_now, RUNHEADER.target_id2name(args.m_target_index),
                                                       str(args.forward_ndx))
            _result = '{}/{}'.format(_result_sub2, _model_location.split('/')[-1])
            target_list = [_result_sub1, _result_sub2,
                           _result, _result + '/fig_index', _result + '/fig_bound', _result + '/fig_scatter',
                           _result + '/fig_index/index', _result + '/fig_index/return', _result + '/fig_index/analytics',
                           _result + '/validation', _result + '/validation/fig_index',
                           _result + '/validation/fig_index/index', _result + '/validation/fig_index/return',
                           _result + '/final']
        else:
            _result = './save/result'
            _result = '{}/{}'.format(_result, _model_location.split('/')[-1])
            target_list = [_result, _result + '/fig_index', _result + '/fig_bound', _result + '/fig_scatter',
                           _result + '/fig_index/index', _result + '/fig_index/return', _result + '/fig_index/analytics',
                           _result + '/validation', _result + '/validation/fig_index',
                           _result + '/validation/fig_index/index', _result + '/validation/fig_index/return']
        for target in target_list:
            if os.path.isdir(target):
                shutil.rmtree(target, ignore_errors=True)
            try:
                os.mkdir(target)
            except FileExistsError:
                print('try one more time')
                os.mkdir(target)

        copy_file = ['/selected_x_dict.json', '/agent_parameter.json', '/agent_parameter.txt',
                     '/shuffled_episode_index.txt']
        if bool(args.operation_mode):  # use predefined base model
            dir_name = _model_location
            [shutil.copy2(dir_name + file_name, target_list[-7] + file_name) for file_name in copy_file]
        else:
            dir_name = './save/model/rllearn/{}'.format(index_forecasting_test.recent_procedure('./working_model_p', args.process_id, 'r'))
            [shutil.copy2(dir_name + file_name, target_list[-6] + file_name) for file_name in copy_file]

        # check _dataset_dir in operation mode
        _n_step, _cv_number, _n_cpu, _env_name, _file_pattern, _infer_set \
            = index_forecasting_test.meta_info(_model_location, _dataset_dir)
        if bool(args.operation_mode):
            _env_name = 'IF-{}'.format(RUNHEADER.dataset_version)
            _file_pattern = 'if_{}_cv%02d_%s.pkl'.format(RUNHEADER.dataset_version)

        exp_result = None
        for _mode in _infer_set:
            """ run application
            """
            # register
            BaseManager.register('DataSet', DataSet)
            manager = BaseManager()
            manager.start(index_forecasting_test.init_start, (pickable_header,))

            if _mode == 'validation':
                exp_result = '{}/validation'.format(_result)
            else:
                exp_result = _result

            # dataset injection
            sc = index_forecasting_test.Script(so=manager.DataSet(
                dataset_dir=_dataset_dir, file_pattern=_file_pattern,
                split_name=_mode, cv_number=_cv_number))

            sc.run(mode=_mode, env_name=_env_name, tensorboard_log=_tensorboard_log,
                   full_tensorboard_log=_full_tensorboard_log,
                   model_location=_model_location,
                   n_cpu=_n_cpu, n_step=_n_step, result=exp_result, m_inference_buffer=m_inference_buffer)

        # adhoc-process - confidence and align reg and classifier
        if bool(args.operation_mode):
            target_name = RUNHEADER.target_id2name(args.m_target_index)
            domain_detail = '{}_T{}_{}'.format(target_name, str(args.forward_ndx), args.dataset_version)
            domain = '{}_T{}'.format(target_name, str(args.forward_ndx))
            t_info = '{}_{}'.format(domain, time_now)
            target_file = './save/model_repo_meta/{}.pkl'.format(domain)
            meta_list = index_forecasting_adhoc.load(target_file, 'pickle')
            meta = [meta for meta in meta_list if meta['latest']][0]
            # result: 후처리 결과 파일 떨어지는 위치, model_location: 에폭별 실험결과 위치, f_base_model
            sc = index_forecasting_adhoc.Script(result=_result + '/final', model_location=_result, f_base_model=m_name,
                                                f_model=get_f_model_from_base(_result, meta['model_name']),
                                                adhoc_file='AC_Adhoc.csv', infer_mode=True, info=t_info)
            pd.set_option('mode.chained_assignment', None)
            sc.run_adhoc()
            pd.set_option('mode.chained_assignment', 'warn')
        else:
            pass

        # print test environments
        print('\nEnvs ID: {}'.format(_env_name))
        print('Data Set Number: {}'.format(_cv_number))
        print('Num Agents: {}'.format(_n_cpu))
        print('Num Step: {}'.format(_n_step))
        print('Result Directory: {}'.format(exp_result))
    except Exception as e:
        print('\n{}'.format(e))
        exit(1)
