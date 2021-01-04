import header.index_forecasting.RUNHEADER as RUNHEADER
from util import get_unique_list

import pandas as pd
from collections import OrderedDict
# import numpy as np


def add_item(it, d_f_summary, var_list, b_current_pt_only=False):
    Condition = d_f_summary['var_name'] == it
    selected = d_f_summary[Condition]
    if b_current_pt_only:
        if selected['units'].values[0] in ['currency', 'pt'] and \
                (selected['data_type'] == 'Daily').values[0]:
            var_list.append(it)
    else:  # use all units
        var_list.append(it)

    return var_list


def script_run(f_name=None):
    if f_name is None:  # Demo Test
        f_index = './datasets/rawdata/index_data/data_vars_US10YT_Indices.csv'
        max_x = 300
        assert False, 'Demo Disabled'
    else:  # header configured
        f_index = f_name
        max_x = RUNHEADER.max_x
        print('script_run - f_index: {}'.format(f_index))
        print('script_run - max_x: {}'.format(max_x))
    f_summary = RUNHEADER.var_desc

    # load data
    d_f_index = pd.read_csv(f_index, header=None).values.squeeze()
    d_f_summary = pd.read_csv(f_summary)

    # get variables except derived variables
    b_use_derived_vars = False
    var_list = list()
    for it in d_f_index:
        if '-' in it:
            if b_use_derived_vars:
                # Not in use -
                a, b = it.split('-')
                var_list = add_item(a, d_f_summary, var_list)
                var_list = add_item(b, d_f_summary, var_list)
            else:
                pass
        else:
            var_list = add_item(it, d_f_summary, var_list)

    # merge & save
    source_1_head = get_unique_list(d_f_index)[:int(max_x*0.5)]
    source_1_tail = get_unique_list(d_f_index)[int(max_x * 0.5):]
    source_2_head = get_unique_list(var_list)[:int(max_x*0.5)]

    my_final_list = OrderedDict.fromkeys(source_1_head + source_2_head)
    my_final_list = list(my_final_list) + source_1_tail
    pd.DataFrame(data=my_final_list, columns=['VarName']). \
        to_csv(f_index, index=None, header=None)
    print('{} has been saved'.format(f_index))

    # save desc
    basename = f_index.split('.csv')[0]
    write_var_desc(my_final_list, d_f_summary, basename)

    # var_desc = list()
    # for it in my_final_list:
    #     if '-' in it:
    #         for cnt in range(2):
    #             Condition = d_f_summary['var_name'] == it.split('-')[cnt]
    #             tmp = d_f_summary[Condition].values.squeeze().tolist()[1:]
    #             var_desc.append(tmp)
    #     else:
    #         Condition = d_f_summary['var_name'] == it
    #         tmp = d_f_summary[Condition].values.squeeze().tolist()[1:]
    #         var_desc.append(tmp)
    # pd.DataFrame(data=var_desc, columns=d_f_summary.keys()[1:]). \
    #     to_csv(basename + '_desc.csv')
    # print('{} has been saved'.format(f_index.split('.csv')[0] + '_desc.csv'))


def write_var_desc(my_final_list, d_f_summary, basename):
    # save desc
    var_desc = list()
    for it in my_final_list:
        if '-' in it:
            for cnt in range(2):
                Condition = d_f_summary['var_name'] == it.split('-')[cnt]
                tmp = d_f_summary[Condition].values.squeeze().tolist()[1:]
                var_desc.append(tmp)
        else:
            Condition = d_f_summary['var_name'] == it
            tmp = d_f_summary[Condition].values.squeeze().tolist()[1:]
            var_desc.append(tmp)
    pd.DataFrame(data=var_desc, columns=d_f_summary.keys()[1:]). \
        to_csv(basename + '_desc.csv')
    print('{} has been saved'.format(basename + '_desc.csv'))


if __name__ == '__main__':
    script_run()







