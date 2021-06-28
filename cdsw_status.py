import re
import os
import sys
from PIL import Image
import glob
import platform
import numpy as np
import header.index_forecasting.RUNHEADER as RUNHEADER
import argparse
import datetime
import util
import pickle
from util import get_domain_on_CDSW_env
import random

domain_search_parameter = {
    'INX_20': 1, 'KS_20': 1, 'Gold_20': 1, 'FTSE_20': 1, 'GDAXI_20': 1, 'SSEC_20': 1, 'BVSP_20': 1, 'N225_20': 1,
    'INX_60': 2, 'KS_60': 2, 'Gold_60': 2, 'FTSE_60': 2, 'GDAXI_60': 2, 'SSEC_60': 2, 'BVSP_60': 2, 'N225_60': 2, 
    'INX_120': 3, 'KS_120': 3, 'Gold_120': 3, 'FTSE_120': 3, 'GDAXI_120': 3, 'SSEC_120': 3, 'BVSP_120': 3, 'N225_120': 3,
    'US10YT_20': 4, 'GB10YT_20': 4, 'DE10YT_20': 4, 'KR10YT_20': 4, 'CN10YT_20': 4, 'JP10YT_20': 4, 'BR10YT_20': 4,
    'US10YT_60': 5, 'GB10YT_60': 5, 'DE10YT_60': 5, 'KR10YT_60': 5, 'CN10YT_60': 5, 'JP10YT_60': 5, 'BR10YT_60': 5,
    'US10YT_120': 6, 'GB10YT_120': 6, 'DE10YT_120': 6, 'KR10YT_120': 6, 'CN10YT_120': 6, 'JP10YT_120': 6, 'BR10YT_120': 6,
}


def write_file(filename, val):
    fp = open(filename, 'w')
    fp.write(val)
    fp.close()

def read_file(filename):
    fp = open(filename, 'r')
    val = fp.readline().replace('\n','')
    fp.close()
    return val

if __name__ == "__main__":
    parser = argparse.ArgumentParser("")
    parser.add_argument("--update_system_status", type=int, default=0)
    args = parser.parse_args()

    if args.update_system_status:  # auto clean 이후 시작
        write_file('./cdsw_status.txt', 'system_idle')
    else:  # script_all_in_one 앞에 시작
        system_idel = read_file('./cdsw_status.txt')
        if system_idel == 'system_idle':
            sl = list(domain_search_parameter.keys())
            random.shuffle(sl)
            for it in sl:
                _, forward_ndx = it.split('_')
                cond = '_T'.join(it.split('_'))
                if os.path.isfile('./save/model_repo_meta/{}.pkl'.format(cond)):
                    pass
                else:
                    write_file('./cdsw_{}.txt'.format(forward_ndx), it)
                    write_file('./cdsw_status.txt', 'system_busy')
        else:
            print('system_busy')
            sys.exit()


        

    