"""Contains utilities for downloading and converting datasets."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

domain_search_parameter = {
    'INX_20': 1, 'KS_20': 1, 'Gold_20': 1, 'FTSE_20': 1, 'GDAXI_20': 1, 'SSEC_20',: 1 'BVSP_20': 1, 'N225_20': 1,
    'INX_60': 2, 'KS_60': 2, 'Gold_60': 2, 'FTSE_60': 2, 'GDAXI_60': 2, 'SSEC_60',: 2 'BVSP_60': 2, 'N225_60': 2, 
    'INX_120': 3, 'KS_120': 3, 'Gold_120': 3, 'FTSE_120': 3, 'GDAXI_120': 3, 'SSEC_120',: 3 'BVSP_120': 3, 'N225_120': 3,
    'US10YT_20': 4, 'GB10YT_20': 4, 'DE10YT_20': 4, 'KR10YT_20': 4, 'CN10YT_20': 4, 'JP10YT_20': 4, 'BR10YT_20': 4,
    'US10YT_60': 5, 'GB10YT_60': 5, 'DE10YT_60': 5, 'KR10YT_60': 5, 'CN10YT_60': 5, 'JP10YT_60': 5, 'BR10YT_60': 5,
    'US10YT_120': 6, 'GB10YT_120': 6, 'DE10YT_120': 6, 'KR10YT_120': 6, 'CN10YT_120': 6, 'JP10YT_120': 6, 'BR10YT_120': 6,
}
mkidx_mkname = {
    0: 'INX',
    1: 'KS',
    2: 'Gold',
    3: 'US10YT',
    4: 'FTSE',
    5: 'GDAXI',
    6: 'SSEC',
    7: 'BVSP',
    8: 'N225',
    9: 'GB10YT',
    10: 'DE10YT',
    11: 'KR10YT',
    12: 'CN10YT',
    13: 'JP10YT',
    14: 'BR10YT',
}
forward_map = {20: 1, 60: 2, 120: 3}
mkname_mkidx = {v: k for k, v in mkidx_mkname.items()}
mkname_dataset = {v: 'v' + str(k+11) for k, v in mkidx_mkname.items()}


class ScriptParameters:
    
    def __init__(self, domain, job_id_int=1, search_parameter=None, n_cpu=0, s_test=None, e_test=None, on_cloud=1, operation_mode=1, actual_inference=0):
        
        self.operation_mode = operation_mode
        self.s_test = s_test
        self.e_test = e_test
        self.job_id_int = int(job_id_int)
        self.m_target_name = domain.split('_')[0]
        self.forward_idx = forward_map[int(domain.split('_')[1])]
        self.vervose = 3
        self.on_cloud = on_cloud
        if search_parameter is None:
            self.search_parameter = 0
        else
            self.search_parameter = domain_search_parameter[domain]
        
        self.m_target_index = mkname_mkidx[self.m_target_name]
        self.dataset_version = mkname_dataset[self.m_target_name]
        ref_pid = self.dataset_version[1:] + str(self.forward_idx)
                
        self.process_id = int(ref_pid + str(self.job_id_int))
        self.ref_pid = int(ref_pid)
            
