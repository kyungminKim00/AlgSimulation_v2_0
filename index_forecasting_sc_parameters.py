"""Contains utilities for downloading and converting datasets."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import header.index_forecasting.RUNHEADER as RUNHEADER

class ScriptParameters:
    def __init__(self, job_id_int, domain, s_test=None, e_test=None):
        self.operation_mode = None

        if s_test is None or e_test is None:
            self.operation_mode = 0
