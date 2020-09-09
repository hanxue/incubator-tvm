# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=no-else-return, unidiomatic-typecheck, invalid-name, W0611, C0413

""" Vitis-AI runtime test """

import sys
import numpy as np

import pytest
pytest.importorskip('pyxir')
import pyxir.contrib.target.DPUCADX8G

import tvm
import tvm.relay.testing
from tvm import relay

from infrastructure import skip_test, build_and_run, verify_codegen, verify_result

def test_extern_vitis_ai_resnet18():
    """Test resnet18 model using Vitis-AI byoc flow"""
    if skip_test():
        return

    dtype = 'float32'
    ishape = (1, 3, 224, 224)
    mod, params = relay.testing.resnet.get_workload(num_layers=18, batch_size=1)
    ref_mod, params = relay.testing.resnet.get_workload(num_layers=18, batch_size=1)

    ref_ex = relay.create_executor("graph", mod=ref_mod, ctx=tvm.cpu(0))
    i_data = np.random.uniform(0, 1, ishape).astype(dtype)

    ref_res = ref_ex.evaluate()(i_data, **params)
    verify_result(mod, {"data": i_data},
                  (1, 1000), ref_res.asnumpy(),
                  tol=1e-5, params=params,
                  dpu_target='DPUCADX8G', tvm_ops=4)

if __name__ == "__main__":
    if sys.platform == "win32":
        print("Skip test on Windows for now")
        sys.exit(0)
    test_extern_vitis_ai_resnet18()
