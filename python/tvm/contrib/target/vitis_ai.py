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
# pylint: disable=invalid-name, unused-argument, import-outside-toplevel
"""Utility to compile VAI models"""

import os
import shutil
import numpy as np
import json

from tvm.relay.expr_functor import ExprVisitor
from tvm import relay
from .. import vitis_ai_runtime
import tvm._ffi
from tvm.relay.expr import If, Tuple, TupleGetItem, Call
from tvm.relay.function import Function
from tvm.relay import transform
from tvm.relay.op.annotation import compiler_begin, compiler_end

import pyxir

@transform.function_pass(opt_level=0)
class VitisAIAnnotationPass:
    def __init__(self, compiler, relay_ids):
        self.compiler = compiler
        self.relay_ids = relay_ids

    def transform_function(self, func, mod, ctx):

        annotator = self
        class Annotator(tvm.relay.ExprMutator):

            def visit_tuple(self, expr):
                field_list = []
                cond = int(hash(expr))
                for field in expr.fields:    
                    if ( cond in annotator.relay_ids ):
                        field_list.append(compiler_begin(super().visit(field), annotator.compiler))
                    else:
                        field_list.append(super().visit(field))
                if cond in annotator.relay_ids:
                    return compiler_end(Tuple(field_list), annotator.compiler)
                else:
                    return Tuple(field_list)

            def visit_tuple_getitem(self, expr):
              
                if ( int(hash(expr.tuple_value)) in annotator.relay_ids ):
                    tuple_value = compiler_begin(super().visit(expr.tuple_value), annotator.compiler)
                    return compiler_end(TupleGetItem(tuple_value, expr.index), annotator.compiler)
                else:
                    tuple_value = super().visit(expr.tuple_value)
                    return TupleGetItem(tuple_value, expr.index)
                
            def visit_call(self, call):

                if ( int(hash(call)) in annotator.relay_ids ):
                    new_args = []
                    for arg in call.args:
                        ann = compiler_begin(super().visit(arg),
                                             annotator.compiler)
                        new_args.append(ann)
                    new_call = relay.Call(call.op, new_args, call.attrs,
                                          call.type_args)
                    return compiler_end(new_call, annotator.compiler)

                else:
                    return super().visit_call(call)
        return Annotator().visit(func)



def annotation(mod, params, target):
    """
    An annotator for VAI.
    """
    xgraph = pyxir.frontend.tvm.from_relay(mod,params,postprocessing = None)
    xgraph = pyxir.partition(xgraph, targets=[target]) 
    layers = xgraph.get_layers()
    relay_ids = [list(np.array(layer.attrs['relay_id']).flatten()) for layer in layers if layer.target == target]
    relay_ids_flatten = [item for sublist in relay_ids for item in sublist]
    mod = VitisAIAnnotationPass("vai", relay_ids_flatten)(mod)
    return mod


class CodegenVitisAI:
    """
    Traverse subgraphs and build XGraph
    """
    def __init__(self, model_name, function):

        self.model_name = model_name
        self.function = function
        self.params = {}



    def convert_pyxir(self, target, out_dir):
        """
         Convert relay submodule expression to PYXIR(XGRAPH)
        """
        xgraph = pyxir.frontend.tvm.from_relay(self.function,
                        params         = self.params,
                        postprocessing = None)

        xgraph = pyxir.partition(xgraph, targets=[target])
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        pyxir.graph.io.xgraph_io.XGraphIO.save(xgraph, out_dir + 'dpu_xgraph')
        return xgraph

    def get_output_names(self):
        """
        Get output names from subgraph
        """
        func = self.function
        output_relay_ids = []
        expr = func.body
        if isinstance(expr, Tuple):
            for field in expr.fields:
                output_relay_ids.append(hash(field))
        elif isinstance(expr, Call):
            output_relay_ids.append(hash(expr))
        else:
            raise ValueError("does not support {}".format(type(expr)))    
        return output_relay_ids

@tvm._ffi.register_func("relay.ext.vai")
def vai_compiler(ref):
    """
    Create a VAI runtime from a Relay module.
    """
    pass_context = tvm.get_global_func("transform.GetCurrentPassContext")()
    model_dir = os.getcwd()
    out_tensor_names = []
    target = str(pass_context.config['target_'])
    assert isinstance(ref, tvm.relay.function.Function)
    name = str(ref.attrs.global_symbol)
    builder = CodegenVitisAI(name, ref)
    model_dir = target + "_build/"
    xgraph = builder.convert_pyxir( target, model_dir)
    output_relay_ids = builder.get_output_names()
    layers = xgraph.get_layers()
    # get the output tensor names using xgraph and output relay ids
    out_tensor_names= []
    for layer in layers:
        if not layer.internal:
            if layer.attrs['relay_id'][0] in output_relay_ids:
                out_tensor_names.append(layer.name)     
    return vitis_ai_runtime.create(name, model_dir, target, out_tensor_names).module