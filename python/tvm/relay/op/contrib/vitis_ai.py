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
# pylint: disable=invalid-name, unused-argument
"""VITISAI codegen supported operators."""

import os
import numpy as np

from tvm.relay.expr_functor import ExprVisitor
from tvm import relay
import tvm._ffi
from tvm.relay.expr import If, Tuple, TupleGetItem, Call
from tvm.relay.function import Function
from tvm.relay import transform
from tvm.relay.op.annotation import compiler_begin, compiler_end

import pyxir
import pyxir.frontend.tvm


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
    An annotator for VITISAI.
    """
    xgraph = pyxir.frontend.tvm.from_relay(mod,params,postprocessing = None)
    xgraph = pyxir.partition(xgraph, targets=[target]) 
    layers = xgraph.get_layers()
    relay_ids = [list(np.array(layer.attrs['relay_id']).flatten()) for layer in layers if layer.target == target]
    relay_ids_flatten = [item for sublist in relay_ids for item in sublist]
    mod = VitisAIAnnotationPass("vai", relay_ids_flatten)(mod)
    return mod