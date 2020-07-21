/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file vai_runtime.cc
 */
#include <sstream>
#include <tvm/runtime/registry.h>
#include <bits/stdc++.h> 
#include <tvm/ir/transform.h>
#include "vai_runtime.h"


void splitSen(std::string str, std::vector<std::string> &out_tensor_names) 
{ 
   std::string w = ""; 
   for (auto rem : str) 
   { 
       if (rem==',') 
       { 
          out_tensor_names.push_back(w);  
          w=""; 
       } 
       else
       { 
           w=w+rem; 
       } 
   }  
   //cout<<w<<endl;
   out_tensor_names.push_back(w); 
}

namespace tvm {
namespace runtime {


std::shared_ptr<pyxir::graph::XGraph> load_xgraph_model(const std::string& model_path) {
  std::string model_name = model_path + "/" + "dpu_xgraph.json";
  std::string model_weights = model_path + "/" + "dpu_xgraph.h5";
  return pyxir::load(model_name, model_weights);
}

void VaiRuntime::Init(const std::string& model_path, const std::string& target,   const std::vector<std::string> out_tensor_names  ) {
  model_path_ = model_path;
  target_ = target;
  xgraph_ = load_xgraph_model(model_path_);
  pyxir::partition(xgraph_, std::vector<std::string>{target},"");
  pyxir::RunOptionsHolder run_options(new pyxir::runtime::RunOptions());
  run_options->on_the_fly_quantization = true;
  in_tensor_names_ = xgraph_->get_input_names();
  //out_tensor_names_ =  xgraph->get_output_names();
  out_tensor_names_ =  out_tensor_names;
  rt_mod_ = pyxir::build_rt(xgraph_, target_ , in_tensor_names_, out_tensor_names_,
                                                                "vai", run_options);
}


//Module VaiRuntimeCreate(const std::string& name, const std::string& model_path, const std::string& target,const std::string out_tensor_names) {
  Module VaiRuntimeCreate(const std::string& name, const std::string& model_path, const std::string& target,const Array<String>& out_tensor_names) {
   Array<String> const_vars;
   
  auto exec = make_object<VaiRuntime>(name,model_path, const_vars );
  std::vector<std::string>vec_out_tensor_names;
  //splitSen(out_tensor_names, vec_out_tensor_names);
  //std::vector <std::String> const_names;
    for (const auto& it : out_tensor_names) {
      vec_out_tensor_names.push_back(it);
    }
  exec->Init(model_path, target, vec_out_tensor_names);
  return Module(exec);
}



TVM_REGISTER_GLOBAL("tvm.vai_runtime.create").set_body([](TVMArgs args, TVMRetValue* rv) {
  *rv = VaiRuntimeCreate(args[0], args[1], args[2], args[3]);
});

Module VaiRuntimeLoadFromBinary(void* strm ) {
    dmlc::Stream* stream = static_cast<dmlc::Stream*>(strm);

    std::string model_path;
    std::string symbol_name;
    std::vector<std::string> out_tensor_names;
    std::vector<std::string> const_vars;
    std::string target;
    stream->Read(&model_path);
    stream->Read(&out_tensor_names);
    stream->Read(&target);
    stream->Read(&symbol_name);
    stream->Read(&const_vars);
    Array<String> const_names;
    for (const auto& it : const_vars) {
      const_names.push_back(it);
    }
    auto exec = make_object<VaiRuntime>(symbol_name, model_path, const_names);
    exec->Init(model_path, target, out_tensor_names);
    return Module(exec);
  }

TVM_REGISTER_GLOBAL("runtime.module.loadbinary_VaiRuntime").set_body_typed(VaiRuntimeLoadFromBinary);

TVM_REGISTER_PASS_CONFIG_OPTION("target_", String);
void VaiRuntime::SaveToBinary( dmlc::Stream* stream)   {
   stream->Write(this-> model_path_);
   stream->Write(this-> out_tensor_names_);
   stream->Write(this-> target_);
   stream->Write(this->symbol_name_);
   std::vector<std::string> consts;
   for (const auto& it : const_names_) {
      consts.push_back(it);
    }
    stream->Write(consts);


   
  }


PackedFunc VaiRuntime::GetFunction(const std::string& name,
                                      const ObjectPtr<Object>& sptr_to_self) {
 
 
  if (name == "get_symbol") {
      return PackedFunc(
          [sptr_to_self, this](TVMArgs args, TVMRetValue* rv) { *rv = this->symbol_name_; });
    } else if (name == "get_const_vars") {
      return PackedFunc(
          [sptr_to_self, this](TVMArgs args, TVMRetValue* rv) { *rv = this->const_names_; });
    }
    else if ("__init_" + this->symbol_name_ == name) {
      // The function to initialize constant tensors.
      return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
        CHECK_EQ(args.size(), 1U);
        this->initialized_ = true;
        *rv = 0;
      });
    }
      else {
       return PackedFunc(
        [sptr_to_self, this](TVMArgs args, TVMRetValue* rv) { 

        DLTensor* inputs=args[0]; 
        pyxir::XBufferHolder xb_in = std::shared_ptr<pyxir::XBuffer>(
            new pyxir::XBuffer((void *) static_cast<float*>(inputs->data), 4, "f", 4,
                              std::vector<ssize_t>{1, inputs->shape[1],  inputs->shape[2],  inputs->shape[3]}, false, false));
        std::vector<pyxir::XBufferHolder> out_tensors;

      for (unsigned i = 0; i < out_tensor_names_.size(); ++i) {
          auto shape = xgraph_->get(out_tensor_names_[i])->shapes[0];
          std::cout<<out_tensor_names_[i]<<std::endl;
          std::vector<ssize_t> out_shape{shape.begin(), shape.end()};
          //TODO: Add batch size as a parameter
          out_shape[0] = 1; //Batch size
          std::vector<int64_t> ort_shape{out_shape.begin(), out_shape.end()};

          DLTensor* output_tensor = args[args.size() - out_tensor_names_.size()+i];
          void* output_data = (void *) static_cast<float*>(output_tensor->data);
          out_tensors.push_back(std::shared_ptr<pyxir::XBuffer>(
            new pyxir::XBuffer(output_data, 4, "f", out_shape.size(), out_shape,
                                false, false)));
        }

        std::vector<pyxir::XBufferHolder> in_tensors{xb_in};
        rt_mod_->execute(in_tensors, out_tensors);
 


        });
      }
  }
      
  



}  // namespace runtime
}  // namespace tvm
