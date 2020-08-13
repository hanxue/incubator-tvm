# Vitis-AI Integration
[Vitis-AI](https://github.com/Xilinx/Vitis-AI) is Xilinx's development stack for hardware-accelerated AI inference on Xilinx platforms, including both edge devices and Alveo cards. It consists of optimized IP, tools, libraries, models, and example designs. It is designed with high efficiency and ease of use in mind, unleashing the full potential of AI acceleration on Xilinx FPGA and ACAP.

The current Vitis-AI Byoc flow inside TVMRuntime enables acceleration of Neural Network model inference using DPUCADX8G. DPUCADX8G is a hardware accelerator for Convolutional Neural Networks (CNN) on top of the Xilinx [Alveo](https://www.xilinx.com/products/boards-and-kits/alveo.html) platform and targets U200 and U250 accelerator cards.

On this page you will find information on how to [build](#Build) TVMRuntime with Vitis-AI and on how to [get started](#Getting-started) with an example.

## Build

For building TVMRuntime with the Vitis-AI Byoc flow, you will have to setup the hardware environment and build the docker, see [build steps](#Hardware-setup-and-docker-build).

### System requirements

The following table lists system requirements for running docker containers as well as Alveo cards.  


| **Component**                                       | **Requirement**                                            |
|-----------------------------------------------------|------------------------------------------------------------|
| Motherboard                                         | PCI Express 3\.0\-compliant with one dual\-width x16 slot  |
| System Power Supply                                 | 225W                                                       |
| Operating System                                    | Ubuntu 16\.04, 18\.04                                      |
|                                                     | CentOS 7\.4, 7\.5                                          |
|                                                     | RHEL 7\.4, 7\.5                                            |
| CPU                                                 | Intel i3/i5/i7/i9/Xeon 64-bit CPU                          |
| GPU \(Optional to accelerate quantization\)         | NVIDIA GPU with a compute capability > 3.0                 |
| CUDA Driver \(Optional to accelerate quantization\) | nvidia\-410                                                |
| FPGA                                                | Xilinx Alveo U200 or U250                                  |
| Docker Version                                      | 19\.03\.1                                                  |

### Hardware setup and docker build

1. Clone the Vitis AI repository:
    ```
    git clone https://github.com/xilinx/vitis-ai
    ```
2. Install the Docker, and add the user to the docker group. Link the user to docker installation instructions from the following docker's website:
    * https://docs.docker.com/install/linux/docker-ce/ubuntu/
    * https://docs.docker.com/install/linux/docker-ce/centos/
    * https://docs.docker.com/install/linux/linux-postinstall/
3. Any GPU instructions will have to be separated from Vitis AI.
4. Set up Vitis AI to target Alveo cards. To target Alveo cards with Vitis AI for machine learning workloads, you must install the following software components:
    * Xilinx Runtime (XRT)
    * Alveo Deployment Shells (DSAs)
    * Xilinx Resource Manager (XRM) (xbutler)
    * Xilinx Overlaybins (Accelerators to Dynamically Load - binary programming files)

    While it is possible to install all of these software components individually, a script has been provided to automatically install them at once. To do so:
      * Run the following commands:
        ```
        cd Vitis-AI/alveo/packages
        sudo su
        ./install.sh
        ```
      * Power cycle the system.
5. Clone tvm repo
    ```
    git clone --recursive https://github.com/apache/incubator-tvm.git
    ```
6. Build and start the tvm runtime Vitis-AI Docker Container.
   ```
   cd tvm
   bash tvm/docker/build.sh ci_vai_1x bash 
   bash tvm/docker/bash.sh tvm.ci_vai_1x
   ```
   Setup inside container
   ```
   source /opt/xilinx/xrt/setup.sh
   conda activate vitis-ai-tensorflow
   ```

## Getting started

### On-the-fly quantization

Usually, to be able to accelerate inference of Neural Network models with Vitis-AI DPU accelerators, those models need to quantized upfront. In the TVMRuntime with Vitis-AI byoc flow, we make use of on-the-fly quantization to remove this additional preprocessing step. In this flow, one doesn't need to quantize his/her model upfront but can make use of the typical inference execution calls (InferenceSession.run) to quantize the model on-the-fly using the first N inputs that are provided (see more information below). This will set up and calibrate the Vitis-AI DPU and from that point onwards inference will be accelerated for all next inputs.

### Config/Settings

A couple of environment variables can be used to customize the Vitis-AI Byoc flow.

| **Environment Variable**   | **Default if unset**      | **Explanation**                                         |
|----------------------------|---------------------------|---------------------------------------------------------|
| PX_QUANT_SIZE              | 128                    | The number of inputs that will be used for quantization (necessary for Vitis-AI acceleration) |
| PX_BUILD_DIR               | Use the on-the-fly quantization flow | Loads the quantization and compilation information from the provided build directory and immediately starts Vitis-AI hardware acceleration. This configuration can be used if the model has been executed before using on-the-fly quantization during which the quantization and comilation information was cached in a build directory. |

### Usage
This example shows how to build a TVM convolutional neural network model with Relay for Vitis-AI acceleration.

Annotate and partition the graph for Vitis-AI DPU

Create a relay graph from frontend model. Annotate the graph for the given Vitis-AI DPU (Deep Learning Processing Unit) target using PyXIR.
```
import pyxir
import pyxir.contrib.target.DPUCADX8G
import pyxir.contrib.target.DPUCZDX8G

import tvm
import tvm.relay as relay
from tvm.contrib.target import vitis_ai
from tvm.relay.build_module import bind_params_by_name
from tvm.relay.op.contrib.vitis_ai import annotation

mod["main"] = bind_params_by_name(mod["main"], params)
mod = annotation(mod, params, target)
mod = relay.transform.MergeCompilerRegions()(mod)
mod = relay.transform.PartitionGraph()(mod)
```
Build the Relay graph
```
tvm_target = 'llvm'
target='DPUCADX8G'

with tvm.transform.PassContext(opt_level=3, config= {'target_': target}):   
	graph, lib, params = relay.build(mod, tvm_target, params=params)
```

Run Inference

 We make use of on-the-fly calibration to remove the additional preprocessing step and do not need to explicitly call quantization on relay. Using this method one doesn’t need to quantize their model upfront and can make use of the typical inference execution calls (module.run) to calibrate the model on-the-fly using the first N inputs that are provided. After first N iterrations, computations will be accelerated on the FPGA.
```
module = tvm.contrib.graph_runtime.create(graph, lib, tvm.cpu())
module.set_input(**params)
# First N (default = 128) inputs are used for quantization calibration and will
# be executed on the CPU
# This config can be changed by setting the 'PX_QUANT_SIZE' (e.g. export PX_QUANT_SIZE=64)
for i in range(128):
    module.set_input(input_name, inputs[i]) 
    print("running") 
    module.run()
# Afterwards, computations will be accelerated on the FPGA
module.set_input(name, data)
module.run()
```

Save and Load Compiled Module

We can also save the graph, lib and parameters into files and load them back in deploy environment.

```
# save the graph, lib and params into separate files
from tvm.contrib import util

temp = util.tempdir()
path_lib = temp.relpath("deploy_lib.so")
lib.export_library(path_lib)
with open(temp.relpath("deploy_graph.json"), "w") as fo:
    fo.write(graph)
with open(temp.relpath("deploy_param.params"), "wb") as fo:
    fo.write(relay.save_param_dict(params))
print(temp.listdir())
```
Load the module from compiled files and run inference

```
# load the module back.
loaded_json = open(temp.relpath("deploy_graph.json")).read()
loaded_lib = tvm.runtime.load_module(path_lib)
loaded_params = bytearray(open(temp.relpath("deploy_param.params"), "rb").read())
module = tvm.contrib.graph_runtime.create(loaded_json, loaded_lib, ctx)
module.load_params(loaded_params)
module.set_input(name, data)
module.run()
```