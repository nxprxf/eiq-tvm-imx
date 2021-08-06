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

import torch
import torchvision
from torchvision import transforms
import numpy as np
import tvm
from tvm import rpc
from tvm.contrib import graph_runtime
from tvm import te
import tvm.relay as relay
from tvm.contrib.download import download_testdata
from tvm.contrib import graph_runtime
from PIL import Image
from tflite_models import *


RPC_HOST = ""
RPC_PORT = 9090
MEASURE_PERF = False
VERBOSE = False
def inference_remotely(tfmodel, lib_path, image_data):
    remote = rpc.connect(RPC_HOST, RPC_PORT)
    remote.upload(lib_path)
    lib = remote.load_module(os.path.basename(lib_path))
    ctx = remote.cpu()

    # Create a runtime executor module
    module = graph_runtime.GraphModule(lib["default"](ctx))
    # Feed input data
    module.set_input("input0", tvm.nd.array(image_data))

    if MEASURE_PERF:
        print("Evaluate graph runtime inference cost on VSI NPU")
        ftimer = module.module.time_evaluator("run", ctx, number=1, repeat=10)
        # Measure in millisecond.
        prof_res = np.array(ftimer().results) * 1000
        print("VSI NPU runtime inference time (std dev): %.2f ms (%.2f ms)"
              % (np.mean(prof_res), np.std(prof_res)))
    module.run()
    tvm_output = module.get_output(0).asnumpy()

    return tvm_output


# Customize the image data for pytorch processing
def myprocess(image):
    p = transforms.Compose(
      [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
      ])

    return p(image)


# Download pytorch models
def get_pytorch_model(model_name):
    model = getattr(torchvision.models, model_name)(pretrained=True)
    model = model.eval()

    return model


# Download the picture to be tested as input
def get_img_data(shape, is_quant=False, myprocess=None):
    image_url =\
        "https://github.com/dmlc/mxnet.js/blob/main/data/cat.png?raw=true"

    image_path = download_testdata(image_url, "cat.png", module="data")
    resized_image = Image.open(image_path).resize(shape)

    if myprocess is not None:
        resized_image = myprocess(resized_image)

    DTYPE = "uint8" if is_quant else "float32"

    image_data = np.asarray(resized_image).astype(DTYPE)

    # Add a dimension to the image so that we have NHWC format layout
    image_data = np.expand_dims(image_data, axis=0)

    if not is_quant:
        # Preprocess image as described here:
        # https://github.com/tensorflow/models/blob/edb6ed22a801665946c63d650ab9a0b23d98e1b1/research/slim/preprocessing/inception_preprocessing.py#L243

        image_data[:, :, :, 0] = 2.0 / 255.0 * image_data[:, :, :, 0] - 1
        image_data[:, :, :, 1] = 2.0 / 255.0 * image_data[:, :, :, 1] - 1
        image_data[:, :, :, 2] = 2.0 / 255.0 * image_data[:, :, :, 2] - 1

    return image_data


def load_pytorch_model(model_name):
    # cannot decode model to get input_shape easily
    shape = (1, 3, 224, 224)
    model = getattr(torchvision.models, model_name)(pretrained=True)
    model = model.eval()

    input_data = torch.randn(shape)
    scripted_model = torch.jit.trace(model, input_data).eval()
    shape_list = [("input0", shape)]

    return relay.frontend.from_pytorch(scripted_model, shape_list)


# Compile the model
def compile_pytorch_model(input_shape, model_name):
    mod, params = load_pytorch_model(model_name)
    cross_compile_model(mod, params, verbose=VERBOSE)

    return mod, params


def get_ref_result(model_name, input_data):

    m = SUPPORTED_MODELS[model_name]
    inputs = m.inputs
    DTYPE = "uint8" if m.is_quant else "float32"

    mod, params = load_pytorch_model(model_name)

    tvm_out = run_tvm_model(mod, params, input_data)

    return tvm_out


class TORCH_Model:
    def __init__(self, name, inputs, input_size, is_quant=False):
        self.name = name
        self.inputs = inputs
        self.input_size = input_size
        self.is_quant = is_quant


models = [
          "mobilenet_v2",
          "resnet18",
          "resnet34",
          "resnet50",
          "resnet101",
          "resnet152",
          "inception_v3",
          "alexnet",
          "densenet121",
          "densenet161",
          "densenet169",
          "densenet201",
          "squeezenet1_0",
          "squeezenet1_1",
          "mnasnet0_5",
          "mnasnet1_0",
#          "shufflenet_v2_x0_5",
#          "shufflenet_v2_x1_0",
         ]
parser = argparse.ArgumentParser(description='VSI-NPU test script for pytorch models.')
parser.add_argument('-i', '--ip', type=str, required=True,
                    help='ip address for remote target board')
parser.add_argument('-p', '--port', type=int, default=9090,
                    help='port number for remote target board')
parser.add_argument('-m', '--models', nargs='*', default=SUPPORTED_MODELS,
                    help='models list to test')
parser.add_argument('--perf', action='store_true',
                    help='benchmark performance')
parser.add_argument('--verbose', action='store_true',
                    help='print more logs')

args = parser.parse_args()

RPC_HOST = args.ip
RPC_PORT = args.port
MEASURE_PERF = args.perf
VERBOSE = args.verbose


SUPPORTED_MODELS = {}

for m in models:
    mod = TORCH_Model(m, "input0", 224)
    SUPPORTED_MODELS[m] = mod


labels = load_test_label()
input_shape = [1, 3, 224, 224]
image_data = get_img_data(input_shape[2:], myprocess=myprocess)

for name, m in SUPPORTED_MODELS.items():
    print(f"\n========== Testing {name} ==========")
    # We grab the TorchScripted model via tracing
    try:
        mod, params = compile_pytorch_model(input_shape, m.name)
        tvm_output = get_ref_result(m.name, image_data)
    except Exception as err:
        print(f"FAIL ==> {name} failed with {err}")
        continue

    # Get top-1 result for TVM
    top1_tvm = np.argmax(np.squeeze(tvm_output))

    # Convert input to PyTorch variable and get PyTorch result for comparison
    with torch.no_grad():
        model = get_pytorch_model(m.name)
        output = model(torch.from_numpy(image_data))

        # Get top-1 result for PyTorch
        top1_torch = np.argmax(output.numpy())

    LIB_PATH = "./model.so"
    vsi_out = inference_remotely(mod, LIB_PATH, image_data)
    top1_vsi = np.argmax(vsi_out)

    print(f"Relay top-1 id: {top1_vsi}, class name: {labels[top1_vsi]}")
    print(f"Relay top-1 id: {top1_tvm}, class name: {labels[top1_tvm]}")
    print(f"Torch top-1 id: {top1_torch}, class name: {labels[top1_torch]}")
