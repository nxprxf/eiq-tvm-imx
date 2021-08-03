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

import onnx
import onnxruntime as rt
import numpy as np
import tvm
from tvm import te
from tvm import rpc
from tvm.contrib import graph_runtime
import tvm.relay as relay
from tvm.contrib.download import download_testdata
from PIL import Image
from tflite_models import *

RPC_HOST = "10.193.20.195"
RPC_PORT = 9090
MEASURE_PERF = False
def inference_remotely(input_name, lib_path, image_data):
    remote = rpc.connect(RPC_HOST, RPC_PORT)
    remote.upload(lib_path)
    lib = remote.load_module(os.path.basename(lib_path))
    ctx = remote.cpu()

    # Create a runtime executor module
    module = graph_runtime.GraphModule(lib["default"](ctx))
    # Feed input data
    module.set_input(input_name, tvm.nd.array(image_data))

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


def softmax(x):
    x = x.reshape(-1)
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def preprocess(input_data):
    img_data = input_data.astype("float32")

    # normalize
    mean_vec = np.array([0.485, 0.456, 0.406])
    stddev_vec = np.array([0.229, 0.224, 0.225])
    norm_img_data = np.zeros(img_data.shape).astype('float32')
    for i in range(img_data.shape[0]):
        norm_img_data[i, :, :] = (img_data[i, :, :]/255 - mean_vec[i]) / stddev_vec[i]

    norm_img_data = norm_img_data.reshape(1, 3, 224, 224).astype('float32')
    return norm_img_data


def load_onnx_model(model_url, model_name):
    model_path = download_testdata(model_url, model_name, module="onnx")
    print(model_path)
    onnx_model = onnx.load(model_path)

    input_name = onnx_model.graph.input[0].name
    dims = onnx_model.graph.input[0].type.tensor_type.shape.dim
    shape = list([d.dim_value for d in dims])
    shape[0] = 1 # set batch size to 1
    shape_dict = {input_name: shape}
    print(shape_dict)

    return relay.frontend.from_onnx(onnx_model, shape_dict)


model_url = "".join(
    [
     "https://s3.amazonaws.com/onnx-model-zoo/resnet/resnet50v1/resnet50v1.onnx"
     #"https://media.githubusercontent.com/media/onnx/models/master/vision/classification/mobilenet/model/mobilenetv2-7.onnx"
    ]
)

img = load_test_image()
img_data = np.array(img).transpose(2, 0, 1)
x = preprocess(img_data)

model_path = download_testdata(model_url, "resnet50v1.onnx", module="onnx")
mod, params = load_onnx_model(model_url, "resnet50v1.onnx")
input_data = tvm.nd.array(x.astype("float32"))
tvm_output = run_tvm_model(mod, params, input_data)
tvm_output = softmax(tvm_output)
top1_tvm = np.argmax(tvm_output)
print("tvm out: ", top1_tvm)

# run model with corresponding engine runtime
sess = rt.InferenceSession(model_path)
input_name = sess.get_inputs()[0].name
ref_output = sess.run(None, {input_name: x})[0]
ref_output = softmax(ref_output)
print("ref out: ", np.argmax(ref_output))

LIB_PATH = "./model.so"
cross_compile_model(mod, params, lib_path=LIB_PATH)
vsi_out = inference_remotely(input_name, LIB_PATH, x)
print("vsi out: ", np.argmax(vsi_out))
