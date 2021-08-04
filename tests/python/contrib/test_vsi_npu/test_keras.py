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
import tvm
from tvm import te
import tvm.relay as relay
from tvm import rpc
from tvm.contrib import graph_runtime

import tensorflow.keras as keras
import numpy as np
from tvm.contrib.download import download_testdata
from PIL import Image
from tflite_models import *

RPC_HOST = ""
RPC_PORT = 9090
MEASURE_PERF = False
def inference_remotely(input_name, lib_path, image_data):
    remote = rpc.connect(RPC_HOST, RPC_PORT)
    remote.upload(lib_path)
    lib = remote.load_module(os.path.basename(lib_path))
    ctx = remote.cpu()

    #intrp = relay.build_module.create_executor(
    #                "graph", lib, ctx, target
    #            )
    #tvm_output = intrp.evaluate()(input_data, **params).asnumpy()


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


def load_keras_model(weights_url, model_name, weights_file):
    kera_models = {
        "ResNet50": keras.applications.ResNet50,
    }

    if model_name not in kera_models:
        raise Exception(f"not supported kera model: {model_name}")

    model_path = download_testdata(weights_url, weights_file, module="keras")
    model = kera_models[model_name](
                 include_top=True, weights=None, input_shape=(224, 224, 3),
                 classes=1000
             )
    model.load_weights(model_path)

    # get the HWC value from picture
    _, h, w, c = model.input_shape
    input_name = model.input_names[0]

    shape_dict = {input_name: (1, c, h, w)}

    # the shape needs to be NCHW
    return relay.frontend.from_keras(model, shape_dict), input_name


if tuple(keras.__version__.split(".")) < ("2", "4", "0"):
    weights_url = "".join(
        [
            "https://github.com/fchollet/deep-learning-models/releases/",
            "download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels.h5",
        ]
    )
    weights_file = "resnet50_keras_old.h5"
else:
    weights_url = "".join(
        [
            " https://storage.googleapis.com/tensorflow/keras-applications/",
            "resnet/resnet50_weights_tf_dim_ordering_tf_kernels.h5",
        ]
    )
    weights_file = "resnet50_keras_new.h5"

parser = argparse.ArgumentParser(description='VSI-NPU test script for keras models.')
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


(mod, params), input_name = load_keras_model(weights_url, "ResNet50", weights_file)

img = load_test_image()
data = np.array(img)[np.newaxis, :].astype("float32")
data = keras.applications.resnet50.preprocess_input(data)
data = data.transpose([0, 3, 1, 2])


LIB_PATH = "./model.so"
cross_compile_model(mod, params, verbose=VERBOSE, lib_path=LIB_PATH)
vsi_out = inference_remotely(input_name, LIB_PATH, data)
print("vsi out: ", np.argmax(vsi_out))

tvm_out = run_tvm_model(mod, params, data)
top1_tvm = np.argmax(tvm_out[0])
print("ref out: ", top1_tvm)
