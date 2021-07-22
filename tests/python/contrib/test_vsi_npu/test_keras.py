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
import tensorflow.keras as keras
import numpy as np
from tvm.contrib.download import download_testdata
from PIL import Image
from tflite_models import *


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
    return relay.frontend.from_keras(model, shape_dict)


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

mod, params = load_keras_model(weights_url, "ResNet50", weights_file)

img = load_test_image()
data = np.array(img)[np.newaxis, :].astype("float32")
data = keras.applications.resnet50.preprocess_input(data)
data = data.transpose([0, 3, 1, 2])

tvm_out = run_tvm_model(mod, params, data)

top1_tvm = np.argmax(tvm_out[0])
print(top1_tvm)
