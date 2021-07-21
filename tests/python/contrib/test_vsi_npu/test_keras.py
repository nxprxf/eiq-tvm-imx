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
import keras
import numpy as np
from tvm.contrib.download import download_testdata
from PIL import Image

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


weights_path = download_testdata(weights_url, weights_file, module="keras")
keras_resnet50 = keras.applications.ResNet50(
    include_top=True, weights=None, input_shape=(224, 224, 3), classes=1000
)
keras_resnet50.load_weights(weights_path)

img_url = "https://github.com/dmlc/mxnet.js/blob/main/data/cat.png?raw=true"
img_path = download_testdata(img_url, "cat.png", module="data")
img = Image.open(img_path).resize((224, 224))

input_name = "input_1"
data = np.array(img)[np.newaxis, :].astype("float32")
data = preprocess_input(data).transpose([0, 3, 1, 2])
shape_dict = {input_name: data.shape}
print(shape_dict)

mod, params = relay.frontend.from_keras(keras_resnet50, shape_dict)

target = "llvm"
with tvm.transform.PassContext(opt_level=3):
    executor = relay.build_module.create_executor("graph", mod, tvm.cpu(0), target)

dtype = "float32"
tvm_output = executor.evaluate()(tvm.nd.array(data.astype(dtype)), **params).asnumpy()

top1_tvm = np.argmax(tvm_outout.numpy()[0])
print(top1_tvm)
