import torch
import torchvision
from torchvision import transforms
import numpy as np
import tvm
from tvm import te
import tvm.relay as relay
from tvm.contrib.download import download_testdata
from tvm.contrib import graph_runtime
from PIL import Image
from tflite_models import *

# Customize the image data for pytorch processing
def myprocess(image):
    p = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
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

# Compile the model
def compile_pytorch_model(shape, model_name, verbose=False, lib_path="./model.so"):
    m = SUPPORTED_MODELS[model_name]

    model = get_pytorch_model(model_name)
    input_data = torch.randn(shape)
    scripted_model = torch.jit.trace(model, input_data).eval()
    shape_list = [(m.inputs, shape)]
    mod, params = relay.frontend.from_pytorch(
            scripted_model, shape_list
    )        

    kwargs = {}
    kwargs["cc"] = "aarch64-linux-gnu-gcc"
    target = "llvm  -mtriple=aarch64-linux-gnu"

    with transform.PassContext(opt_level=3, disabled_pass=["AlterOpLayout"]):
        mod = vsi_npu.partition_for_vsi_npu(mod, params)
        if verbose:
            print(mod.astext(show_meta_data=False))
        lib = relay.build(mod, target, params=params)
        lib.export_library(lib_path, fcompile=False, **kwargs)


def get_ref_result(shape, model_name, image_data):

    m = SUPPORTED_MODELS[model_name]
    inputs = m.inputs
    DTYPE = "uint8" if m.is_quant else "float32"

    model = get_pytorch_model(model_name)
    input_data = torch.randn(shape)
    scripted_model = torch.jit.trace(model, input_data).eval()
    shape_list = [(m.inputs, shape)]
    mod, params = relay.frontend.from_pytorch(
            scripted_model, shape_list
    )        

    target = "llvm"
    with tvm.transform.PassContext(opt_level=3):
        lib = tvm.relay.build(mod, target, params=params)

    ctx = tvm.cpu()
    cpu_mod = graph_runtime.GraphModule(lib["default"](ctx))
    cpu_mod.set_input(inputs, image_data)

    cpu_mod.run()
    tvm_out = cpu_mod.get_output(0).asnumpy()
    return tvm_out


def get_label_index():
    label_file_url = "".join(
        [
        "https://raw.githubusercontent.com/",
        "tensorflow/tensorflow/master/tensorflow/lite/java/demo/",
        "app/src/main/assets/",
        "labels_mobilenet_quant_v1_224.txt",
        ]
    )
    label_file = "labels_mobilenet_quant_v1_224.txt"
    label_path = download_testdata(label_file_url, label_file, module="data")

    # List of 1001 classes
    with open(label_path) as f:
        labels = f.readlines()

        # remove the first "background"
        return labels[1:]


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
          "googlenet",
          "densenet121",
          "densenet161",
          "densenet169",
          "densenet201",
          "squeezenet1_0",
          "squeezenet1_1",
          "mnasnet0_5",
          "mnasnet0_75",
          "mnasnet1_0",
          "mnasnet1_3",
          "vgg11"
          "vgg11_bn"
          "vgg13"
          "vgg13_bn"
          "vgg16"
          "vgg16_bn"
          "vgg19"
          "vgg19_bn"
          "shufflenet_v2_x0_5",
          "shufflenet_v2_x1_0",
          "shufflenet_v2_x1_5",
          "shufflenet_v2_x2_0",
         ]

SUPPORTED_MODELS = {}

for m in models:
    mod = TORCH_Model(m, "input0", 224)
    SUPPORTED_MODELS[m] = mod


labels = get_label_index()
input_shape = [1, 3, 224, 224]
image_data = get_img_data(input_shape[2:], myprocess=myprocess)

for name, m in SUPPORTED_MODELS.items():
    print(f"\n========== Testing {name} ==========")
    # We grab the TorchScripted model via tracing
    try:
        compile_pytorch_model(input_shape, m.name)
        tvm_output = get_ref_result(input_shape, m.name, image_data)
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

    print("Relay top-1 id: {}, class name: {}".format(top1_tvm, labels[top1_tvm]))
    print("Torch top-1 id: {}, class name: {}".format(top1_torch, labels[top1_torch]))
