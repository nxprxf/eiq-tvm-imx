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
 * \file src/runtime/contrib/vsi_npu/dnnl_json_runtime.cc
 * \brief A simple JSON runtime for VsiNpu.
 */

#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/registry.h>

#include <cstddef>
#include <string>
#include <vector>

#include "../json/json_node.h"
#include "../json/json_runtime.h"

#ifdef USE_VSI_NPU_RUNTIME
#include "ovxlibxx/context.h"
#include "ovxlibxx/graph.h"
#include "ovxlibxx/tensor.h"
#include "ovxlibxx/operation.h"
#include "ovxlibxx/operations/fullyconnected.h"
#include "ovxlibxx/operations/activations.h"
#include "ovxlibxx/operations/softmax.h"
#include "ovxlibxx/operations/reshape.h"
#include "ovxlibxx/operations/pool2d.h"
#include "ovxlibxx/operations/conv2d.h"

#include "vsi_utils.h"
#endif

namespace tvm {
namespace runtime {
namespace contrib {

using namespace tvm::runtime;
using namespace tvm::runtime::json;

#ifdef USE_VSI_NPU_RUNTIME
class VsiNpuJSONRuntime : public JSONRuntimeBase {

 public:
  VsiNpuJSONRuntime(const std::string& symbol_name, const std::string& graph_json,
                  const Array<String> const_names)
      : JSONRuntimeBase(symbol_name, graph_json, const_names) {}

  const char* type_key() const { return "vsi_npu_json"; }

  void Init(const Array<NDArray>& consts) override {
    // Setup constants entries for weights.
    SetupConstants(consts);

    CHECK_EQ(consts.size(), const_idx_.size())
        << "The number of input constants must match the number of required.";

    BuildEngine();
  }

  void Run() override {
    for (size_t i = 0; i < input_nodes_.size(); ++i) {
      auto nid = input_nodes_[i];
      uint32_t eid = EntryID(nid, 0);
      if (nodes_[nid].GetOpType() == "input") {
	auto vsi_tensor = entry_out_tensor_[eid];

        void* data = data_entry_[eid]->data;
	int data_size = 1;
        for (int j = 0; j < data_entry_[eid]->ndim; j++) {
          data_size *= data_entry_[eid]->shape[j];
        }
        assert(vsi_tensor->CopyDataToTensor(data, data_size));
      }
    }

    assert(graph_->Run());

    for (size_t i = 0; i < outputs_.size(); ++i) {
      uint32_t eid = EntryID(outputs_[i]);
      void* data = data_entry_[eid]->data;

      auto vsi_tensor = entry_out_tensor_[eid];
      vsi_tensor->CopyDataFromTensor(data);
    }
  }
 private:

  void BuildEngine() {
    context_ = vsi::Context::Create();
    graph_ = context_->CreateGraph();

    for (size_t nid = 0; nid < nodes_.size(); ++nid) {
      const auto& node = nodes_[nid];
      if (node.GetOpType() == "kernel") {
        CHECK_EQ(node.GetOpType(), "kernel");
        auto op_name = node.GetOpName();
	LOG(INFO) << "Build op: " << op_name;
        if ("nn.batch_flatten" == op_name) {
	  Flatten(nid);
        } else if ("nn.dense" == op_name) {
	  Dense(nid);
        } else if ("nn.relu" == op_name) {
          Relu(nid);
        } else if ("nn.softmax" == op_name) {
          Softmax(nid);
        } else if ("nn.conv2d" == op_name) {
          Conv2D(nid);
        } else if (("nn.global_avg_pool2d" == op_name) || ("nn.global_max_pool2d" == op_name)) {
          GlobalPool2d(nid);
        } else if ("add" == op_name) {
        } else {
          LOG(FATAL) << "Unsupported op: " << op_name;
        }
      }
    }
    assert(graph_->Compile());
    LOG(INFO) << "Build graph successfully" << std::endl;
  }

  void Flatten(const size_t& nid) {
    auto node = nodes_[nid];
    JSONGraphNodeEntry out_entry(nid, 0);
    std::vector<JSONGraphNodeEntry> inputs = node.GetInputs();

    CHECK(inputs.size() == 1U) << "Flatten layer requires 1 inputs.";

    std::vector<std::shared_ptr<vsi::Tensor>> vsi_inputs;
    std::vector<std::shared_ptr<vsi::Tensor>> vsi_outputs;
    vsi_inputs.push_back(MakeVSITensorFromJSONEntry(inputs[0]));
    vsi_outputs.push_back(MakeVSITensorFromJSONEntry(out_entry));

    std::vector<int64_t> tvm_shape = nodes_[inputs[0].id_].GetOpShape()[0];
    uint32_t data_size = 1;
    for (unsigned int i = 0; i < tvm_shape.size(); i ++) {
      data_size *= tvm_shape[i];
    }

    std::vector<uint32_t> output_shape({data_size});
    auto flatten = graph_->CreateOperation<vsi::Reshape>(output_shape.data(), 1);
    (*flatten).BindInputs(vsi_inputs).BindOutputs(vsi_outputs);
  }

  void Dense(const size_t& nid) {
    auto node = nodes_[nid];
    // Collect inputs and outputs, handling both nn.dense and qnn.dense cases.
    std::vector<JSONGraphNodeEntry> inputs = node.GetInputs();
    size_t num_inputs = inputs.size();
    JSONGraphNodeEntry out_entry(nid, 0);
    
    std::vector<std::shared_ptr<vsi::Tensor>> vsi_inputs;
    std::vector<std::shared_ptr<vsi::Tensor>> vsi_outputs;

    if (node.GetOpName() == "qnn.dense") {
	    //qnn.dense
	    
    } else {
      CHECK(num_inputs >= 2U && num_inputs <= 3U)
          << "Fully connected (dense) layer requires 3 inputs with a bias, 2 inputs without.";
      for (const auto& i : inputs) {
        vsi_inputs.push_back(MakeVSITensorFromJSONEntry(i));
      }
      vsi_outputs.push_back(MakeVSITensorFromJSONEntry(out_entry));
    }

    auto weight_tensor = vsi_inputs[1];
    auto fc = graph_->CreateOperation<vsi::FullyConnected>(1, weight_tensor->GetShape()[1]);
    (*fc).BindInputs(vsi_inputs).BindOutputs(vsi_outputs);
  }

  void Relu(const size_t& nid) {

    auto node = nodes_[nid];
    //JSONGraphNodeEntry input
    auto data_entry = node.GetInputs()[0];

    JSONGraphNodeEntry out_entry(nid, 0);

    std::shared_ptr<vsi::Tensor> vsi_input;
    std::shared_ptr<vsi::Tensor> vsi_output;

    vsi_input = MakeVSITensorFromJSONEntry(data_entry);

    vsi_output = MakeVSITensorFromJSONEntry(out_entry);

    auto _op = graph_->CreateOperation<vsi::Relu>();
    (*_op).BindInput(vsi_input).BindOutput(vsi_output);

  }

  void Add(const size_t& nid) {
  }

  void GlobalPool2d(const size_t& nid) {

    vsi::PoolType pool_type;

    auto node = nodes_[nid];
    //JSONGraphNodeEntry input
    auto data_entry = node.GetInputs()[0];

    JSONGraphNodeEntry out_entry(nid, 0);

    if (node.GetOpName() == "nn.global_max_pool2d") {
      pool_type = vsi::PoolType::MAX;
    } else if (node.GetOpName() == "nn.global_avg_pool2d") {
      pool_type = vsi::PoolType::AVG;
    } else {
      LOG(FATAL) << "Pooling type not supported: " << node.GetOpName();
    }

    std::shared_ptr<vsi::Tensor> vsi_input;
    std::shared_ptr<vsi::Tensor> vsi_output;

    vsi_input = MakeVSITensorFromJSONEntry(data_entry);

    vsi_output = MakeVSITensorFromJSONEntry(out_entry);
    auto vsi_shap = vsi_input->GetShape();
    //layout is swapped, NCHW-->WHCN, [0]:W, [1]:H
    std::vector<uint32_t> ksize = {vsi_shap[0], vsi_shap[1]};
    //stride
    std::vector<uint32_t> stride = {1, 1};

    auto _op = graph_->CreateOperation<vsi::Pool2d>(pool_type, vsi::PadType::AUTO, ksize, stride);
    (*_op).BindInput(vsi_input).BindOutput(vsi_output);

  }

  void Softmax(const size_t& nid) {
    auto node = nodes_[nid];
    //JSONGraphNodeEntry input
    auto data_entry = node.GetInputs()[0];
    //softmax aixs
    auto axis_data_tvm = node.GetAttr<std::vector<std::string>>("axis")[0];
    auto shape_tvm = nodes_[data_entry.id_].GetOpShape()[data_entry.index_];
    uint32_t axis_data_vsi = 1;

    axis_data_vsi = ConvertAxis(std::stoi(axis_data_tvm), shape_tvm.size());

    JSONGraphNodeEntry out_entry(nid, 0);

    std::shared_ptr<vsi::Tensor> vsi_input;
    std::shared_ptr<vsi::Tensor> vsi_output;

    vsi_input = MakeVSITensorFromJSONEntry(data_entry);
    vsi_output = MakeVSITensorFromJSONEntry(out_entry);

    //set beta to 1.0
    auto _op = graph_->CreateOperation<vsi::Softmax>(1.0f, axis_data_vsi);
    (*_op).BindInput(vsi_input).BindOutput(vsi_output);
  }

  void Conv2D(const size_t& nid) {
    auto node = nodes_[nid];
    std::vector<std::string> pad = node.GetAttr<std::vector<std::string>>("padding");
    std::vector<std::string> strides = node.GetAttr<std::vector<std::string>>("strides");
    std::vector<std::string> dilation = node.GetAttr<std::vector<std::string>>("dilation");
    auto is_depthwise = node.GetAttr<std::vector<std::string>>("is_depthwise")[0];

    int groups = std::stoi(node.GetAttr<std::vector<std::string>>("groups")[0]);

    // Collect inputs and outputs, handling both nn.conv2d and qnn.conv2d cases.
    std::vector<JSONGraphNodeEntry> inputs = node.GetInputs();
    std::vector<std::shared_ptr<vsi::Tensor>> vsi_inputs;
    std::vector<std::shared_ptr<vsi::Tensor>> vsi_outputs;
    size_t num_inputs = inputs.size();
    JSONGraphNodeEntry out_entry(nid, 0);

    if (node.GetOpName() == "qnn.conv2d") {
      CHECK(num_inputs >= 8U && num_inputs <= 9U)
          << "Quantized convolution requires 9 inputs with a bias, 8 inputs without.";
    } else {
      CHECK(num_inputs >= 2U && num_inputs <= 3U)
          << "Convolution requires 3 inputs with a bias, 2 inputs without.";
      for (const auto& i : inputs) {
        vsi_inputs.push_back(MakeVSITensorFromJSONEntry(i));
      }
      vsi_outputs.push_back(MakeVSITensorFromJSONEntry(out_entry));
    }

    // TVM: top, left, bottom, right -> VSI: left, right, top, bottom
    auto weight_tensor = vsi_inputs[1];
    std::vector<uint32_t> vsi_pad;
    vsi_pad.push_back(std::stoi(pad[1]));
    vsi_pad.push_back(std::stoi(pad[3]));
    vsi_pad.push_back(std::stoi(pad[0]));
    vsi_pad.push_back(std::stoi(pad[2]));

    std::vector<uint32_t> vsi_strides;
    vsi_strides.push_back(std::stoi(strides[0]));
    vsi_strides.push_back(std::stoi(strides[1]));

    std::vector<uint32_t> vsi_dilation;
    vsi_dilation.push_back(std::stoi(dilation[0]));
    vsi_dilation.push_back(std::stoi(dilation[1]));

    std::vector<uint32_t> vsi_ksize;
    vsi_ksize.push_back(weight_tensor->GetShape()[0]);
    vsi_ksize.push_back(weight_tensor->GetShape()[1]);

    if (vsi_inputs.size() == 2) {
      vsi_inputs.push_back(MakeDummyBiasTensor(vsi_inputs[0]->GetDataType(),
			      {weight_tensor->GetShape()[3]}));
    }
    int32_t vsi_multiplier = 0;
    if (std::stoi(is_depthwise) == 1) {
      vsi_multiplier = static_cast<int32_t>(weight_tensor->GetShape()[2]);
    }

    auto fc = graph_->CreateOperation<vsi::Conv2d>(static_cast<int32_t>(weight_tensor->GetShape()[3]),
		    vsi::PadType::AUTO, vsi_ksize, vsi_strides, vsi_dilation,
		    vsi_pad, groups, vsi_multiplier);
    (*fc).BindInputs(vsi_inputs).BindOutputs(vsi_outputs);
  }


  void BatchFlatten(const size_t& nid) {
  }

  bool IsInputNode(const size_t& nid) {
    return std::find(input_nodes_.begin(), input_nodes_.end(), nid) != input_nodes_.end();
  }

  bool IsOutputNode(const size_t& nid) {
    int size = outputs_.size();
    for(int i = 0; i< size; i++) {
      if(outputs_[i].id_ == nid)
        return true;
    }
    return false;
  }

  /*!
   * \brief Create an VSI tensor given the JSON representation. If scale
   * and offset are given, then create a quantized VSI tensor.
   *
   * \param tensor The tensor to represent.
   * \param scale (optional) The scale of the tensor as an input.
   * \param offset (optional) The offset of the tensor as an input.
   * \return VSI Tensor.
   */
  std::shared_ptr<vsi::Tensor> MakeVSITensorFromJSONEntry(const JSONGraphNodeEntry& tensor,
                                                 JSONGraphNodeEntry* scale = nullptr,
                                                 JSONGraphNodeEntry* offset = nullptr) {
    auto eid = EntryID(tensor);

    if (entry_out_tensor_.count(eid) != 0) {
      //using the existed VSItensor
      return entry_out_tensor_[eid];
    }
    //create new VSItensor
    JSONGraphNode node = nodes_[tensor.id_];
    void* node_data = nullptr;
    vsi::TensorAttribute vsi_attr;

    if (node.GetOpType() == "const") {
      node_data = data_entry_[EntryID(tensor)]->data;
      vsi_attr = vsi::TensorAttribute::CONSTANT;
    } else if (IsInputNode(tensor.id_)) {
      vsi_attr = vsi::TensorAttribute::INPUT;
    } else if (IsOutputNode(tensor.id_)) {
      vsi_attr = vsi::TensorAttribute::OUTPUT;
    } else {
      vsi_attr = vsi::TensorAttribute::TRANSIENT;
    }

    auto vsi_tensor = MakeVSITensor(node, node_data, vsi_attr, scale, offset);
    entry_out_tensor_.insert({eid, vsi_tensor});
    return entry_out_tensor_[eid];
  }

  std::shared_ptr<vsi::Tensor> MakeDummyBiasTensor(vsi::DataType dtype,
		 		 vsi::ShapeType bias_shape) {
    std::vector<float> bias_data(bias_shape[0], 0);

    vsi::TensorSpec bias_spec(dtype, bias_shape,
		    vsi::TensorAttribute::CONSTANT);
    auto bias = graph_->CreateTensor(bias_spec, bias_data.data());
    dummy_tensor_.push_back(bias);

    return bias;
  }

  std::shared_ptr<vsi::Tensor> MakeVSITensor(const JSONGraphNode& tensor_rep, void* data,
				  vsi::TensorAttribute vsi_attr,
                                  JSONGraphNodeEntry* scale = nullptr,
                                  JSONGraphNodeEntry* offset = nullptr) {
    //VSI parameter
    vsi::ShapeType vsi_shape;
    vsi::DataType vsi_dtype;
    //TVM parameter
    std::vector<int64_t> tvm_shape = tensor_rep.GetOpShape()[0];
    DLDataType tvm_dtype = tensor_rep.GetOpDataType()[0];

    for (unsigned int i = 0; i < tvm_shape.size(); i ++) {
      vsi_shape.push_back(tvm_shape[tvm_shape.size() - i - 1]);
    }

    if (tvm_dtype.code == DLDataTypeCode::kDLFloat && tvm_dtype.bits == 32) {
      vsi_dtype = vsi::DataType::FLOAT32;
    } else if (tvm_dtype.code == DLDataTypeCode::kDLUInt && tvm_dtype.bits == 8) {
      vsi_dtype = vsi::DataType::UINT8;
    } else {
      vsi_dtype = vsi::DataType::FLOAT32;
    }

    // If scale and offset provided create quantized tensor.
    if (scale != nullptr && offset != nullptr) {
	    //qnn tensor
    }

    vsi::TensorSpec input_spec(vsi_dtype, vsi_shape, vsi_attr);
    std::shared_ptr<vsi::Tensor> tensor;
    if (data != nullptr)
      tensor = graph_->CreateTensor(input_spec, data);
    else
      tensor = graph_->CreateTensor(input_spec);
    return tensor;
  }

  std::shared_ptr<vsi::Context> context_;
  std::shared_ptr<vsi::Graph> graph_;
  /* The entry ID to its corresponding output memory. */
  std::unordered_map<uint32_t, std::shared_ptr<vsi::Tensor>> entry_out_tensor_;
  std::vector<std::shared_ptr<vsi::Tensor>> dummy_tensor_;
};

#else

class VsiNpuJSONRuntime : public JSONRuntimeBase {

 public:
  VsiNpuJSONRuntime(const std::string& symbol_name, const std::string& graph_json,
                  const Array<String> const_names)
      : JSONRuntimeBase(symbol_name, graph_json, const_names) {}

  const char* type_key() const { return "vsi_npu_json"; }

  void Init(const Array<NDArray>& consts) override {

  }

  void Run() override {
  }
};
#endif



runtime::Module VsiNpuJSONRuntimeCreate(String symbol_name, String graph_json,
                                      const Array<String>& const_names) {
  auto n = make_object<VsiNpuJSONRuntime>(symbol_name, graph_json, const_names);
  return runtime::Module(n);
}

TVM_REGISTER_GLOBAL("runtime.VsiNpuJSONRuntimeCreate").set_body_typed(VsiNpuJSONRuntimeCreate);

#ifdef USE_VSI_NPU_RUNTIME
TVM_REGISTER_GLOBAL("runtime.module.loadbinary_vsi_npu_json")
    .set_body_typed(JSONRuntimeBase::LoadFromBinary<VsiNpuJSONRuntime>);
#endif
}  // namespace contrib
}  // namespace runtime
}  // namespace tvm