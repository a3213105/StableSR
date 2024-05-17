from array import array
from locale import ABDAY_1
import numpy as np
from datetime import datetime
from openvino.runtime import Core, Model, get_version, AsyncInferQueue, InferRequest, Layout, Type, Tensor, Dimension, properties
from openvino.preprocess import PrePostProcessor
import copy
import torch
import time


def print_inputs_and_outputs_info(model: Model):
    inputs = model.inputs
    print("Model inputs:")
    for input in inputs:
        in_name = " , ".join(input.get_names())
        node_name = input.node.get_friendly_name()

        if in_name=="": in_name = "***NO_NAME***"
        if node_name=="": node_name = "***NO_NAME***"

        print(f"    {in_name} (node: {node_name}) : {input.element_type.get_type_name()} / "
                    f"{str(input.node.layout)} / {input.partial_shape}")

    outputs = model.outputs
    print("Model outputs:")
    for output in outputs:
        out_name = " , ".join(output.get_names())
        node_name = output.get_node().input(0).get_source_output().get_node().get_friendly_name()

        if out_name=="": out_name = "***NO_NAME***"
        if node_name=="": node_name = "***NO_NAME***"

        print(f"    {out_name} (node: {node_name}) : {output.element_type.get_type_name()} / "
                    f"{str(output.node.layout)} / {output.partial_shape}")


def print_runtime_params(compiled_model) :
    keys = compiled_model.get_property(properties.supported_properties())
    print("Model:")
    for k in keys:
        skip_keys = (properties.supported_properties())
        if k not in skip_keys:
            value = compiled_model.get_property(k)
            if k == properties.device.properties():
                for device_key in value.keys():
                    print(f'  {device_key}:')
                    for k2, value2 in value.get(device_key).items():
                        if k2 not in skip_keys:
                            print(f'    {k2}: {value2}')
            else:
                print(f'  {k}: {value}')

class OV_Operator(object):
    core = None
    model = None
    input_names = None
    input_shapes = None
    out_name = None
    exec_net = None
    infer_queue = None
    outputs= None

    def __init__(self, model, core=None, postprocess=None):
        self.postprocess = postprocess
        if core is None :
            self.core = Core()
        else :
            self.core = core
        self.model = self.core.read_model(model=model)
        output_size = self.model.get_output_size()
        self.outputs = []
        for i in range (0,output_size):
            self.outputs.append(i)
        # print('output: {}'.format(len(self.outputs)))
        self.input_names = []
        self.input_shapes = {}
        ops = self.model.get_ordered_ops()
        for it in ops:
            if it.get_type_name() == 'Parameter':
                self.input_names.insert(0, it.get_friendly_name())
                self.input_shapes[it.get_friendly_name()] = it.partial_shape
                # print('input {}: {}'.format(it.get_friendly_name(),it.partial_shape))
        self.input_name = self.input_names[0]

    def reshape_model(self, shapes) :
        new_shapes = {}
        for i in range(len(shapes)) :
            new_shapes[self.input_names[i]] = shapes[i]
        # print('Reshaping model: {}'.format(', '.join("'{}': {}".format(k, str(v)) for k, v in new_shapes.items())))
        tic = time.time()
        self.model.reshape(new_shapes)
        tac = time.time()
        print('##### Reshaping model ({}) : {}, using {}'.format(self.model.get_friendly_name(), 
                    ', '.join("'{}': {}".format(k, str(v)) for k, v in new_shapes.items()), (tac-tic)))
        for k, v in new_shapes.items():
            self.input_shapes[k] = v
        self.exec_net = self.core.compile_model(self.model, 'CPU', self.config)

    def setup_model(self, stream_num, bf16, shapes) :
        self.config = self.prepare_for_cpu(stream_num, bf16)
        if shapes is not None:
            self.reshape_model(shapes)
        else :
            self.exec_net = self.core.compile_model(self.model, 'CPU', self.config)
        print_inputs_and_outputs_info(self.model)
        print_runtime_params(self.exec_net)
        self.num_requests = self.exec_net.get_property("OPTIMAL_NUMBER_OF_INFER_REQUESTS")
        if self.num_requests == 1:
            # self.request = InferRequest(self.exec_net)
            self.infer_queue = None
        else :
            # self.request = None 
            self.infer_queue = AsyncInferQueue(self.exec_net, self.num_requests)
            self.num_requests = len(self.infer_queue)
        print('Model ({})  using {} streams'.format(self.model.get_friendly_name(), self.num_requests))
    
    def prepare_for_cpu(self, stream_num, bf16=True) :
        device = "CPU"
        hint = 'THROUGHPUT' if stream_num>1 else 'LATENCY'
        data_type = 'bf16' if bf16 else 'f32'
        config = {}
        config["NUM_STREAMS"] = str(stream_num)
        config['PERF_COUNT'] = False
        config['INFERENCE_PRECISION_HINT'] = data_type #'bf16'#'f32'
        return config

class OV_Result :
    results = None
    outputs = None
    def __init__(self, outputs) :
        self.outputs = outputs
        self.results = {}

    def completion_callback(self, infer_request: InferRequest, index: any) :
        #if index not in self.results :
        self.results[index] = []
        for i in self.outputs:
            self.results[index].append(copy.deepcopy(infer_request.get_output_tensor(i).data))
        return 

    def sync_parser(self, result, index: any) :
        # self.results = {}
        self.results[index] = []
        values = result.values()
        for i, value in enumerate(values):
            self.results[index].append(value)
        return 
    
    def sync_clean(self):
        self.results = {}

class SRResult(OV_Result):
    def __init__(self, outputs) :
        super().__init__(outputs)

class SRProcessor(OV_Operator):
    def __init__(self, model, core=None, postprocess=None):
        super().__init__(model, core, postprocess)

    def setup_model(self, stream_num, bf16, shapes) :
        super().setup_model(stream_num, bf16, shapes)
        self.res = SRResult(self.outputs)
        if self.infer_queue :
            self.infer_queue.set_callback(self.res.completion_callback)

    def run(self, inputs, input_tensors):
        return self.__call__(input_tensors)

    def __call__(self, input_list, cond_list, t_in, context) :
        nsize = input_list.size(0)
        for i in range(input_list.size(0)) :
            self.infer_queue.start_async({0: cond_list[i].unsqueeze(0),
                                          1: t_in[0].unsqueeze(0),
                                          2: input_list[i].unsqueeze(0),
                                          3: context[0].unsqueeze(0)}, 
                                         userdata=i)
        self.infer_queue.wait_all()
        
        res = []
        if self.postprocess is None:
            for i in range(nsize) :
                res.append( torch.tensor(self.res.results[i][0]) )
        else :
            for i in range(nsize) :
                res.append(self.postprocess(self.res.results[i][0]))
        return res

class FirstStageResult(OV_Result):
    def __init__(self, outputs) :
        super().__init__(outputs)

class FirstStageProcessor(OV_Operator):
    def __init__(self, model, core=None, postprocess=None):
        super().__init__(model, core, postprocess)

    def setup_model(self, stream_num, bf16, shapes) :
        super().setup_model(stream_num, bf16, shapes)
        self.res = SRResult(self.outputs)
        if self.infer_queue :
            self.infer_queue.set_callback(self.res.completion_callback)

    def run(self, inputs, input_tensors):
        return self.__call__(input_tensors)

    def __call__(self, input_tensor) :
        _, _, mw, mh = self.input_shapes[self.input_names[0]]
        if  input_tensor.shape[2]!=mw or input_tensor.shape[3]!=mh :
            self.reshape_model([input_tensor.shape])
        if self.infer_queue is None :
            return torch.tensor(self.exec_net.infer_new_request({0: input_tensor})[0])
    
        self.infer_queue.start_async({0: input_tensor}, userdata=0)
        self.infer_queue.wait_all()    
        res = []
        if self.postprocess is None:
            return torch.tensor(self.res.results[0][0])
        else :
            return self.postprocess(self.res.results[0][0])


class VQGanResult(OV_Result):
    def __init__(self, outputs) :
        super().__init__(outputs)

class VQGanProcessor(OV_Operator):
    def __init__(self, model, core=None, postprocess=None):
        super().__init__(model, core, postprocess)

    def setup_model(self, stream_num, bf16, shapes) :
        super().setup_model(stream_num, bf16, shapes)
        self.res = SRResult(self.outputs)
        if self.infer_queue :
            self.infer_queue.set_callback(self.res.completion_callback)

    def run(self, inputs, input_tensors):
        return self.__call__(input_tensors)

    def __call__(self, input_tensor, samples_tensor) :
        _, _, mw0, mh0 = self.input_shapes[self.input_names[0]]
        _, _, mw1, mh1 = self.input_shapes[self.input_names[1]]
        if input_tensor.shape[2]!=mw0 or input_tensor.shape[3]!=mh0 or samples_tensor.shape[2]!=mw1 or samples_tensor.shape[3]!=mh1 :
            print(f"##### input_tensor.shape={input_tensor.shape}, {self.input_shapes[self.input_names[0]]} ")
            print(f"##### samples_tensor.shape={samples_tensor.shape}, {self.input_shapes[self.input_names[1]]} ")
            self.reshape_model([input_tensor.shape, samples_tensor.shape])
        if self.infer_queue is None :
            return torch.tensor(self.exec_net.infer_new_request({0: input_tensor, 1: samples_tensor,})[0])
        
        self.infer_queue.start_async({0: input_tensor, 1: samples_tensor,}, userdata=0)
        self.infer_queue.wait_all()
       
        if self.postprocess is None:
            return torch.tensor(self.res.results[0][0])
        else :
            return self.postprocess(self.res.results[0][0])