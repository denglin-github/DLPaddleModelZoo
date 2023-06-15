import numpy as np

import paddle
from paddle.fluid.core import AnalysisConfig
from paddle.fluid.core import create_paddle_predictor, create_predictor


class Predictor(object):
    # init and configure a dlnne engine
    def __init__(self, pdmodel, pdiparams, disable_graphs, disable_nodes):
        self.pdmodel = pdmodel
        self.pdiparams = pdiparams
        self.disable_graphs = disable_graphs
        self.disable_nodes = disable_nodes
        self.predictor = self.create_predictor(self.pdmodel, 
                                               self.pdiparams,
                                               self.disable_graphs,
                                               self.disable_nodes)
        self.input_nums = self.get_input_nums(self.predictor)
        
    @classmethod
    def create_predictor(self, pdmodel, pdiparams, disable_graphs, disable_nodes):
        config = AnalysisConfig(pdmodel, pdiparams)
        # open runtime profile
        config.enable_profile()
        # open dlnne accelebrate interface
        config.enable_dlnne(min_subgraph_size=1,
                             use_static_batch=False,  
                             # dlnne engine can not support all paddle ops right now 
                             disable_graphs_by_nodes_outputs=set(disable_graphs),
                             disable_nodes_by_outputs=set(disable_nodes)
                            )
        config.switch_use_feed_fetch_ops(False)
        return create_predictor(config)
    
    @classmethod
    def get_input_nums(self, predictor):
        return len(predictor.get_input_names())
    
    # call inference on denglin GPU
    def run(self, data):
        input_names = self.predictor.get_input_names()
        for i, name in enumerate(input_names):
            input_tensor = self.predictor.get_input_handle(name)
            input_tensor.reshape(data[i].shape)
            input_tensor.copy_from_cpu(np.array(data[i]).copy())
        self.predictor.run()
        results = []
        output_names = self.predictor.get_output_names()
        for i, name in enumerate(output_names):
            output_tensor = self.predictor.get_output_handle(name)
            output_data = output_tensor.copy_to_cpu()
            results.append(output_data)
        return results