# 使用 Paddle-dlNNE 预测

登临科技 dlNNE 是一个高性能的 C++ 深度学习预测库。Paddle-dlNNE 是 PaddlePaddle 采用子图的形式集成了 dlNNE，为深度学习推理应用提供更低的延迟和更高的数据吞吐量，从而提升Paddle模型的预测性能。

## 概述

当模型加载后，神经网络可以表示为由权重变量和运算节点组成的计算图。Paddle-dlNNE 对整个图进行扫描，发现图中可以使用 dlNNE 优化的子图，并使用 dlNNE 节点替换它们。在模型的推理期间，如果遇到 dlNNE 节点，Paddle 会调用 dlNNE 库对该节点进行优化，其他的节点调用 Paddle 的原生实现。dlNNE 在推理期间能够进行OP的横向和纵向融合，过滤掉冗余的OP，并对登临硬件环境下的特定的OP进行优化，加快模型的预测速度。dlNNE 除了有常见的OP融合以及显存/内存优化外，还针对性的对OP进行了优化加速实现，降低预测延迟，提升推理的吞吐量。

## 环境准备

1. 请确保您的 Paddle 版本是2.1或更高的版本。Paddle-dlNNE 只能在 Paddle 2.1或更高的版本上运行。

2. 请手动编译源码生成 `whl` 包。手动编译的方法请参照 [编译文档](https://paddle-inference.readthedocs.io/en/latest/user_guides/source_compile.html)。

   **注意：**

   - 手动编译 Paddle 源码时需要在登临 Hamming<sup>TM</sup> SDK 环境下执行。Hamming<sup>TM</sup> SDK由登临官方提供。
   -  cmake 期间，设置  WITH_DLNNE 为 ON，设置 WIHT_GPU 为 OFF，设置 WITH_PYTHON 为 ON。

## API 使用介绍

在 [预测流程](https://paddleinference.paddlepaddle.org.cn/quick_start/workflow.html) 一节中，我们了解到 Paddle Inference 预测包含了以下几个方面：

- 配置推理选项
- 创建 predictor
- 准备模型输入
- 模型推理
- 获取模型输出

使用 Paddle-dlNNE 也是遵照这样的流程。我们先用一个简单的例子来介绍这一流程（我们假设您已经对 Paddle Inference 有一定的了解，如果您刚接触 Paddle Inference，请访问 [这里](https://paddleinference.paddlepaddle.org.cn/quick_start/workflow.html) 对 Paddle Inference 有个初步认识）。

```
import numpy as np
import paddle.inference as paddle_infer


def create_predictor():
    config = paddle_infer.Config("./resnet50/model", "./resnet50/params")
	
	disable_nodes_by_outputs = set()
	disable_nodes_by_outputs = "concat_3.tmp_0, concat_5.tmp".split(",")
	
    # 打开dlNNE。此接口的详细介绍请见下文
    config.enable_dlnne(max_batch_size = 1,
    					min_subgraph_size = 3,
                        precision_mode=paddle_infer.PrecisionType.Float32,
                        use_static_batch=True,
                        disable_node_by_outputs=disable_nodes_by_outputs)

    predictor = paddle_infer.create_predictor(config)
    return predictor

def run(predictor, img):
    # 准备输入
    input_names = predictor.get_input_names()
    for i,  name in enumerate(input_names):
        input_tensor = predictor.get_input_handle(name)
        input_tensor.reshape(img[i].shape)
        input_tensor.copy_from_cpu(img[i].copy())
    # 预测
    predictor.run()
    results = []
    # 获取输出
    output_names = predictor.get_output_names()
    for i, name in enumerate(output_names):
        output_tensor = predictor.get_output_handle(name)
        output_data = output_tensor.copy_to_cpu()
        results.append(output_data)
    return results

if __name__ == '__main__':
    pred = create_predictor()
    img = np.ones((1, 3, 224, 224)).astype(np.float32)
    result = run(pred, [img])
    print ("class index: ", np.argmax(result[0][0]))
```



通过例子可以看出，我们通过 `enable_dlnne` 接口来开启 dlNNE 选项。

    config.enable_dlnne(max_batch_size = 1,
    					min_subgraph_size = 3,
                        precision_mode=paddle_infer.PrecisionType.Float32,
                        use_static_batch=True,
                        input_shape_dict={"input":[1, 3, 512, 512]},
                        disable_node_by_outputs=disable_nodes_by_outputs)

`enable_dlnne` 接口各参数的介绍如下:

- **use_static_batch**

​		为True 时，当inference 的图最高维度为动态时，dLNNE会将该图的最高维度固定为 1。

- **max_batch_size**

  类型为 int。默认值为1。

  需要提前设置最大的 batch 的大小，运行时 batch 的大小不得超过此限定值。

- **min_subgraph_size**

  类型为 int。默认值为3。

  Paddle-dlNNE 以子图的形式运行。为避免性能损失，只有当子图内部节点个数大于 min_subgraph_size 的时候，才会调用 Paddle-dlNNE 运行。

- **precision_mode**

  类型为 **paddle_infer.PrecisionType**。 默认值为 **paddle_infer.PrecisionType.Float32**。

  指定使用 Paddle-dlNNE 的精度，目前支持 FP32 (float32)。int8 量化类型将在后续支持。

- **input_shape_dict**

  类型为 **dict**，其key为input的name，value 为其运行时的shape，目前Paddle-dlNNE只支持静态图，当网络的输入shape不固定时（比如[-1, 3, -1]）,需要指定一个固定的shape（比如[1, 3, 512]）。

- **disable_node_by_outputs**

  类型为python中的set，在该set 中的节点在划分子图时会优先被排除在dLNNE 子图以外，在cpu上执行。

- **disable_graphs_by_nodes_outputs**

  类型为python中的set，在该set 中的子图会优先被排除在dLNNE 子图以外，在cpu上执行。
​		