# PaddlePaddle Denglin Model Zoo
### 🤝 登临科技AI + 飞桨
* 兼容性适配：目前登临科技与百度飞桨深度学习框架已完成三级兼容性适配认证，支持当下主流模型应用场景，覆盖了计算机视觉、智能语音、自然语言处理、推荐、图神经网络和强化学习等领域，支持当下主流模型数量100+;
* 一键启动：通过兼容飞桨推理接口，用户通过指定enable_dlnne()接口一键启动模型，并部署在登临GPU上执行;
* 性能评估：开启enable_profile()接口即可评估模型性能;
* 支持拓展：用户可自行准备飞桨预训练inference模型，通过登临GPU实现加速推理;
* 其他特性：有关enable_dlnne()接口的是详细使用方法可参考[Paddle-dlNNE](Paddle-dlNNE.md);
        
### 📦 模型信息
#### PaddleClas
| Models	                    | Evaluate Datasets|Input shape	| Acc/Metrics(paddle)|Acc/Metrics(Denglin GPU)|	Latency(ms)(Denglin GPU,BS=1) | Inference Model 
|-------------------------------|-------------------|-----------|------------------|------------------------|--------------------------|--------------|
|AlexNet	                    |ImageNet1k	     |1x3x224x224   |0.567	                |0.56644	            |8.388 | [inference_model](https://pan.baidu.com/s/1zu5Ymwq1iWYgEh9tuBeoog?pwd=8umb) 
|DenseNet121	                |ImageNet1k	     |1x3x224x224   |0.7566	                |0.75666	            |9.889     | [inference_model](https://pan.baidu.com/s/1zu5Ymwq1iWYgEh9tuBeoog?pwd=8umb) 
|DLA102        	                |ImageNet1k	     |1x3x224x224   |0.7893	                |0.78926	            |14.285 | [inference_model](https://pan.baidu.com/s/1zu5Ymwq1iWYgEh9tuBeoog?pwd=8umb) 
|DLA34	                        |ImageNet1k	     |1x3x224x224   |0.7603	                |0.76028	            |5.676   | [inference_model](https://pan.baidu.com/s/1zu5Ymwq1iWYgEh9tuBeoog?pwd=8umb) 
|DLA46_c	                    |ImageNet1k	     |1x3x224x224   |0.6321	                |0.63210	            |2.488 | [inference_model](https://pan.baidu.com/s/1zu5Ymwq1iWYgEh9tuBeoog?pwd=8umb) 
|DLA60	                        |ImageNet1k	     |1x3x224x224   |0.7610	                |0.76102	            |8.660   | [inference_model](https://pan.baidu.com/s/1zu5Ymwq1iWYgEh9tuBeoog?pwd=8umb) 
|DLA60x_c        	            |ImageNet1k	     |1x3x224x224   |0.6645	                |0.66448	            |2.907     | [inference_model](https://pan.baidu.com/s/1zu5Ymwq1iWYgEh9tuBeoog?pwd=8umb) 
|DPN68	                        |ImageNet1k	     |1x3x224x224   |0.7678	                |0.76780	            |7.785     | [inference_model](https://pan.baidu.com/s/1zu5Ymwq1iWYgEh9tuBeoog?pwd=8umb) 
|DPN92	                        |ImageNet1k	     |1x3x224x224   |0.7985	                |0.79852	            |23.162    | [inference_model](https://pan.baidu.com/s/1zu5Ymwq1iWYgEh9tuBeoog?pwd=8umb)  
|ESNet_x0_25	                |ImageNet1k	     |1x3x224x224   |0.6248	                |0.62462	            |9.464     | [inference_model](https://pan.baidu.com/s/1zu5Ymwq1iWYgEh9tuBeoog?pwd=8umb)   
|ESNet_x0_5	                    |ImageNet1k	     |1x3x224x224   |0.6882	                |0.68820	            |9.857     | [inference_model](https://pan.baidu.com/s/1zu5Ymwq1iWYgEh9tuBeoog?pwd=8umb) 
|GoogleNet	                    |ImageNet1k	     |1x3x224x224   |0.7070	                |0.70690	            |4.678   | [inference_model](https://pan.baidu.com/s/1zu5Ymwq1iWYgEh9tuBeoog?pwd=8umb) 
|HarDNet39_ds	                |ImageNet1k	     |1x3x224x224   |0.7133	                |0.71332	            |2.968     | [inference_model](https://pan.baidu.com/s/1zu5Ymwq1iWYgEh9tuBeoog?pwd=8umb)   
|HarDNet68_ds	                |ImageNet1k	     |1x3x224x224   |0.7362	                |0.73618	            |4.298     | [inference_model](https://pan.baidu.com/s/1zu5Ymwq1iWYgEh9tuBeoog?pwd=8umb)   
|HRNet_W18_C	                |ImageNet1k	     |1x3x224x224   |0.7692	                |0.76890	            |18.502    | [inference_model](https://pan.baidu.com/s/1zu5Ymwq1iWYgEh9tuBeoog?pwd=8umb)   
|MixNet_S	                    |ImageNet1k	     |1x3x224x224   |0.7628	                |0.76282	            |10.055    | [inference_model](https://pan.baidu.com/s/1zu5Ymwq1iWYgEh9tuBeoog?pwd=8umb)   
|MobileNetV1	                |ImageNet1k	     |1x3x224x224   |0.7099	                |0.71000	            |3.750     | [inference_model](https://pan.baidu.com/s/1zu5Ymwq1iWYgEh9tuBeoog?pwd=8umb)   
|MobileNetV2	                |ImageNet1k	     |1x3x224x224   |0.7215	                |0.72156	            |3.251     | [inference_model](https://pan.baidu.com/s/1zu5Ymwq1iWYgEh9tuBeoog?pwd=8umb)   
|MobileNetV3_small_x0_35_ssld   |ImageNet1k	     |1x3x224x224   |0.5555	                |0.55606	            |2.271       | [inference_model](https://pan.baidu.com/s/1zu5Ymwq1iWYgEh9tuBeoog?pwd=8umb) 
|MobileNetV3_small_x0_5	        |ImageNet1k	     |1x3x224x224   |0.5921	                |0.59192	            |2.484       | [inference_model](https://pan.baidu.com/s/1zu5Ymwq1iWYgEh9tuBeoog?pwd=8umb) 
|MobileNetV3_small_x0_75	    |ImageNet1k	     |1x3x224x224   |0.6602	                |0.66050	            |3.150       | [inference_model](https://pan.baidu.com/s/1zu5Ymwq1iWYgEh9tuBeoog?pwd=8umb) 
|MobileNetV3_small_x1_25	    |ImageNet1k	     |1x3x224x224   |0.7067	                |0.70654	            |4.297       | [inference_model](https://pan.baidu.com/s/1zu5Ymwq1iWYgEh9tuBeoog?pwd=8umb) 
|PPLCNet_x0_25	                |ImageNet1k	     |1x3x224x224   |0.5186	                |0.51812	            |2.287       | [inference_model](https://pan.baidu.com/s/1zu5Ymwq1iWYgEh9tuBeoog?pwd=8umb) 
|PPLCNet_x0_35	                |ImageNet1k	     |1x3x224x224   |0.5809	                |0.58088	            |2.747       | [inference_model](https://pan.baidu.com/s/1zu5Ymwq1iWYgEh9tuBeoog?pwd=8umb) 
|PPLCNet_x0_5	                |ImageNet1k	     |1x3x224x224   |0.6314	                |0.63172	            |2.921       | [inference_model](https://pan.baidu.com/s/1zu5Ymwq1iWYgEh9tuBeoog?pwd=8umb) 
|PPLCNet_x1_0	                |ImageNet1k	     |1x3x224x224   |0.7132	                |0.71312	            |4.591       | [inference_model](https://pan.baidu.com/s/1zu5Ymwq1iWYgEh9tuBeoog?pwd=8umb) 
|RedNet26	                    |ImageNet1k	     |1x3x224x224   |0.7595	                |0.75950	            |219.600   | [inference_model](https://pan.baidu.com/s/1zu5Ymwq1iWYgEh9tuBeoog?pwd=8umb)     
|RedNet38	                    |ImageNet1k	     |1x3x224x224   |0.7747	                |0.77470	            |333.226   | [inference_model](https://pan.baidu.com/s/1zu5Ymwq1iWYgEh9tuBeoog?pwd=8umb)     
|Res2Net50_14w_8s	            |ImageNet1k	     |1x3x224x224   |0.7946	                |0.79462	            |11.200    | [inference_model](https://pan.baidu.com/s/1zu5Ymwq1iWYgEh9tuBeoog?pwd=8umb) 
|ResNet101_vd	                |ImageNet1k	     |1x3x224x224   |0.8017	                |0.80178	            |12.615      | [inference_model](https://pan.baidu.com/s/1zu5Ymwq1iWYgEh9tuBeoog?pwd=8umb) 
|ResNet18	                    |ImageNet1k	     |1x3x224x224   |0.7098	                |0.70988	            |3.411       | [inference_model](https://pan.baidu.com/s/1zu5Ymwq1iWYgEh9tuBeoog?pwd=8umb) 
|ResNet50	                    |ImageNet1k	     |1x3x224x224   |0.7650	                |0.76502	            |7.597       | [inference_model](https://pan.baidu.com/s/1zu5Ymwq1iWYgEh9tuBeoog?pwd=8umb) 
|ResNeXt50_32x4d	            |ImageNet1k	     |1x3x224x224   |0.7775	                |0.77754	            |9.849       | [inference_model](https://pan.baidu.com/s/1zu5Ymwq1iWYgEh9tuBeoog?pwd=8umb) 
|ReXNet_1_0	                    |ImageNet1k	     |1x3x224x224   |0.7746	                |0.77452	            |23.933      | [inference_model](https://pan.baidu.com/s/1zu5Ymwq1iWYgEh9tuBeoog?pwd=8umb) 
|ReXNet_1_3	                    |ImageNet1k	     |1x3x224x224   |0.7913	                |0.79134	            |28.345      | [inference_model](https://pan.baidu.com/s/1zu5Ymwq1iWYgEh9tuBeoog?pwd=8umb) 
|ReXNet_1_5	                    |ImageNet1k	     |1x3x224x224   |0.8006	                |0.80072	            |30.896      | [inference_model](https://pan.baidu.com/s/1zu5Ymwq1iWYgEh9tuBeoog?pwd=8umb) 
|ReXNet_2_0	                    |ImageNet1k	     |1x3x224x224   |0.8122	                |0.81242	            |39.879      | [inference_model](https://pan.baidu.com/s/1zu5Ymwq1iWYgEh9tuBeoog?pwd=8umb) 
|ReXNet_3_0	                    |ImageNet1k	     |1x3x224x224   |0.8209	                |0.82086	            |60.846      | [inference_model](https://pan.baidu.com/s/1zu5Ymwq1iWYgEh9tuBeoog?pwd=8umb) 
|SE_ResNet18_Vd	                |ImageNet1k	     |1x3x224x224   |0.7333	                |0.73332	            |4.402       | [inference_model](https://pan.baidu.com/s/1zu5Ymwq1iWYgEh9tuBeoog?pwd=8umb) 
|SE_ResNet34_vd	                |ImageNet1k	     |1x3x224x224   |0.7651	                |0.76518	            |7.116       | [inference_model](https://pan.baidu.com/s/1zu5Ymwq1iWYgEh9tuBeoog?pwd=8umb) 
|SE_ResNet50_vd	                |ImageNet1k	     |1x3x224x224   |0.7952	                |0.79524	            |15.831      | [inference_model](https://pan.baidu.com/s/1zu5Ymwq1iWYgEh9tuBeoog?pwd=8umb) 
|ShuffleNetV2_x0_25	            |ImageNet1k	     |1x3x224x224   |0.4990	                |0.49904	            |12.256      | [inference_model](https://pan.baidu.com/s/1zu5Ymwq1iWYgEh9tuBeoog?pwd=8umb) 
|ShuffleNetV2_x1_5	            |ImageNet1k	     |1x3x224x224   |0.7163	                |0.71636	            |14.337      | [inference_model](https://pan.baidu.com/s/1zu5Ymwq1iWYgEh9tuBeoog?pwd=8umb) 
|SqueezeNet1_1	                |ImageNet1k	     |1x3x224x224   |0.601	                |0.60076	            |2.295       | [inference_model](https://pan.baidu.com/s/1zu5Ymwq1iWYgEh9tuBeoog?pwd=8umb) 
|VGG11	                        |ImageNet1k	     |1x3x224x224   |0.693	                |0.69294	            |21.421      | [inference_model](https://pan.baidu.com/s/1zu5Ymwq1iWYgEh9tuBeoog?pwd=8umb) 
|VGG13	                        |ImageNet1k	     |1x3x224x224   |0.700	                |0.69994	            |24.873      | [inference_model](https://pan.baidu.com/s/1zu5Ymwq1iWYgEh9tuBeoog?pwd=8umb) 
|VGG19	                        |ImageNet1k	     |1x3x224x224   |0.726	                |0.72556	            |31.127      | [inference_model](https://pan.baidu.com/s/1zu5Ymwq1iWYgEh9tuBeoog?pwd=8umb) 

#### PaddleOCR
| Models	                    | Evaluate Datasets|Input shape	| Hmean(paddle)|Hmean(Denglin GPU)|	Latency(ms)(Denglin GPU,BS=1) | Inference Model 
|-------------------------------|-------------------|------------|----------|------------------------|--------------------------|--------------|
det_mv3_db_v2.0	               |ICDAR2015	                            | 1x3x736x1280|0.7512	            |0.75092  	    |96.365     | [inference_model](https://pan.baidu.com/s/1_jlxennqEwkmJYP2ijJq4w?pwd=cyws)    
det_r50_vd_db_v2.0	           |ICDAR2015	                            | 1x3x736x1280|0.8238	            |0.82368	    |318.926    | [inference_model](https://pan.baidu.com/s/1_jlxennqEwkmJYP2ijJq4w?pwd=cyws)    
det_mv3_east_v2.0	           |ICDAR2015	                            | 1x3x704x1280|0.7865	            |0.78680	    |74.671     | [inference_model](https://pan.baidu.com/s/1_jlxennqEwkmJYP2ijJq4w?pwd=cyws)
det_r50_vd_east_v2.0	       |ICDAR2015	                            | 1x3x704x1280|0.8488	            |0.84903	    |408.758    | [inference_model](https://pan.baidu.com/s/1_jlxennqEwkmJYP2ijJq4w?pwd=cyws)    
det_r50_vd_sast_icdar15_v2.0   |ICDAR2015	                            | 1x3x896x1536|0.8742	            |0.87415	    |1772.236   | [inference_model](https://pan.baidu.com/s/1_jlxennqEwkmJYP2ijJq4w?pwd=cyws)    
det_mv3_pse_v2.0	           |ICDAR2015	                            | 1x3x736x1312|0.7589	            |0.75894	    |304.274    | [inference_model](https://pan.baidu.com/s/1_jlxennqEwkmJYP2ijJq4w?pwd=cyws)    
det_r50_vd_pse_v2.0	           |ICDAR2015	                            | 1x3x736x1312|0.8255	            |0.82538  	    |674.206    | [inference_model](https://pan.baidu.com/s/1_jlxennqEwkmJYP2ijJq4w?pwd=cyws)    
rec_svtr_tiny_none_ctc_en	   |IIIT, SVT, IC03, IC13, IC15, SVTP, CUTE	| 1x3x64x256  |0.9013(Avg_10,acc)	|0.90105 (acc)	|6.564      | [inference_model](https://pan.baidu.com/s/1_jlxennqEwkmJYP2ijJq4w?pwd=cyws)

#### PaddleDetection
| Models	                    | Evaluate Datasets	| Input shape	| mAP(paddle)|mAP(Denglin GPU)|	Latency(ms)(Denglin GPU,BS=1) | Inference Model 
|-------------------------------|---------------|-------|--------------------|------------------------|--------------------------|--------------|
picodet_lcnet_1_5x_416_coco	    |coco	| 1x3x416x416    |0.363	|0.363 	|133.012            | [inference_model](https://pan.baidu.com/s/1mfKuhIkstFCmg6d5T6uEeA?pwd=w4gb)
picodet_s_320_coco	            |coco	| 1x3x320x320    |0.271	|0.271 	|66.497         | [inference_model](https://pan.baidu.com/s/1mfKuhIkstFCmg6d5T6uEeA?pwd=w4gb)
ppyolo_mbv3_large_coco	        |coco	| 1x3x320x320    |0.232	|0.240 	|27.512             | [inference_model](https://pan.baidu.com/s/1mfKuhIkstFCmg6d5T6uEeA?pwd=w4gb)
ppyolo_r50vd_dcn_1x_coco	    |coco	| 1x3x608x608    |0.448	|0.447 	|444.563            | [inference_model](https://pan.baidu.com/s/1mfKuhIkstFCmg6d5T6uEeA?pwd=w4gb)    
ppyolo_tiny_650e_coco	        |coco	| 1x3x320x320    |0.206	|0.207  |	29.661          | [inference_model](https://pan.baidu.com/s/1mfKuhIkstFCmg6d5T6uEeA?pwd=w4gb)    
ppyoloe_crn_s_300e_coco	        |coco	| 1x3x640x640    |0.430	|0.430	|115.896            | [inference_model](https://pan.baidu.com/s/1mfKuhIkstFCmg6d5T6uEeA?pwd=w4gb)    
ppyolov2_r50vd_dcn_365e_        |coco	| 1x3x640x640    |coco	|0.491	|0.491 	630.109     | [inference_model](https://pan.baidu.com/s/1mfKuhIkstFCmg6d5T6uEeA?pwd=w4gb)    
ttfnet_darknet53_1x_coco	    |coco	| 1x3x512x512    |0.335	|0.336 	|413.021            | [inference_model](https://pan.baidu.com/s/1mfKuhIkstFCmg6d5T6uEeA?pwd=w4gb)
yolov3_darknet53_270e_coco	    |coco	| 1x3x608x608    |0.391	|0.391	|279.647            | [inference_model](https://pan.baidu.com/s/1mfKuhIkstFCmg6d5T6uEeA?pwd=w4gb)
yolov3_mobilenet_v1_270e_coco	|coco	| 1x3x608x608     |0.294	|0.294	|136.460        | [inference_model](https://pan.baidu.com/s/1mfKuhIkstFCmg6d5T6uEeA?pwd=w4gb)    
yolox_s_300e_coco	            |coco	| 1x3x640x640    |0.404 	|0.404 	|142.276        | [inference_model](https://pan.baidu.com/s/1mfKuhIkstFCmg6d5T6uEeA?pwd=w4gb)

#### PaddleSeg
| Models	                    | Evaluate Datasets	| Input shape	|mIoU(paddle)|mIoU(Denglin GPU)|	Latency(ms)(Denglin GPU,BS=1)  | Inference Model 
|-------------------------------|-------------------|--------|------------|------------------------|--------------------------|--------------|
BiSeNetV1	                   | Cityscapes	       |  1x3x1024x2048        | 0.7519|0.75191	|23283.000       | [inference_model](https://pan.baidu.com/s/1rt9Dg76FALr7GRkbO1jscw?pwd=4c4l)
BiSeNetv2	                   | Cityscapes	       |  1x3x1024x2048      | 0.7319|0.73169	|594.874         | [inference_model](https://pan.baidu.com/s/1rt9Dg76FALr7GRkbO1jscw?pwd=4c4l)
CCNet	                       | Cityscapes	       |  1x3x1025x2049        | 0.8095|0.80951	|6435.860        | [inference_model](https://pan.baidu.com/s/1rt9Dg76FALr7GRkbO1jscw?pwd=4c4l)
DDRNet_23(DDRNet）	           | Cityscapes	       |  1x3x1024x2048        | 0.7985|0.79847	|729.794         | [inference_model](https://pan.baidu.com/s/1rt9Dg76FALr7GRkbO1jscw?pwd=4c4l)
DeepLabv3p_resnet50_cityscapes|	Cityscapes	       |  1x3x1024x2048          | 0.8036|0.8036	|3712.280        | [inference_model](https://pan.baidu.com/s/1rt9Dg76FALr7GRkbO1jscw?pwd=4c4l)
ENet	                       | Cityscapes	       |  1x3x1024x2048        | 0.6742|0.67420	|801.838         | [inference_model](https://pan.baidu.com/s/1rt9Dg76FALr7GRkbO1jscw?pwd=4c4l)
FCN_HRNet_W18	               | 飞桨内部人像数据集 |   1x3x1024x2048          | 0.787 |0.78969 |1580.298      | [inference_model](https://pan.baidu.com/s/1rt9Dg76FALr7GRkbO1jscw?pwd=4c4l)
GloRe	                       | Cityscapes	       |  1x3x1024x2048          | 0.7826|0.78256	|31732.400       | [inference_model](https://pan.baidu.com/s/1rt9Dg76FALr7GRkbO1jscw?pwd=4c4l)
HRNetW48Contrast	           | Cityscapes	       |  1x3x1024x2048          | 0.8230|0.82398	|3544.080        | [inference_model](https://pan.baidu.com/s/1rt9Dg76FALr7GRkbO1jscw?pwd=4c4l)
OCRNet_HRNetW18         	   | Cityscapes	       |  1x3x1024x2048          | 0.8067|0.80702	|3801.400        | [inference_model](https://pan.baidu.com/s/1rt9Dg76FALr7GRkbO1jscw?pwd=4c4l)
PFPNNet	                       | Cityscapes	       |  1x3x1024x2048          | 0.7907|0.79072	|28974.200       | [inference_model](https://pan.baidu.com/s/1rt9Dg76FALr7GRkbO1jscw?pwd=4c4l)
STDC_STDC1	                   | Cityscapes	       |   1x3x1024x2048         | 0.7474|0.74739	|904.822         | [inference_model](https://pan.baidu.com/s/1rt9Dg76FALr7GRkbO1jscw?pwd=4c4l)
UPERNet	                        |ADE20K         	|  1x3x1024x2048          |0.7958	|0.79581|8477.040        | [inference_model](https://pan.baidu.com/s/1rt9Dg76FALr7GRkbO1jscw?pwd=4c4l)

#### PaddleNLP
| Models	                    | Evaluate Datasets| Sequence Length	| Acc(paddle)|Acc(Denglin GPU)|	Latency(ms)(Denglin GPU,BS=1)  | Inference Model 
|-------------------------------|-------------------|------------|--------|------------------------|--------------------------|--------------|
BERT-Base	|SST-2	      |      128       | 0.92660	|0.92661	|20.455          |  [inference_model](https://pan.baidu.com/s/1zfWh5nqJDPArTWYENjuuPQ?pwd=bb0z)
Bi-LSTM 	|ChnSentiCorp|	     599       |0.8983	|0.89833|	25.231               |  [inference_model](https://pan.baidu.com/s/1zfWh5nqJDPArTWYENjuuPQ?pwd=bb0z)
ConvBert	|SST-2	      |      128       | 0.9139 	|0.91399|	102.281          | [inference_model](https://pan.baidu.com/s/1zfWh5nqJDPArTWYENjuuPQ?pwd=bb0z)
ELECTRA 	|SST-2	      |      128       | 0.9185	|0.91857|	4.601            |   [inference_model](https://pan.baidu.com/s/1zfWh5nqJDPArTWYENjuuPQ?pwd=bb0z)   
LayoutLM	|FUNSD	      |      512       | F1: 0.7913|	F1: 0.79116	|172.491     |   [inference_model](https://pan.baidu.com/s/1zfWh5nqJDPArTWYENjuuPQ?pwd=bb0z)   
MiniLMv2	|AFQMC       |	     128       |0.7138|	0.71362	|9.536               |  [inference_model](https://pan.baidu.com/s/1zfWh5nqJDPArTWYENjuuPQ?pwd=bb0z)
seq2seq	    |IWSLT15 en-vi|	     128       |BLEU: 0.2433|	BLEU: 0.24340	|782.965 |   [inference_model](https://pan.baidu.com/s/1zfWh5nqJDPArTWYENjuuPQ?pwd=bb0z)       
TextCNN	    |ChnSentiCorp|	     599       |0.9107	|0.91000	|1.273               |   [inference_model](https://pan.baidu.com/s/1zfWh5nqJDPArTWYENjuuPQ?pwd=bb0z)           
TinyBert	|SST-2	      |      128       | 0.9300	|0.93005	|20.583          |   [inference_model](https://pan.baidu.com/s/1zfWh5nqJDPArTWYENjuuPQ?pwd=bb0z)   

#### PaddleRec
| Models	                    | Evaluate Datasets	| Metrics(paddle)|Metrics(Denglin GPU)|	Latency(ms)(Denglin GPU,BS=1)  | Inference Model 
|-------------------------------|-------------------|--------------------|------------------------|--------------------------|--------------|
DSSM	      |  BQ	|0.93(正序率)	|0.92875(正序率)|	2.805       | [inference_model]()                                                                         
match-pyramid|	Letor07	|0.39(map)	|0.39296map)|	0.895                                                  | [inference_model](https://pan.baidu.com/s/1eqEFqGzAHu6UEgKv9mGmEQ?pwd=w1sa)                
NCF         	|movielens|	0.58(HR@10) 、0.33(NDCG@10)|	0.58543(HR@10) 、 0.33538(NDCG@10)	|0.699     | [inference_model](https://pan.baidu.com/s/1eqEFqGzAHu6UEgKv9mGmEQ?pwd=w1sa)                    
DLRM 	       | criteo|	Auc:0.79 +	|0.80120	|6.016                                                 | [inference_model](https://pan.baidu.com/s/1eqEFqGzAHu6UEgKv9mGmEQ?pwd=w1sa)                               
DeepFM 	       | Criteo	|Auc:0.78	|0.794357	|1.357                                                     | [inference_model](https://pan.baidu.com/s/1eqEFqGzAHu6UEgKv9mGmEQ?pwd=w1sa)            

#### PARL
| Models	    | Evaluate Datasets	|Metrics   | Reward(CPU)|Reward(Denglin GPU)|	Latency(ms)(Denglin GPU,BS=1)  | Inference Model 
|-------------------------------|-------------|------|--------------------|------------------------|--------------------------|--------------|
DQN_variant	   | Atari games	|Reward	|	3.66667	|3.66667	|7.171             | [inference_model](https://pan.baidu.com/s/1jT8MOohDMg8voZGkgXMniQ?pwd=9ukf)
PPO	Atari      | games	    |Reward	|-21.0	|-21.0	|1.587            | [inference_model](https://pan.baidu.com/s/1jT8MOohDMg8voZGkgXMniQ?pwd=9ukf)
DQN	           | CartPole-v0	|Reward	|19.0	|19.0	|0.350            | [inference_model](https://pan.baidu.com/s/1jT8MOohDMg8voZGkgXMniQ?pwd=9ukf)
MADDPG	       | gym     	|Reward	|	-75.19758	|-75.19758|	0.768            | [inference_model](https://pan.baidu.com/s/1jT8MOohDMg8voZGkgXMniQ?pwd=9ukf)

#### PGL
| Models	                    | Evaluate Datasets	| Acc(paddle)|Acc(Denglin GPU)|	Acc(CPU)|Latency(ms)(Denglin GPU,BS=1)  | Inference Model 
|-------------------------------|-------------------|----------------|--------------------|--------------|--------------------------|--------------|
gin	|MUTAG  |	--	|0.78947|	0.78947	|10.529          | [inference_model](https://pan.baidu.com/s/1a7-FnPsTGE7l_FlW18fwBg?pwd=hnww)
GraphSage 	|reddit |	--|	0.74706|	0.74706	|581.025          | [inference_model](https://pan.baidu.com/s/1a7-FnPsTGE7l_FlW18fwBg?pwd=hnww)
gat	|cora 	|0.83	|0.83333	|--	|103.456          | [inference_model](https://pan.baidu.com/s/1a7-FnPsTGE7l_FlW18fwBg?pwd=hnww)
gcn	|CORA	|0.81	|0.81000	|--	|109.458          | [inference_model](https://pan.baidu.com/s/1a7-FnPsTGE7l_FlW18fwBg?pwd=hnww)

#### PaddleSpeech
| Models	              | Evaluate Datasets	|Metrics      | Acc(paddle)|Acc(Denglin GPU)|	Acc(CPU)|Latency(ms)(Denglin GPU,BS=1)  | Inference Model 
|----------------|---------------|-------------------|----------------|--------------------|--------------|--------------------------|--------------|
hifigan	 |AISHELL-3	|mel_loss	|0.1068	|--	|0.10699	|104.712                 | [inference_model](https://pan.baidu.com/s/1HWaKnDbKyKQt7G1EaS_8ag?pwd=bfy7)
Tacotron2	|CSMSC	|eval/loss	|--	|1.928438	|1.928438	|--              | [inference_model](https://pan.baidu.com/s/1HWaKnDbKyKQt7G1EaS_8ag?pwd=bfy7)
Speedyspeech|	CSMSC	|eval/loss	|--	|0.879209	|0.879209|	146.011              | [inference_model](https://pan.baidu.com/s/1HWaKnDbKyKQt7G1EaS_8ag?pwd=bfy7)

### 🎈 推理预测
#### 以图像分类为例简要介绍模型使用方法，其他模型场景详细用法请参考飞桨官方模型库：

##### 1.模型准备： 通过链接下载登临飞桨ImageNet1K图像分类模型，例如 MobileNetV3.pdmodel 、 MobileNetV3.pdiparams
##### 2.数据准备： 输入图像应符合NCHW Format , Shape 为 [1,3,224,224]
##### 3.执行推理：
```bash
python3 tools/deploy/predict.py                             \
    --model_file    ${MODEL_PATH}/MobileNetV3.pdmodel       \
    --params_file   ${MODEL_PATH}/MobileNetV3.pdiparams     \
    --input_data    ${INPUT_DATA}
```
##### 4.获取最终推理结果，如图像类别、Bouding Box可视化、OCR检测结果等，可参考飞桨模型库相关代码