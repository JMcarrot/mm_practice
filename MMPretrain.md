# MMPretrain

## 背景：使用基于卷积的深度神经网络 ResNet50 对 30 种水果进行分类

## 任务：

1. 划分训练集和验证集

2. 按照 MMPreTrain CustomDataset 格式组织训练集和验证集

3. 使用 MMPreTrain 算法库，编写配置文件，正确加载预训练模型

4. 在水果数据集上进行微调训练

5. 使用 MMPreTrain 的 ImageClassificationInferencer 接口，对网络水果图像，或自己拍摄的水果图像，使用训练好的模型进行分类

6. 需提交的验证集评估指标（不能低于 60%）

   resnet 50[70.7207]

```
2023/06/08 21:40:46 - mmengine - INFO - Exp name: resnet50_finetune_20230608_210701
2023/06/08 21:40:46 - mmengine - INFO - Epoch(val)  [86][10/28]    eta: 0:00:01  time: 0.0796  data_time: 0.0178  memory: 2965  
2023/06/08 21:40:47 - mmengine - INFO - Epoch(val)  [86][20/28]    eta: 0:00:00  time: 0.0618  data_time: 0.0005  memory: 612  
2023/06/08 21:40:47 - mmengine - INFO - Epoch(val) [86][28/28]  accuracy/top1: 70.7207  data_time: 0.0004  time: 0.0601

```

resnet18[65.8784]

```
2023/06/08 18:33:56 - mmengine - INFO - Epoch(train)  [98][100/109]  lr: 1.0000e-04  eta: 0:00:14  time: 0.0624  data_time: 0.0007  memory: 878  loss: 0.8838
2023/06/08 18:33:57 - mmengine - INFO - Exp name: resnet18_finetune_20230608_181932
2023/06/08 18:33:57 - mmengine - INFO - Epoch(val) [98][28/28]  accuracy/top1: 65.8784  data_time: 0.0028  time: 0.0232
```



## 数据集下载：

链接:(https://pan.baidu.com/share/init?surl=YgoU1M_v7ridtXB9xxbA1Q)
t提取码:52m9