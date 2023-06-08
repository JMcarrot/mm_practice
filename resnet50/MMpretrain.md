# MMpretrain

## 题目：基于 ResNet50 的水果分类

## 背景：使用基于卷积的深度神经网络 ResNet50 对 30 种水果进行分类

## 任务：

1. 划分训练集和验证集

2. 按照 MMPreTrain CustomDataset 格式组织训练集和验证集

3. 使用 MMPreTrain 算法库，编写配置文件，正确加载预训练模型

4. 在水果数据集上进行微调训练

5. 使用 MMPreTrain 的 ImageClassificationInferencer 接口，对网络水果图像，或自己拍摄的水果图像，使用训练好的模型进行 分类

6. 需提交的验证集评估指标（不能低于 60%）
   **ResNet-50**

   ```
   2023/06/08 21:40:46 - mmengine - INFO - Exp name: resnet50_finetune_20230608_210701
   2023/06/08 21:40:46 - mmengine - INFO - Epoch(val)  [86][10/28]    eta: 0:00:01  time: 0.0796  data_time: 0.0178  memory: 2965  
   2023/06/08 21:40:47 - mmengine - INFO - Epoch(val)  [86][20/28]    eta: 0:00:00  time: 0.0618  data_time: 0.0005  memory: 612  
   2023/06/08 21:40:47 - mmengine - INFO - Epoch(val) [86][28/28]  accuracy/top1: 70.7207  data_time: 0.0004  time: 0.0601
   ```

   

## 数据集下载：

链接:(https://pan.baidu.com/share/init?surl=YgoU1M_v7ridtXB9xxbA1Q)
t提取码:52m9