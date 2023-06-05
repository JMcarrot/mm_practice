# 题目：基于RTMPose的耳朵穴位关键点检测

**背景**：根据中医的“倒置胎儿”学说，耳朵的穴位反映了人体全身脏器的健康，耳穴按摩可以缓解失眠多梦、内分泌失调等疾病。耳朵面积较小，但穴位密集，涉及耳舟、耳轮、三角窝、耳甲艇、对耳轮等三维轮廓，普通人难以精准定位耳朵穴位。

**任务**
1.Labelme标注关键点检测数据集（子豪兄已经帮你完成了）
2.划分训练集和测试集（子豪兄已经帮你完成了）
3.Labelme标注转MS COCO格式（子豪兄已经帮你完成了）
4.使用MMDetection算法库，训练RTMDet耳朵目标检测算法，提交测试集评估指标
5.使用MMPose算法库，训练RTMPose耳朵关键点检测算法，提交测试集评估指标
6.用自己耳朵的图像预测，将预测结果发到群里
7.用自己耳朵的视频预测，将预测结果发到群里
需提交的测试集评估指标（不能低于baseline指标的50%）

- 目标检测Baseline模型（RTMDet-tiny）

  ```
  2023/06/03 23:51:18 - mmengine - INFO - Evaluating bbox...
  2023/06/03 23:51:19 - mmengine - INFO - bbox_mAP_copypaste: 0.741 0.968 0.968 -1.000 -1.000 0.741
  2023/06/03 23:51:19 - mmengine - INFO - Epoch(val) [200][11/11]  coco/bbox_mAP: 0.7410  coco/bbox_mAP_50: 0.9680  coco/bbox_mAP_75: 0.9680  coco/bbox_mAP_s: -1.0000  coco/bbox_mAP_m: -1.0000  coco/bbox_mAP_l: 0.7410  data_time: 0.5592  time: 0.6041
  ```

  

- 关键点检测Baseline模型（RTMPose-s）

  ```
  2023/06/04 16:36:22 - mmengine - INFO - Evaluating CocoMetric...
  2023/06/04 16:36:22 - mmengine - INFO - Evaluating PCKAccuracy (normalized by ``"bbox_size"``)...
  2023/06/04 16:36:22 - mmengine - INFO - Evaluating AUC...
  2023/06/04 16:36:22 - mmengine - INFO - Evaluating NME...
  2023/06/04 16:36:22 - mmengine - INFO - Epoch(val) [300][6/6]  coco/AP: 0.740148  coco/AP .5: 1.000000  coco/AP .75: 0.970297  coco/AP (M): -1.000000  coco/AP (L): 0.740148  coco/AR: 0.783333  coco/AR .5: 1.000000  coco/AR .75: 0.976190  coco/AR (M): -1.000000  coco/AR (L): 0.783333  PCK: 0.970522  AUC: 0.126644  NME: 0.040235  data_time: 1.942605  time: 1.978077
  ```

  

