# Reconstruction

> 本项目旨在实现温度场重建任务

## 项目详情

* `main.py`程序运行入口

    ```python
    python main.py
    ```

* `src`

* `loss.py`温度场重建损失函数

    * `PointLoss`关键点损失函数：计算预测温度场监测点和实际监测值之间偏差

    * `LaplaceLoss`重建损失函数：计算预测温度场分布和额定功率对应温度场偏差

    * `OutsideLoss`边界损失函数：边界条件辅助温度场训练

* `models`模型文件

    * `fcn.py`、`fpn.py`、`segnet.py`、`unet.py`三种图到图回归模型

    * `backbone`骨干网络，包括`AlexNet`、`ResNet`、`VGG`网络

* `data_processing.py`数据处理文件

* `jupyter`文件夹用于放置jupyter调试文件