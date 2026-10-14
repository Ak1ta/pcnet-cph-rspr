# 来源说明

本项目基于 PCNet 公开实现进行结构整理和模块扩展，新增部分为 CPH 挑战属性辅助预测头与 RSPR 区域结构原型校正，以及统一实验配置、记录与评估工程。

| 来源 | 项目地址 | 用途 |
| --- | --- | --- |
| PCNet | https://github.com/Angknpng/PCNet | 未对齐 RGB-T 共同框架与 UVT20K 数据来源 |
| Swin Transformer | https://github.com/microsoft/Swin-Transformer | 双模态编码骨干与预训练组件 |
| IHN | https://github.com/imdumpl78/IHN | 迭代单应估计组件 |
| Asymmetric Loss | https://github.com/Alibaba-MIIL/ASL | 非对称多标签损失的思想来源 |
| PySODMetrics | https://github.com/lartpang/PySODMetrics | 显著性评估工具 |

数据集、预训练组件和第三方实现的使用条件以各原项目说明为准。本文件是来源记录，不替代第三方许可证，也不为第三方内容授予新的使用权。
