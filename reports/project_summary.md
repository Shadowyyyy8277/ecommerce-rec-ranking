# 作品集项目说明：基于深度学习的电商搜索推荐排序系统

## 项目定位

本项目面向“算法工程师-机器学习”实习岗位，模拟电商搜索/推荐链路中的核心工程闭环：数据生成、特征工程、多路召回、CTR 排序、DeepFM 深度模型、离线评估、实验管理与在线推理服务。

## 业务问题

电商平台需要在用户进入搜索或推荐场景时，从海量商品中快速找出可能感兴趣的候选商品，并通过排序模型预测点击/转化倾向，最终返回 Top-K 商品列表。本项目将该链路拆成召回层与排序层：召回层重在覆盖，排序层重在精排质量。

## 技术亮点

- 使用时间切分构造 train/valid/test，并对 valid/test 只使用 train 统计特征。
- 训练集统计特征采用 cumulative history，当前样本不读取自身 label。
- 实现 popular、ItemCF、Item2Vec 三路召回，覆盖规则、协同过滤和表示学习思路。
- 对比 LR、LightGBM、DeepFM 三种排序模型，并使用 AUC、LogLoss 和 Top-K 指标评估。
- 使用 MLflow 记录参数与指标，但训练失败保护和本地 artifacts 保存不依赖 MLflow 成功。
- FastAPI 和 Streamlit 均支持模型缺失时 fallback，方便演示和调试。

## 简历表达建议

- 基于电商用户行为数据构建多路召回与 CTR 排序系统，完成数据清洗、用户/商品/上下文特征工程、ItemCF/Item2Vec 召回、LightGBM/DeepFM 排序模型训练与离线评估。
- 设计 AUC、LogLoss、Recall@K、NDCG@K、HitRate@K 等指标，对比 LR、LightGBM、DeepFM 等模型效果，并使用 MLflow 管理实验参数与指标。
- 使用 FastAPI 封装在线推理接口，支持输入用户 ID 返回 Top-K 商品推荐结果，并通过 Streamlit 搭建可交互 demo 展示推荐结果和模型效果。

## 面试可讲点

1. 为什么 behavior_type 不能作为 CTR 特征输入。
2. 为什么 valid/test 的统计特征必须只从 train 中计算。
3. ItemCF 与 Item2Vec 的区别：共现统计 vs. 表示学习。
4. DeepFM 如何同时建模一阶特征、二阶交叉和高阶非线性特征。
5. 为什么工程侧要设计模型缺失 fallback 和 MLflow 失败保护。
