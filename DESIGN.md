# K-SPADA 设计说明

## 设计目标

- 结构尽量接近原始TOSICA代码，减少学习成本。
- 一个核心功能只对应一个主要代码文件。
- 模型、训练、预测、通信图和数据处理职责清楚分开。
- 保留教程所需的路径友好接口，但不再为每个小功能建立独立包装模块。

## 当前结构

```text
AnnData
   │
   ├── data_processing/ ──► createCGraph.py
   │                              │
   │                              ▼
   ├──────────────────────────► train.py ──► checkpoint
   │                                             │
   └──────────────────────────► pre.py ◄─────────┘
                                      │
                                      ▼
                              visualization.py
```

## 文件职责

- `KSPADA_model.py`：模型结构。
- `train.py`：图采样、损失、优化器、训练循环和训练入口。
- `pre.py`：查询图推理、预测结果、准确率统计和预测入口。
- `createCGraph.py`：通路打分、通信图、训练图和验证图合并。
- `customized_linear.py`：带基因—通路掩码的线性投影。
- `config.py`：可追溯的教程超参数。
- `visualization.py`：沿用交接训练脚本的双面板空间散点图。
- `data_processing/`：通用预处理、分箱、liver cancer和Fly3d数据处理。

## 精简原则

以下旧模块已经合并，不再保留在主包中：

- `model.py`合并到`KSPADA_model.py`；
- `training.py`和`api.py`的训练部分合并到`train.py`；
- `prediction.py`、`metrics.py`和`api.py`的预测部分合并到`pre.py`；
- `pathway_graph.py`和`cgraph.py`合并到`createCGraph.py`；
- `data.py`、`io.py`和`workflow.py`合并到`data_processing/__init__.py`。

旧文件保存在`/home/lixiangyu/sh/program/K-SPADA_source_backup_20260830/`，不参与当前包安装。

## 兼容性

- 保留`KSPADA.train()`和`KSPADA.pre()`原始风格接口。
- 保留教程使用的`train_model()`、`predict_model()`和`prepare_data_workflow()`。
- distribution名称仍为`kspada`，导入名称仍为`KSPADA`。
- 大型数据、模型权重和预测结果不打包进Python wheel。

## 变更记录

### 2026-08-30 - 核心结构精简

**变更内容**：将18个顶层Python模块收敛为8个职责明确的模块；将数据处理收敛为4个文件；移除重复命名和薄包装层。

**变更理由**：上一版过度拆分，结构比原始TOSICA复杂，不利于代码交接和直接阅读。

**影响范围**：Python包内部导入路径发生变化；README公开接口和四个教程使用方式保持不变。

**回退方式**：使用`/home/lixiangyu/sh/program/K-SPADA_source_backup_20260830/`中的源码备份。

### 2026-08-30 - 注释与废弃代码清理

**变更内容**：删除Python源码中的中文注释、注释掉的旧实验代码、脚本头和无作用字符串块；保留的函数说明统一使用简洁英文docstring。

**变更理由**：减少历史实验残留，提高核心模型、训练、预测和数据处理代码的可读性。

**影响范围**：仅清理注释和不可执行代码，不修改模型结构、训练参数、数据处理流程或公开接口。

**验证结果**：6项单元测试、editable安装、`pip check`以及4个教程Notebook的安全默认流程均通过。
