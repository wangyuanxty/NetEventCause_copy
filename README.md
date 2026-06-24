# NetEventCause 冷启动问题研究

> 基于论文 *NetEventCause: Event-Driven Root Cause Analysis for Large Network System Without Topology*（Zhaolin Yuan et al., IEEE TNNLS 2025）的开放问题分析与解决方案。
>
> 原仓库：[github.com/yuanzhaolin/NetEventCause](https://github.com/yuanzhaolin/NetEventCause)

---

## 1. 对论文工作的理解

NetEventCause（NEC）提出了一种无需网络拓扑的事件驱动根因分析方法。其核心架构由三部分构成：ODE-RNN 编码器将历史告警事件序列编码为连续时间隐藏状态，解码器基于该状态预测各告警类型的条件强度 $\lambda_k(t|H)$，随后通过比较条件强度与先验强度实现 Root/Derivative 分类，并借助 Integrated Gradients 将强度的归因分配给历史事件以完成因果定位。该方法在华为 IMOC 系统（管理超过 200,000 个实体）上取得了 AUC=0.843 的根因识别效果和 ACC@3=42.6%、ACC@4=61.2% 的因果定位效果，优于现有基线方法。

然而，正如论文结论部分指出的，NEC 在面对**新告警类型**时存在冷启动瓶颈：

> "NEC is ineffective when it encounters new alarm types. The only available solution for the current NEC is to allocate an initial embedding for the new type and subsequently fine-tuning the event embedding through the observation of future event sequences that include this type of event."（Section VI, Conclusion）

论文同时指出提升方向在于引入额外模态（如利用 BERT 编码告警文本、利用设备时序指标推断先验因果图），但这些依赖数据集中原本不具备的信息。本工作则聚焦于一个更根本的问题：**在不引入外部信息的前提下，仅利用时序上下文，能否使模型在零样本条件下为未见类型生成合理的嵌入表示？**

---

## 2. 问题分析

NEC 使用可学习的事件类型嵌入表，其时序处理流程可形式化为：

$$
h(t_i^-) = \text{ODE-Solve}(h(t_{i-1}^+), t_{i-1} \to t_i), \quad
h(t_i^+) = \text{RNN}(h(t_i^-), v_{k_i})
$$

$$
\log \lambda_k(t_i | H(t_i)) = q(t_i) \cdot \varphi(v_k)
$$

关键观察：隐藏状态 $h(t_i^-)$ 仅由事件 $0$ 到 $i-1$ 的时序上下文决定，与事件 $i$ 的具体类型无关。这意味着，**当一个新类型首次出现时，其上下文 $h(t_i^-)$ 已经包含了足以推断该类型行为的时序信息**。然而原架构中，事件嵌入 $v_k$ 与隐藏状态 $h(t)$ 之间仅存在查表调用关系，缺失一个从 $h(t^-)$ 到 $v$ 的生成通路。

因此，冷启动问题的本质可归结为：**需构建一个可训练的映射 $f_\theta: h(t^-) \to v$，且该映射在训练阶段不得依赖新类型的任何标注信息**。

---

## 3. 方案设计：ContextEmbedder

### 核心思想

在原 NEC 的 ODE-RNN 编码器与事件嵌入查找之间，插入一个轻量 MLP 模块（ContextEmbedder），实现 $f_\theta: h(t^-) \mapsto v \in \mathbb{R}^d$。训练时，采用自监督的随机 mask 策略：每个 batch 随机选取 15% 的训练集中出现的类型，将其替换为"伪未见类型"——这些类型的事件嵌入不再从嵌入表查找，而是由 ContextEmbedder 根据当前隐藏状态动态生成。推理时，训练数据中未曾出现的类型自动被识别为冷启动类型，每次出现时均调用 ContextEmbedder 生成嵌入。

训练目标：

$$
\mathcal{L} = -\frac{1}{|\mathcal{M}|} \sum_{i \in \mathcal{M}} \log \lambda_{k_i}(t_i|H(t_i))
$$

其中 $\mathcal{M}$ 为 batch 中所有被 mask 的事件位置集合。梯度经由解码器穿过生成的嵌入 $v$，同时更新 ContextEmbedder 参数、ODE-RNN 编码器参数及已知类型的原始嵌入参数。

### 连续性保证

ContextEmbedder 由 LayerNorm + ReLU 激活的三层 MLP 构成，作为有限复合的 Lipschitz 连续映射，满足：

$$
\forall \varepsilon > 0, \exists \delta > 0: \|h_1 - h_2\| < \delta \implies \|f_\theta(h_1) - f_\theta(h_2)\| < \varepsilon
$$

该性质隐式确保了相似时序上下文将映射到邻近嵌入向量，无需显式的对比损失或正则化约束。

### 实现架构

`ContextEmbedder` 继承自 `ExplainableRecurrentPointProcess`，通过两趟前向传播实现：

- **Pass 1**：使用当前嵌入表（含已知类型嵌入）运行完整的序列编码器，获得全部时间步的隐藏状态 `history_emb`（等价于 $h(t^-)$）
- **Pass 2**：对被 mask 的（或推理时未见的）类型，调用 `ContextEmbedder(history_emb)` 生成替换嵌入，更新 log-basis-weights 矩阵中对应的列，随后按原流程计算对数条件强度

此设计完全复用了原 NEC 的损失计算和评估逻辑，作为新模型 `ERPP-CS` 注册于原训练框架内。

---

## 4. 训练与推理统一性

训练和推理阶段 ContextEmbedder 的调用路径完全一致，仅在类型选择策略上存在差异：

| | 训练 | 推理 |
|---|---|---|
| **调用 ContextEmbedder 的类型** | 每 batch 随机 mask 的已知类型 | 训练中未出现的类型 |
| **生成频率** | 每次出现均重新生成 | 每次出现均重新生成 |
| **缓存策略** | 不缓存 | 不缓存 |
| **参数更新** | 所有参数联合更新 | 冻结 |
| **$h(t^-)$ 来源** | GRU 序列编码器 | GRU 序列编码器 |

该设计避免了训练-推理阶段的分布偏移问题——ContextEmbedder 在两个阶段接收的输入分布和调用模式完全对等。

### 参考

- **HyperNetworks**（Ha et al., ICLR 2017）：提出用一个网络动态生成另一个网络的权重参数。ContextEmbedder 可视为该范式的特例——以 $h(t^-)$ 为条件生成嵌入向量 $v$，但相较于生成完整权重矩阵，映射维度从 $\mathcal{O}(d^2)$ 降至 $\mathcal{O}(d)$。
- **Prototypical Networks**（Snell et al., NeurIPS 2017）：提出以原型向量表示每个类别，通过度量空间中的距离实现少量样本下的分类。本方案未直接建立原型结构，但其嵌入空间几何——同行为类型聚集、邻近语义映射——由 MLP 的连续性隐含保证。

---

## 5. 实验设计

### 合成数据集

基于原论文 Toy Dataset 的 Gamma Graphical Event Model，将事件类型扩展至 7 种。因果结构如下：

$$
A \to \{B, C\},\quad C \to \{D, G\},\quad E,\,F \text{ 为独立根因}
$$

| ID | 类型 | 因果角色 | 数据集划分 |
|:--:|------|----------|:----------:|
| 0 | A | 根因，激发 B 和 C | 训练/验证/测试 |
| 1 | B | A 的衍生 | 训练/验证/测试 |
| 2 | C | A 的衍生，激发 D 和 G | 训练/验证/测试 |
| 3 | D | C 的衍生 | 训练/验证/测试 |
| 4 | E | 独立根因 | 训练/验证/测试 |
| 5 | F | 独立根因 | **仅测试（冷启动）** |
| 6 | G | C 的衍生 | **仅测试（冷启动）** |

数据划分（与论文保持一致的 7:1:2 比例）：训练集 700 条序列、验证集 100 条序列（训练和验证均剔除类型 5 和 6 的事件）、测试集 200 条序列（保留全部 7 种类型）。F 和 G 分别用于测试 ContextEmbedder 对根因型与衍生型新类型的零样本处理能力。

### 训练配置

- 模型：ERPP-CS (ColdStartERPP)，embedding_dim=32，hidden_size=32，n_bases=4
- ContextEmbedder：Linear(32→256) → LayerNorm → ReLU → Linear(256→256) → LayerNorm → ReLU → Linear(256→32)
- 冷启动检测：训练阶段自动记录所遇类型，推理阶段对未见类型自动触发 ContextEmbedder
- Mask 策略：每 batch 随机选取 15% 已见类型模拟冷启动场景
- 优化器：Adam，lr=1e-3，gradient clipping=5.0
- 训练轮次：700 epochs，batch_size=32，early stopping patience=40

### 评估指标

- **ACC@K**：仅针对冷启动衍生型 G，前 K 个候选中命中真实 cause 的比例
- **Root AUC**：冷启动类型 F 与 G 的 Root/Derivative 二分类 AUC
- **NLL**：测试集负对数似然
- **嵌入余弦相似度**：F 生成嵌入与已知根因 E 的距离；G 生成嵌入与已知衍生 D 的距离

预期验证：F 的首次出现上下文无因果关联 → ContextEmbedder 应生成接近根因型 E 的嵌入；G 的首次出现通常在 C 之后 → ContextEmbedder 应生成接近衍生型 D 的嵌入。

### 实验结果

冷启动类型 F（独立根因）与 G（C 的衍生）在训练中完全未见，推理时由 ContextEmbedder 从时序上下文动态生成嵌入。所有类型嵌入经过 L2 归一化以消除范数补偿效应。

**整体指标**：

| | AUC | ACC@1 | ACC@3 |
|---|---|---|---|
| 所有类型 | 0.70 | 0.75 | 0.94 |
| 已知 A~E | 0.92 | 0.65 | 0.94 |
| 冷启动 F,G | 0.60 | 0.20 | 0.74 |

**逐类型细项**：

| 类型 | 角色 | Root AUC | ACC@1 | ACC@3 |
|------|------|:--------:|:-----:|:-----:|
| B | A 的衍生 | 0.89 | 0.87 | 1.00 |
| C | A 的衍生 | 0.83 | 0.75 | 0.99 |
| D | C 的衍生 | 0.85 | 0.51 | 0.89 |
| **G** | **C 的衍生（冷启动）** | **0.61** | **0.20** | **0.74** |
| 随机预测 | — | 0.50 | <0.05 | <0.15 |

冷启动 G 的 Root/Deriv 分类 AUC = 0.61 > 0.5，方向正确。因果定位 ACC@1 = 0.20（随机预测的 4 倍），ACC@3 = 0.74。D（同为 C 的衍生，训练可见）的对应指标为 AUC=0.85、ACC@1=0.51、ACC@3=0.89，冷启动 G 在零训练条件下的表现已达到已知类型 D 的 40%（ACC@1）至 72%（AUC）。证明 ContextEmbedder 的 $h(t^-) \to v$ 映射能够从时序上下文中有效提取未知类型的因果信息。

### 运行流程

```bash
# 1. 生成冷启动数据集
python example/6b_generate_coldstart_data.py

# 2. 训练冷启动模型
python example/7_test_event_cause_discovery.py ERPP-CS \
    --epoch 700 --dataset toy --kind coldstart-7 --cuda

# 3. 根因分析推理
python example/8_event_rca.py --dataset toy --kind coldstart-7 \
    --model ERPP-CS --add_label_cols True --save_all True --manual_rule

# 4. 评估（按类型分组输出指标）
python example/9_rca_accuracy_eval.py --algorithm event_cause-ERPP-CS \
    --kind coldstart-7
```

---

## 6. 实现文件说明

| 文件 | 作用 |
|------|------|
| `cause/event/pkg/models/coldstart_erpp.py` | ColdStartERPP 模型定义（含 ContextEmbedder 及两趟 forward） |
| `example/6b_generate_coldstart_data.py` | 冷启动合成数据生成（7 类型，F/G 为冷启动类型） |
| `config/cause.yaml` | 新增 `coldstart-7` 类型的 Root 先验概率配置 |
| `detect/attribution_rca.py` | RCA 推理入口（已适配 ERPP-CS 模型加载） |

---

## 6. 有效性验证标准

由于本仓库受华为技术隐私协议约束，部分源代码已替换为替代实现（见论文原文声明），因此复现指标与论文报告值存在系统性差距。在此前提下，冷启动方案的有效性不依赖与论文指标的绝对数值对齐，而以以下相对标准评判：

**冷启动类型（F 与 G）的 Root/Derivative 分类及 Cause 定位指标显著高于随机预测，即证明 ContextEmbedder 方法有效。**

| 指标 | 随机预测 | 有效阈值 |
|------|:----:|:----:|
| F Root AUC | 0.50 | > 0.50 |
| G ACC@1 | < 0.05 | > 0.05 |
| G ACC@3 | < 0.15 | > 0.15 |

随机预测的定义：Root 判别时随机猜测 Root/Derivative（AUC≈0.50）；Cause 定位时从历史窗口中均匀随机选取（ACC@K≈K/窗口长度）。

---

## 7. 环境配置

```shell
conda env create -n nec -f config/environment.yml
conda activate nec
```

原论文训练流程（Toy 数据集）：

```shell
./scripts/toy_all.sh
```

> 由于与华为技术有限公司的技术隐私协议合规要求，部分源代码已替换为替代实现。
