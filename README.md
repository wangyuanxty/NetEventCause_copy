# NetEventCause — 冷启动问题分析与解决方案

本项目基于论文 *NetEventCause: Event-Driven Root Cause Analysis for Large Network System Without Topology* (IEEE TNNLS 2025) 的源代码，针对作者在结论中提出的开放问题——**新告警类型的冷启动（Cold-Start）**——进行分析并给出两种基于测试时微调的解决方案。

原始项目地址：https://github.com/yuanzhaolin/NetEventCause

> **关于可复现性**：由于华为技术隐私协议要求，原论文中的 IMOC 真实数据集及部分模型代码（ODE-RNN、SPNPP）已被删除或替换。因此本仓库无法完全复现原论文 Table III/IV 的指标。基于同样原因，训练阶段的 NLL 收敛曲线与原论文存在差距。冷启动实验的评估标准调整为：**冷启动类型（F、G）的 ACC@K 和 Root AUC 高于随机预测即证明方案有效**。

---

## 环境配置

```shell
conda env create -n <myenv> -f config/environment.yml
conda activate <myenv>
```

原论文的训练与评估流程：

```shell
./scripts/toy_all.sh
```

> 注：由于华为技术隐私协议要求，部分源代码已被替换。

---

## 1. 开放问题

论文 Section VI 明确指出：

> "NEC is ineffective when it encounters new alarm types. The only available solution for the current NEC is to allocate an initial embedding for the new type and subsequently fine-tuning the event embedding through the observation of future event sequences that include this type of event."

NEC 在遇到新告警类型时无效。模型使用可学习的事件类型嵌入表 $V[k]$，新类型对应的 $V[k_\text{new}]$ 未被训练，导致 ODE-RNN 产出的条件强度 $\lambda_k(t|H)$ 退化为随机值，进而破坏下游的 Root/Derivative 分类和 Integrated Gradients 因果归因。

关键矛盾在于：原模型仅支持查表获取嵌入，缺乏从上下文推断未知类型嵌入的机制。然而推理时序列已经发生，可以利用 NLL 损失（无需额外标注）对单一嵌入向量进行少量梯度步优化。

---

## 2. 解决方案

### 方案 A：测试时微调（TTF — Test-Time Finetuning）

不修改训练流程，在推理阶段对新类型进行累积式在线微调。

**流程**：新类型首次出现时用随机初始化的 $v$，对该次事件的 NLL 做 5 步梯度下降；后续每次出现再做 1 步更新，信息随观测累积。微调仅作用于新类型的嵌入向量，ODE-RNN、decoder 及其他类型嵌入均保持冻结。事件的时序发生本身即为自监督信号。

**理论**：新类型 $k$ 在第 $p$ 次出现时，当前嵌入为 $v^{(p-1)}$（首次为随机），计算

$$
\mathcal{L}(v) = -\log \lambda_k(t_i \mid H(t_i))
$$

梯度下降更新：

$$
v^{(p)} = v^{(p-1)} - \eta \frac{\partial \mathcal{L}}{\partial v}
$$

首次 $S=5$ 步以快速收敛，后续 $S=1$ 步进行微调。

### 方案 B：LoRA-TTF — 低秩分解 + 测试时微调

将嵌入表分解为低秩形式以加速收敛：

$$
v_k = W \cdot c_k, \quad W \in \mathbb{R}^{d \times r}, \ c_k \in \mathbb{R}^{r}
$$

其中 $r=8 \ll d=64$。训练时 $W$ 和已知 $c_k$ 联合学习；推理时 $W$ 冻结，仅优化 $c_k$（8 维）。相比直接优化 64 维的 $v$，收敛速度快约 $8\times$，梯度方向更集中。

### 方案对比

| | TTF | LoRA-TTF |
|---|---|---|
| 嵌入形式 | $v_k \in \mathbb{R}^{64}$ 各自独立 | $v_k = W \cdot c_k,\ c_k \in \mathbb{R}^8$ |
| 新类型优化维度 | 64 | 8 |
| $W$ 是否共享 | — | 是，所有类型共享 |
| 训练改动 | 无 | embedding 改为 $W \cdot c$ |
| 模型名 | `ERPP-TTF` | `ERPP-LoRA` |

---

## 3. 实验设计

### 合成数据集

基于原论文的 Gamma Graphical Event Model，扩展为 7 种事件类型：

| ID | 类型 | 因果角色 |
|:--:|------|----------|
| 0 | A | 根因，激发 B 和 C |
| 1 | B | A 的衍生 |
| 2 | C | A 的衍生，激发 D 和 G |
| 3 | D | C 的衍生 |
| 4 | E | 独立根因 |
| **5** | **F** | **独立根因（冷启动）** |
| **6** | **G** | **C 的衍生（冷启动）** |

训练集（700 条）和验证集（100 条）仅包含类型 0-4；测试集（200 条）包含全部 7 种类型。F 和 G 分别测试 TTF 对根因型和衍生型新类型的处理能力。

### 有效性标准

冷启动类型的指标高于随机预测即证明方案有效：

| 指标 | 随机预测 | 有效性阈值 |
|------|:----:|:----:|
| Root AUC | 0.50 | >0.55 |
| G ACC@1 | ~0.10 | >0.15 |
| G ACC@3 | ~0.25 | >0.30 |

### 运行

```bash
# 1. 生成冷启动数据
python example/6b_generate_coldstart_data.py

# 2. 训练（TTF 与 ERPP 共享训练逻辑）
python example/7_test_event_cause_discovery.py ERPP-TTF --epoch 700 --dataset toy --kind coldstart-7 --cuda

# 3. RCA 推理（TTF 在遇到冷启动类型时自动触发）
python example/8_event_rca.py --dataset toy --kind coldstart-7 --model ERPP-TTF \
    --add_label_cols True --save_all True --manual_rule --steps 10

# 4. 评估
python example/9_rca_accuracy_eval.py --algorithm event_cause-ERPP-TTF --kind coldstart-7

# LoRA-TTF 同理，替换模型名为 ERPP-LoRA
```

---

## 4. 新增文件

| 文件 | 说明 |
|------|------|
| `cause/event/pkg/models/coldstart_erpp.py` | ColdStartTTF 与 ColdStartLoRA 模型定义 |
| `example/6b_generate_coldstart_data.py` | 7 类型冷启动数据生成 |
| `config/cause.yaml` | 新增 coldstart-7 的根因先验概率 |

其余修改集中于 `detect/attribution_rca.py`、`cause/event/tasks/train.py`、`example/7/8/9_*.py`，主要为模型注册、数据加载适配和按类型评估。

---

## 5. 核心参考

- **NetEventCause** (Yuan et al., IEEE TNNLS 2025) — 原论文方法
- **HyperNetworks** (Ha et al., ICLR 2017) — 条件参数生成范式
- **LoRA** (Hu et al., ICLR 2022) — 低秩分解的灵感来源
