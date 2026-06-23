
# NetEventCause: Event-Driven Root Cause Analysis for Large Network System Without Topology

**Authors:** Zhaolin Yuan, Long Ma, Wenjia Wei, Xia Zhu, Mingjie Sun, Duxin Chen, and Xiaojuan Ban  
**Journal:** IEEE Transactions on Neural Networks and Learning Systems, Vol. 36, No. 10, October 2025  
**DOI:** 10.1109/TNNLS.2025.3574316  
**Code:** https://github.com/yuanzhaolin/NetEventCause

---

## Abstract

Root cause analysis (RCA) is a crucial technique in network systems for uncovering the abnormal nodes that lead to the network alarm flood. Within private cloud network systems, the calling chains and topologies among entities, such as hosts, routes, and services, are always incomplete due to nonstandardized management.

Existing topology-free RCA techniques, which rely on causal discovery, are inapplicable when the scale of the network system is extremely large or the number of triggered alarms is sparse. This article proposes **NetEventCause (NEC)**, an event-driven, unsupervised, and nonintrusive RCA algorithm for large network systems, where the network topology is unknown.

NEC learns from historical alarm events to model the occurrences of various alarm types using a multivariate neural temporal point process (TPP). Based on the conditional intensity predicted by the learned TPP, NEC can identify the root alarms from a cascade of alarm events and locate the causal alarms of derivative alarms using the attribution method.

The experimental section evaluates NEC using both a synthetic event dataset and a large real-world dataset. The real-world dataset is exported from the Huawei Shennong Intelligent Maintenance and Operation Center (IMOC), a platform deployed at one of China’s largest airports and managing over 200,000 entities.

Results obtained from the two datasets demonstrate that NEC outperforms most state-of-the-art TPP models in modeling alarm events and surpasses general RCA methods in terms of identifying root alarms and recovering transmission chains of anomalies.

**Index Terms:** Artificial intelligence for IT operations (AIOps), attribution method, data stream mining, event modeling.

---

## Nomenclature

| Symbol | Definition |
|---|---|
| `M` | Candidate set of alarm event types |
| `s` | Streaming alarm sequence |
| `e_i ∈ s` | Alarm event, denoted as `e_i = (k_i, t_i)` |
| `k_i ∈ M` | Type of event `e_i` |
| `t_i` | Occurring time of event `e_i` |
| `v_k` | Feature embedding of event type `k` |
| `H(t)` | Historical events occurring before `t` |
| `λ_k(t)` | Intensity of event type `k` at time `t` |
| `N_k` | Count of events with type `k` occurring as root events in the training dataset |
| `T` | Total duration of the training dataset |
| `A_i` | Contribution vector of event `e_i` |
| `R̂_{e_i}` | Inferred causative alarm set of event `e_i` |

---

## I. Introduction

In cloud network systems, failures or anomalies are frequently raised by network entities, such as devices, hosts, and services. These issues are typically represented as frequent alarm events. Due to the strong relations between linked network entities, a root alarm usually causes massive derivative alarms, resulting in an alarm flood for network operators.

Root cause analysis (RCA) is a crucial technique in the field of artificial intelligence for IT operations (AIOps). It aims to determine whether the reported alarm events are caused by the entities themselves or by other failures. A common challenge of RCA within private cloud systems is the incomplete calling chains and topologies due to nonstandardized management. This limitation renders most existing tracing-based RCA techniques inapplicable in these scenarios.

To recover the missed relations between entities, previous studies introduced causal discovery algorithms, including Peter Clark (PC), Granger causality in time series, and event causal discovery. However, existing methods suffer from several challenges in discovering causality in large network systems:

1. A large private network system may contain tens of thousands of entities, and the dimensions of time series and event types are too high for most causal discovery algorithms.
2. The number of triggered alarms on abnormal entities is sparse, so inadequate log datasets degrade the accuracy of restored topology.
3. Most event-based RCA models, such as Hawkes process, proximal graphical event model, and semiparametric neural point process (SPNPP), rely on strong assumptions about event intensities, such as memoryless properties and decaying excitation.

To tackle these challenges, this article proposes **NetEventCause (NEC)**, a novel RCA technique for network systems of arbitrary scale and unknown topology. NEC is an unsupervised RCA technique driven by log events and is capable of:

1. Modeling Granger causality among alarm events in continuous time.
2. Distinguishing root alarms and derivative alarms from a cascade of alarm events.
3. Inferring direct causative alarms of derivative alarms as a directed transmission graph.

NEC builds an ODE-RNN-based continuous-time temporal point process to learn the conditional intensity of each alarm type at any timestamp given historical alarms. The predicted intensities are used to classify whether an alarm is a root alarm by incorporating the prior intensity of each alarm type. To identify direct causative alarms, NEC introduces attribution methods from explainable AI to measure the Granger causality from each past alarm on evaluated intensities of new alarms.

The contributions are summarized as follows:

1. NEC is an unsupervised, nonintrusive, log-event-driven RCA technique for large network systems where topology is unknown and the number of entities is extremely large.
2. NEC employs a novel continuous-time neural TPP model to fit complicated excitation between distinct alarm events and integrates evaluated conditional intensities with attribution methods to identify root alarms and causative alarms.
3. NEC is applied and verified in Huawei Shennong IMOC, a large network system managing more than 200,000 entities, such as servers, routers, clusters, and containers.

---

## II. Related Work

In most private cloud systems, calling chains and topology are usually incomplete because of nonstandardized management. To recover missed relations between entities, some studies introduce causal discovery algorithms to identify missed topology.

Previous studies represent alarm events of each entity as time series, defined as the number of reported alarms in time bins. Temporal causal discovery methods, such as PC and Granger causality, are then applied to these multidimensional time series to find causation relationships between entities.

Event models are also employed to estimate the conditional intensity of alarm events for each entity and assess mutual excitation between entities. The causal graphs generated by these methods consist of vertices representing entities and directed edges modeling anomaly propagation. Candidate entities that may induce root causes are identified by traversing causality graphs using random walk algorithms.

However, existing methods face several challenges in large private cloud network systems:

- Most causal discovery algorithms operate on entities and rely on sufficiently frequent triggered alarms.
- Alarm occurrences are usually long-tail and sparse, which degrades restored topology accuracy.
- Existing techniques cannot handle cold-start scenarios where a new entity raises its first alarm.
- A large network system may contain tens of thousands of entities, making the dimensions too high for many causal discovery algorithms.

These challenges motivate an RCA technique that is applicable in large topology-free network systems and generalizable to new entities.

Many existing event models make idealized assumptions about event intensities. Hawkes process assumes excitation gradually decays with time. Proximal graphical event models assume constant conditional intensities in influential time windows. SPNPP introduces mixed Gaussian bases to predict intensities. However, in network systems, time differences between root and derivative alarms are affected by factors such as self-check periods and network delay. Therefore, actual alarm intensities may not follow these assumptions.

NEC models multivariate TPPs of alarm events using a continuous-time neural network without making assumptions about event intensities.

---

## III. Problem Definition

This section formulates RCA from the perspective of events. Taking the IMOC dataset as an example, four key fields are reserved for each alarm event:

- Alarm type
- Alarm description
- Entity type
- Timestamp

> **Table I. Example of Four Triggered Alarms**  
> The table is present in the original PDF but its cell contents were not fully extracted from the text layer.

A streaming alarm sequence `s` consists of `|s|` alarm events sorted by timestamp:

```math
s = \{ e_i = (t_i, k_i) \}_{i=1}^{|s|}
```

For each event, `t_i ∈ R+` is the normalized timestamp of the `i`-th event, and `k_i ∈ M` is the corresponding event type.

To distinguish different alarms and generalize the learned TPP on new entities with the same entity type, NEC defines the event type of alarm `a_i` as the combination of its alarm type and entity type:

```math
k_i = (a_i.at, a_i.et)
```

Because topology is unknown and many alarms are triggered from new devices, entity identity, such as entity name and ID, cannot provide meaningful information for RCA. NEC therefore omits identity and uses entity type, alarm type, and triggering time to analyze propagation chains.

Each alarm `e_i` is either a root alarm or a derivative alarm of other past causative alarms. The past causative alarm set of `e_i` is denoted as:

```math
R_{e_i}
```

It is assumed that recorded triggering timestamps are accurate and causative alarms always occur before their derivative alarms:

```math
\forall e_j \in R_{e_i}, \quad t_j < t_i
```

This study aims to recover directed anomaly propagation among alarms, represented as a directed acyclic graph. Specifically, the algorithm focuses on:

1. Distinguishing root alarms and derivative alarms.
2. Identifying the possible causative alarm set `R_{e_i}` for each alarm `e_i`.

---

## IV. NetEventCause

The workflow of NEC consists of the following steps:

1. Train a neural continuous-time TPP using an offline alarm dataset to model multivariate alarm event sequences in an unsupervised manner.
2. Use the conditional intensity predicted by the neural TPP, given historical alarms, to estimate the conditional probability that a new alarm is a root alarm.
3. Use the differentiability of the neural TPP to measure the contribution of past events to subsequent alarms through attribution methods, thereby discovering Granger causalities between alarms.

> **Fig. 1. Overall structure of NEC.**  
> The original figure is not included in the extracted text.

---

### A. Modeling Alarm Events as a Multivariate TPP

NEC models alarm event sequences using a continuous-time neural TPP based on an encoder-decoder architecture. The goal is to predict the event intensity:

```math
\lambda_k(t \mid H(t))
```

at any time `t`, given historical triggered alarm events `H(t)`. The model is trained in an unsupervised manner without annotated datasets from IT experts.

In the encoding stage, each event with type `k_i` is transformed into a feature vector:

```math
v_{k_i} = V[k_i]
```

The parameter dictionary `V` stores changeable keys and parameters, which is useful for incremental learning.

The hidden state `h(t)` defined in the continuous-time domain is inferred by an ODE-RNN:

```math
h(t_i^-) = \text{ODESolve}(h(t_{i-1}^+), f_{\theta_1}, t_{i-1}, t_i)
```

```math
h(t_i^+) = \text{RNN}_{\theta_2}(h(t_i^-), v_{k_i})
```

where `f_{θ1}` and `RNN_{θ2}` are trainable modules. Here, `h(t_i^-)` represents the hidden state at `t_i` before the event feature at `t_i` is fed into the model, while `h(t_i^+)` is the updated embedding after the model observes event type `k_i` at time `t_i`.

In the decoding stage, the hidden state `h(t_i^-)` is fed to a neural network to predict a query embedding `q(t_i)`. The logarithmic conditional intensity of any event type `k` is estimated as:

```math
\log \lambda_k(t_i) = q(t_i) \cdot \psi(v_k)
```

```math
q(t_i) = \text{NN}_{\theta_3}(h(t_i^-))
```

By recursively feeding historical events `H(t)` into the ODE-RNN, the trained model can predict the conditional intensity of event type `k` at timestamp `t`.

The neural encoder-decoder model is trained by minimizing the expected negative log likelihood (NLL) over all sequences in dataset `S`:

```math
L_{\text{NLL}}(\Theta)
=
\mathbb{E}_{s \in S}
\sum_{i=1}^{|s|}
\left[
-\log \lambda^s_{k_i^s}(t_i^s)
+
\sum_{k \in M}
\int_{t_{i-1}^s}^{t_i^s}
\lambda^s_k(t')dt'
\right]
```

where:

```math
\Theta = \{\theta_1, \theta_2, \theta_3, \psi, V\}
```

---

### B. Identifying Root Alarms by Comparing Conditional Intensity and Prior Intensity

To differentiate root alarms and derivative alarms, NEC introduces a latent discrete random variable:

```math
c \in \{0,1\}
```

where:

- `c = 0` means `e_i` is a root alarm.
- `c = 1` means `e_i` is a derivative alarm.

Given historical events `H(t_i)` and event `e_i` at time `t_i`, inferring `c` is equivalent to estimating:

```math
P(c = 0 \mid e_i, H(t_i))
```

Using conditional probability:

```math
P(c = 0 \mid e_i, H(t_i))
=
\frac{
P(c = 0, e_i)
}{
P(c = 0, e_i) + P(c = 1, e_i \mid H(t_i))
}
```

The probability that an event occurs at a specific time can be represented by a Poisson distribution with parameter `λ`:

```math
P(N([t_i, t_i + \Delta t]) > 0) = 1 - e^{-\lambda \Delta t}
```

Using the limit theorem `lim_{x→0}(1-e^{-x}) = x`, NEC rewrites the root probability as:

```math
P(c = 0 \mid e_i, H(t_i))
=
\frac{
\bar{\lambda}_{k_i}
}{
\bar{\lambda}_{k_i} + \lambda_c
}
```

where:

```math
\lambda_c = \lambda_{k_i}(t_i \mid H(t_i)) - \bar{\lambda}_{k_i}
```

and `\bar{\lambda}_{k_i}` is the prior intensity that an event with type `k_i` occurs as an individual root event.

The prior intensity is estimated as:

```math
\bar{\lambda}_k = \frac{N_k}{T}
```

where `T` is the total duration of the training dataset, and `N_k` is the count of events of type `k` occurring as root events.

NEC regards alarm `e_i` as a root alarm if:

```math
P(c = 0 \mid e_i, H(t_i)) \ge \delta
```

The default threshold is:

```math
\delta = 0.2
```

---

### C. Discovering Causal Relationships by Attribution Method

NEC uses attribution methods to discover causal relationships among events. Since the conditional intensity is modeled by the neural TPP, attribution can identify which past events in `H(t)` significantly contribute to the occurrence of an event at time `t`.

When analyzing the causes of event `e_i`, the conditional intensity predicted by the neural TPP is regarded as the target function. The input is defined as the concatenation of feature embeddings and timestamps of historical events:

```math
x =
[H(t_i), t_i, v_{k_i}]
=
[t_1, v_{k_1}, \ldots, t_{i-1}, v_{k_{i-1}}, t_i, v_{k_i}]
```

The baseline input has the same size, with historical event embeddings substituted by zero vectors:

```math
\bar{x}
=
[\bar{H}(t_i), t_i, v_{k_i}]
=
[t_1, 0, \ldots, t_{i-1}, 0, t_i, v_{k_i}]
```

NEC uses integrated gradients (IG) to generate the contribution vector `A_i`:

```math
A_i = IG(f, x, \bar{x})
```

```math
A_i =
(x - \bar{x})
\odot
\int_0^1
\frac{\partial f(\tilde{x})}{\partial \tilde{x}}
\bigg|_{\tilde{x} = \bar{x} + \alpha(x-\bar{x})}
d\alpha
```

where:

```math
f(\tilde{x}) = \log \lambda_{k_i}(t_i \mid \tilde{H})
```

The scalar value `A_i(j)` represents the contribution from historical event `e_j` to the occurrence of event `e_i`.

NEC identifies the top-`K` causative alarms as the events with the highest attribution scores:

```math
\hat{R}_{e_i}[r] = e_{j^*}
```

where `e_{j^*}` has the `r`-th largest contribution score.

To ensure that the predicted intensities given null events are close to prior intensities, NEC introduces regularization:

```math
L_{\text{reg}}(\Theta)
=
\sum_{s=1}^{S}
\sum_{i=1}^{|s|}
\sum_{k \in M}
\left|
\lambda_k(t_i \mid \bar{H}(t_i))
-
\bar{\lambda}_k
\right|^2
```

The total loss is:

```math
L_{\text{NEC}}(\Theta)
=
L_{\text{NLL}}(\Theta)
+
L_{\text{reg}}(\Theta)
```

---

### Algorithm 1: Training of Neural TPP in NEC

**Input:** Event sequence dataset `S`, initialized parameters `Θ`, and root-event intensity `\bar{\Lambda}`  
**Output:** Trained parameters `Θ`

```text
1. for each event sequence s ∈ S do
2.     for each event e_i ∈ s with past events H(t_i) do
3.         Generate a null event sequence H̄(t_i) by replacing event embeddings in H(t_i) with 0.
4.         for each event type k ∈ M do
5.             Calculate λ_k(t_i | H(t_i)) and λ_k(t_i | H̄(t_i)).
6.         end for
7.     end for
8.     Compute NLL and regularization term.
9.     Update parameters Θ to minimize L_NEC(Θ).
10. end for
11. return Θ
```

---

### Algorithm 2: Identifying Root Event and Analyzing Causal Events

**Input:** Event `e_i`, past event sequence `H(t_i)`, trained neural TPP parameters `Θ`, root-event threshold `δ`, root-event intensity `\bar{\Lambda}`, and candidate set size `K`  
**Output:** Type of `e_i`, list of causal events

```text
1. Infer conditional intensity λ_{k_i}(t_i | H(t_i)).
2. Estimate P(c = 0 | e_i, H(t_i)).
3. if P(c = 0 | e_i, H(t_i)) ≥ δ then
4.     return type: Root, []
5. else
6.     Compute contribution vector A_i.
7.     Identify causal events from H(t_i) with top-K contributions.
8.     return type: Derivative, [R̂_{e_i}[1], ..., R̂_{e_i}[K]]
9. end if
```

---

### D. Summary of NEC

The time complexity of Algorithm 1 is dominated by solving the NLL. Assuming each event sequence has length `N` and solving the ODE requires `K` invocations of the differential network, the cost is:

```math
O(NK|M|)
```

Identifying whether an event is a root event costs:

```math
O(1)
```

For causal event localization, computing integrated gradients dominates the cost. If the number of integration steps is `K'`, the cost for identifying causal events from past events is:

```math
O(NKK')
```

Therefore, the complexity of RCA for the entire sequence is:

```math
O(N^2KK')
```

In practice, the sequence length `N` is defined by the maximum possible transmission time from root alarms to derivative alarms. In the real-world application, `N` is less than 20, and RCA for each event is significantly faster than the generation of new alarms.

---

## V. Experiments

The experiments evaluate NEC using both a synthetic event dataset and a large real-world dataset. Two research questions are considered:

1. Does the proposed continuous-time neural TPP effectively fit alarm event sequences?
2. How accurate is NEC in identifying root alarms and locating causative alarms of derivative alarms?

---

### A. Introduction to the Datasets

#### 1. IMOC Dataset

The real-world dataset is exported from Huawei Shennong IMOC from January 2021 to May 2023. IMOC is deployed in one of China’s largest airports and manages about 200,000 entities, including network devices, containers, and microservices.

Preprocessing consists of:

1. Alarm filtering.
2. Alarm compression.

Alarm types with fewer than five events are removed. Repeated alarms triggered by the same entity simultaneously are compressed. The final dataset contains:

- 125,384 alarm events
- 3,634 entities
- 518 event types

> **Fig. 2. Frequency distribution histogram of the number of alarms that one entity ever reported in the dataset.**

The number of alarms reported by each entity follows a long-tail distribution. Approximately one-third of entities reported only one alarm.

For TPP training, 10,000 event sequences are randomly selected from January 2021 to December 2022. Each sequence spans less than 24 hours. The split is:

- 60% training
- 20% validation
- 20% testing

For RCA evaluation, logs from January 2023 to May 2023 are used. Ground truths of causative alarms are manually annotated by domain experts.

Because the event log is streaming and long, NEC uses a sliding window strategy. A window of size `W` moves with distance `d`, where `d << W`. At each step, alarms in the window are input to the detection module, and results from the last range of `d` are retained.

#### 2. Synthetic Dataset

The synthetic dataset simulates alarm events in a network system. It contains five event types:

```text
A, B, C, D, E
```

Event types A and E are root event types with constant base intensities. Event types B, C, and D may occur as root or derivative events.

The excitation from causative event type `k_i` to derivative event type `k_j` is modeled using a scaled gamma distribution:

```math
c_{k_i,k_j}(\Delta t)
=
\begin{cases}
r \cdot
\frac{\beta^\alpha \Delta t^{\alpha-1}e^{-\beta \Delta t}}{\Gamma(\alpha)},
& t \le W \\
0,
& t > W
\end{cases}
```

The total intensity of event type `k` at time `t` is:

```math
\lambda_k(t)
=
\sum_{e_i \in H(t)}
c_{k_i,k}(t - t_i)
+
B
```

The task is to detect root events and identify the sole causative event for each derivative event.

The synthetic dataset contains:

- 1,000 sequences
- Average sequence length about 100
- 700 training sequences
- 100 validation sequences
- 200 testing sequences

The dataset imitates three challenges:

1. Conditional intensity does not simply decay over time.
2. Events with the same type may occur simultaneously but originate from different entities.
3. A specific event type may occur as either a root event or derivative event.

> **Fig. 3. Event intensity of each event type varies with time difference Δt.**  
> **Table II. Derivative relationship between event types in the synthetic dataset.**

---

### B. Experiment 1: Evaluating Neural TPP

#### Baselines

NEC is compared with the following baselines:

1. **SPNPP** and **RPPN**: state-of-the-art neural TPP models.
2. **HEXP, HSG, NHPC**: Hawkes process with exponential kernels, Gaussian kernels, and nonparametric Hawkes process.

#### Experimental Setting

The ODE functions and NLL are solved using the Dopri5 solver in the `torchdiffeq` package.

Hyperparameters:

| Item | Value |
|---|---|
| Event embedding size | 64 |
| Derivative network | Single-layer MLP |
| Derivative hidden size | 32 |
| Decoder networks | Single-layer MLPs |
| Decoder hidden/output size | 64 |
| Activation | `tanh` |
| Max epoch | 500 |
| Early stopping patience | 50 epochs |
| Optimizer | Adam |
| Learning rate | `1e-3` |

Training the TPP on the IMOC dataset took about 3 hours on an Intel Xeon Silver 4310 processor and an NVIDIA GeForce RTX 3090 GPU.

#### Evaluation Metrics

Two metrics are used:

1. **NLL**: evaluates accuracy of predicted conditional intensity.
2. **Accuracy**: predicts the event type occurring at `t_i` given historical events before `t_i`.

#### Results

> **Fig. 4. Training and evaluation losses over epochs.**  
> **Fig. 5. NLLs and accuracies on test dataset of various trained event models.**

The proposed continuous-time neural TPP outperforms baselines on both NLL and accuracy. This verifies that the unrestricted ODE-RNN is better suited for fitting complicated time-varying intensities under excitation among multiple alarm event types.

---

### C. Experiment 2: Identifying Root Alarms and Locating Derivative Alarms

This experiment evaluates NEC in identifying root alarms from streaming alarms and locating causative alarms of each derivative alarm.

#### Evaluation Metrics

Root alarm detection is a binary classification problem, evaluated by:

```text
AUC
```

Causative alarm localization is evaluated by `ACC@k`:

```math
ACC@k =
\frac{1}{|A|}
\sum_{e_i \in A}
\frac{
\sum_{1 \le r \le k}
\mathbf{1}[\hat{R}_{e_i}[r] \in R_{e_i}]
}{
\min(k, |R_{e_i}|)
}
```

In AIOps, recall is often more important than precision because engineers can manually verify false positives at relatively low cost.

#### Baselines

The baselines are:

1. **SPNPP + Attribution**
2. **Frequency Item Set (FIS)**
3. **Peter Clark (PC)**
4. **CAUSE**
5. **Random Guess (RG)**

#### Results

> **Table III. Comparison of NEC and baselines on the IMOC dataset.**  
> **Table IV. Comparison of NEC and baselines on the synthetic dataset.**

NEC achieves the best AUC value on the IMOC dataset:

```text
AUC = 0.843
```

NEC also outperforms baselines in locating causative alarms. Static event-type graph methods perform worse because they model causality between event types rather than specific events. NEC directly models conditional intensity at the event level and incorporates timestamp information.

---

### D. Balancing Time Cost and Accuracy in RCA

The time complexity of NEC for analyzing causes of an event is:

```math
O(NKK')
```

where `K'` is the number of integration steps for integrated gradients.

An ablation study varies `K'` among:

```text
3, 5, 10, 20, 30
```

> **Fig. 6. Relationship between time consumption and accuracy in RCA.**

Results show that accuracy remains usable even with only three steps. Time increases linearly with the number of steps, while accuracy improvements almost stop when `K'` exceeds 20. In practice, reducing the number of steps can accelerate detection without significantly compromising accuracy.

---

### E. Case Analysis

> **Fig. 7. Visualization of detected causative alarms compared with ground truth.**  
> (a) Ground-truth causative alarms.  
> (b) NEC.  
> (c) CAUSE.

Fig. 7 shows a representative example of detected causative alarms. Each marker color and shape corresponds to an event type. Directed arcs represent discovered causality. Correct arcs are marked in green, and incorrect arcs are marked in red.

The case shows that NEC can distinguish alarms with the same types originating from different entities and accurately identify causative alarms, while CAUSE sometimes fails because it is based on a static causal graph of event types.

NEC only uses triggering timestamps and event types to detect root causes, making it suitable for generalizing to private cloud systems without manual annotations and network topology.

---

## VI. Conclusion

This article proposes NEC for RCA in large network systems where topology among devices and services is unknown. As a nonintrusive and unsupervised log-event-based RCA technique, NEC uses an ODE-RNN backbone to model temporal alarm events as a multivariate TPP.

The trained TPP evaluates conditional intensities of alarms given previously occurred alarms. Predicted intensities are used to identify root alarms via Bayesian inference and local causative alarms via attribution methods.

Experiments on a synthetic dataset and a real Huawei Shennong IMOC dataset show that:

- The ODE-RNN-based neural TPP outperforms existing SOTA neural TPPs in goodness-of-fit.
- NEC achieves strong root alarm identification performance:

```text
AUC = 0.843
```

- NEC achieves strong causative alarm localization performance:

```text
ACC@3 = 42.6%
ACC@4 = 61.2%
```

Current NEC is limited because it only uses timestamps and event types. It is ineffective when encountering new alarm types. A possible solution is to allocate an initial embedding for the new type and fine-tune it using future event sequences.

Future work will incorporate additional modalities, such as:

- Alarm log text encoded by language models like BERT.
- Time-series metrics of devices, such as CPU utilization and time delay.
- Prior Granger causal graphs between entities.

---

## References

[1] J. Soldani and A. Brogi, “Anomaly detection and failure root cause analysis in (micro) service-based cloud applications: A survey,” *ACM Computing Surveys*, vol. 55, no. 3, pp. 1–39, 2022.

[2] Y. Dang, Q. Lin, and P. Huang, “AIOps: Real-world challenges and research innovations,” in *Proc. IEEE/ACM ICSE Companion*, 2019, pp. 4–5.

[3] P. Wang et al., “CloudRanger: Root cause identification for cloud native systems,” in *Proc. IEEE/ACM CCGRID*, 2018, pp. 492–502.

[4] A. Brandón et al., “Graph-based root cause analysis for service-oriented and microservice architectures,” *Journal of Systems and Software*, vol. 159, 2020, Art. no. 110432.

[5] Y. Meng et al., “Localizing failure root causes in a microservice through causality inference,” in *Proc. IEEE/ACM IWQoS*, 2020.

[6] Y. Liu et al., “Simplified Granger causality map for data-driven root cause diagnosis of process disturbances,” *Journal of Process Control*, vol. 95, pp. 45–54, 2020.

[7] V. Arya et al., “Evaluation of causal inference techniques for AIOps,” in *Proc. ACM CODS-COMAD*, 2021, pp. 188–192.

[8] J. Qiao et al., “Structural Hawkes processes for learning causal structure from discrete-time event sequences,” in *Proc. IJCAI*, 2023, pp. 5702–5710.

[9] A. G. Hawkes, “Spectra of some self-exciting and mutually exciting point processes,” *Biometrika*, vol. 58, pp. 83–90, 1971.

[10] W. Zhang et al., “CAUSE: Learning Granger causality from event sequences using attribution methods,” in *Proc. ICML*, vol. 119, 2020, pp. 11235–11245.

[11] P. Linardatos, V. Papastefanopoulos, and S. Kotsiantis, “Explainable AI: A review of machine learning interpretability methods,” *Entropy*, vol. 23, no. 1, p. 18, 2020.

[12] C. Chen et al., “BALANCE: Bayesian linear attribution for root cause localization,” *Proc. ACM Management of Data*, vol. 1, no. 1, pp. 1–26, 2023.

[13] M. Kim, R. Sumbaly, and S. Shah, “Root cause detection in a service-oriented architecture,” in *Proc. ACM SIGMETRICS*, 2013, pp. 93–104.

[14] D. Wang et al., “Incremental causal graph learning for online root cause analysis,” in *Proc. ACM SIGKDD*, 2023, pp. 2269–2278.

[15] M. Li et al., “Causal inference-based root cause analysis for online service systems with intervention recognition,” in *Proc. ACM SIGKDD*, 2022, pp. 3230–3240.

[16] M. Grbovic et al., “Cold start approach for data-driven fault detection,” *IEEE Transactions on Industrial Informatics*, vol. 9, no. 4, pp. 2264–2273, 2013.

[17] Z. Hao et al., “Causal discovery on high dimensional data,” *Applied Intelligence*, vol. 42, no. 3, pp. 594–607, 2015.

[18] D. Bhattacharjya, D. Subramanian, and T. Gao, “Proximal graphical event models,” in *Proc. NeurIPS*, vol. 31, 2018, pp. 8136–8145.

[19] O. Shchur et al., “Neural temporal point processes: A review,” in *Proc. IJCAI*, 2021, pp. 4585–4593.

[20] J. Yu et al., “Abnormal event detection and localization via adversarial event prediction,” *IEEE Transactions on Neural Networks and Learning Systems*, vol. 33, no. 8, pp. 3572–3586, 2022.

[21] P. Chapfuwa et al., “Calibration and uncertainty in neural time-to-event modeling,” *IEEE Transactions on Neural Networks and Learning Systems*, vol. 34, no. 4, pp. 1666–1680, 2023.

[22] W. Wu et al., “Modeling event propagation via graph biased temporal point process,” *IEEE Transactions on Neural Networks and Learning Systems*, vol. 34, no. 4, pp. 1681–1691, 2023.

[23] H. Mei and J. Eisner, “The neural Hawkes process: A neurally self-modulating multivariate point process,” in *Proc. NeurIPS*, vol. 30, 2017, pp. 6757–6767.

[24] Y. Rubanova, R. T. Q. Chen, and D. K. Duvenaud, “Latent ordinary differential equations for irregularly-sampled time series,” in *Proc. NeurIPS*, 2019.

[25] Z. Yuan et al., “Autonomous-Jump-ODENet: Identifying continuous-time jump systems for cooling-system prediction,” *IEEE Transactions on Industrial Informatics*, vol. 19, no. 7, pp. 7894–7904, 2022.

[26] Z. Yuan et al., “ODE-RSSM: Learning stochastic recurrent state space model from irregularly sampled data,” in *Proc. AAAI*, vol. 37, 2023, pp. 11060–11068.

[27] P. Dutta et al., “Deep representation learning for prediction of temporal event sets in the continuous time domain,” in *Proc. ACML*, vol. 222, 2023, pp. 343–358.

[28] M. Ancona et al., “Gradient-based attribution methods,” in *Explainable AI: Interpreting, Explaining and Visualizing Deep Learning*, Springer, 2019, pp. 169–191.

[29] M. Sundararajan, A. Taly, and Q. Yan, “Axiomatic attribution for deep networks,” in *Proc. ICML*, vol. 70, 2017, pp. 3319–3328.

[30] Y. Chen, “Thinning algorithms for simulating point processes,” Florida State University, Tech. Rep., 2016.

[31] S. Xiao et al., “Learning time series associated event sequences with recurrent point process networks,” *IEEE Transactions on Neural Networks and Learning Systems*, vol. 30, no. 10, pp. 3124–3136, 2019.

[32] R. T. Q. Chen, “torchdiffeq,” 2018. [Online]. Available: https://github.com/rtqichen/torchdiffeq

[33] F. Lin et al., “Fast dimensional analysis for root cause investigation in a large-scale service environment,” *ACM SIGMETRICS Performance Evaluation Review*, vol. 48, no. 1, pp. 25–26, 2020.

[34] W. Hu et al., “Root cause identification of industrial alarm floods using word embedding and few-shot learning,” *IEEE Transactions on Industrial Informatics*, vol. 20, no. 2, pp. 1465–1475, 2023.

[35] Y. Chen et al., “Automatic root cause analysis via large language models for cloud incidents,” in *Proc. EuroSys*, 2024, pp. 674–688.

---

## Author Biographies

### Zhaolin Yuan

Zhaolin Yuan received the Ph.D. degree in computer science from the University of Science and Technology Beijing in 2023. He is currently an Associate Professor with the Institute of Artificial Intelligence, University of Science and Technology Beijing. His research interests include AIOps, intelligent manufacturing, AI in fashion, industrial system optimization, and model-based reinforcement learning.

### Long Ma

Long Ma received the Ph.D. degree in computer science from Delft University of Technology in 2022. He is currently a Research Engineer with Huawei Technologies Company Ltd. His interests include network modeling and prediction.

### Wenjia Wei

Wenjia Wei received the Ph.D. degree in information and communication engineering from the University of Science and Technology of China in 2020. He is currently a Research Engineer with Huawei Technologies Company Ltd. His interests include network modeling and simulation.

### Xia Zhu

Xia Zhu received the Ph.D. degree in computer science and engineering from Southeast University in 2014. He is currently a Technical Expert with Huawei Technologies Company Ltd. His interests include network modeling and optimization.

### Mingjie Sun

Mingjie Sun received the bachelor’s degree from Nanjing University of Aeronautics and Astronautics in 2016, the master’s degree from Xidian University in 2019, and the Ph.D. degree from the University of Liverpool in 2022. He is currently an Associate Professor with Soochow University. His research directions include computer vision, multimodal computing, and reinforcement learning.

### Duxin Chen

Duxin Chen received the B.S. degree in automatic control and the Ph.D. degree in control science and engineering from Huazhong University of Science and Technology in 2013 and 2018, respectively. He is currently an Associate Professor with Southeast University. His research interests include causal inference, prediction/generation, system identification for complex networks and system sciences, and AI-related theory and applications.

### Xiaojuan Ban

Xiaojuan Ban received the Ph.D. degree from the University of Science and Technology Beijing in 2003. She is currently a Ph.D. Supervisor with the University of Science and Technology Beijing and Managing Director of the Chinese Association for Artificial Intelligence. She has authored more than 300 articles.
