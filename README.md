# 🚀 EndSide-Agent-LoRA: 面向端侧多域 Agent 的轻量级工具调用微调与高并发推理网关

[![Framework](https://img.shields.io/badge/Framework-Apple%20MLX-blue.svg)](https://github.com/ml-explore/mlx-examples)
[![Model](https://img.shields.io/badge/Base%20Model-Qwen2.5--3B--Instruct--4bit-green.svg)](https://huggingface.co/Qwen)
[![Serving](https://img.shields.io/badge/Serving-FastAPI%20%2B%20Prefix%20Cache-orange.svg)](#)
[![Experiment](https://img.shields.io/badge/Tracking-SwanLab-purple.svg)](https://swanlab.cn/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 📌 项目背景与痛点

在端侧或低资源环境下部署轻量级大模型（如 3B 参数量级）作为 Agent 调度引擎时，原生模型普遍存在以下三大核心痛点：

1. **Schema 键名漂移（Schema Drift）**：输出结构不稳定，频繁出现键名不匹配或包含冗余 Markdown 标记，导致下游解析器崩溃。
2. **复合意图与长链路处理能力弱**：难以在单轮推理中正确拆解**并行多工具调用（Parallel Function Calling）**，且缺乏多轮工具回填（Observation）后的信息整合能力。
3. **工具调用幻觉与推理延迟瓶颈**：面对缺失关键参数的指令时倾向于盲目编造参数而非主动反问；同时在长 System Prompt（16 个工具定义）下，全量预填充（Prefill）带来较高的首字延迟（TTFT）。

本项目基于 **Apple Silicon（Apple M6，24GB 统一内存）** 与 **MLX** 框架，构建了一套涵盖 **自动化数据工程 → QLoRA 微调与消融 → 8 维定量评测 → SwanLab 追踪 → Prefix KV Cache 异步推理网关** 的全栈级端侧 Agent 工程闭环。

> 📌 **关于数据**：本文所有指标均由本仓库脚本在本地实测产出，测量时间 2026-11-24。延迟类指标受机器负载与热状态影响，绝对值会有 ±10% 波动，但相对结论稳定。

---

## 🏗️ 系统全景架构

    ┌─────────────────────────────────────────────────────────────────────────┐
    │ 1. 数据工程流水线 (data_pipeline.py)                                     │
    │    • 覆盖 电商/办公/运维/通用 4 域共 16 个工具                            │
    │    • 单轮路由 + 并行多工具调用 + 多轮 Observation 总结 + 缺参反问/越界拒识 │
    │    • 混入通用指令回放 (Replay Buffer) 抑制灾难性遗忘                      │
    │    • 自动化 JSON Schema 门禁校验 + MD5 语义去重                          │
    └────────────────────────────────────┬────────────────────────────────────┘
                                         ▼
    ┌─────────────────────────────────────────────────────────────────────────┐
    │ 2. QLoRA 微调与实验追踪 (run_swanlab_exp.py)                            │
    │    • 4-bit 量化底座 + Prompt Loss Masking (仅对 Assistant 输出计算梯度)  │
    │    • MLX 原生 TrainingCallback 实时同步 SwanLab 看板与超参消融对比       │
    └────────────────────────────────────┬────────────────────────────────────┘
                                         ▼
    ┌─────────────────────────────────────────────────────────────────────────┐
    │ 3. AI Infra 异步推理网关与优化 (server.py & benchmark.py)               │
    │    • 4-bit 底座动态挂载高精 FP16 Adapter (规避二次量化精度损失)          │
    │    • Prefix KV Cache 固化系统提示词前缀，TTFT 首字延迟降低 43.3%         │
    │    • FastAPI 异步非阻塞调度 + 端到端沙箱工具执行闭环                      │
    └─────────────────────────────────────────────────────────────────────────┘

---

## 📊 核心实验结果与量化评测

### 1. 超参数消融实验（Rank=4 vs Rank=16）

两组控制变量实验统一使用 `mask_prompt: true`、`scale: 2.0`、学习率 `1.5e-4`、`iters: 300`、`data_pro` 数据集，唯一变量为 LoRA rank：

| 实验组 | 可训练参数量 (占比) | 首测 Val Loss (Step 30) | 最佳 Val Loss (Step 300) | 最终 Train Loss | 核心工程结论 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **QLoRA-Rank4** | 3.326M (**0.108%**) | 0.2188 | **0.00017** | 0.00015 | 结构化路由具低内在维度特性，小秩参数量极省且最终收敛更彻底 |
| **QLoRA-Rank16** | 13.304M (**0.431%**) | 0.1562 | 0.00025 | 0.00023 | 高秩在早期拟合明显更快（step30 val 0.156 vs 0.219），适合复杂参数抽取 |

> 💡 **曲线观察**：高秩的领先只存在于最初的 30 个 step（val 0.156 vs 0.219），到 step 60 就被低秩反超（0.019 vs 0.077），此后两者交替下行、收敛到同一量级；最终 step 300 时低秩略优。说明本任务的表征容量需求很低，Rank=4 已足够，Rank=16 多出的 4 倍参数没有换来收益。完整 Loss 曲线见本地 `swanlog/`（`swanlab watch ./swanlog` 打开，截图可放 `docs/`）。

---

### 2. 8 维业务评测矩阵 (`eval_pro.py`)

在严格隔离的测试集（`data_pro/test.jsonl`，23 条）与 **分布外口语化（Out-of-Distribution, OOD）** 高难度测试集上的端到端对比：

| 评测维度 | Base Model (微调前) | Pro-LoRA (微调后) | 绝对提升 / 工程意义 |
| :--- | :---: | :---: | :--- |
| **1. JSON 协议合法率** | 100.0% (11/11) | **100.0%** (11/11) | 零语法崩溃，100% 符合下游解析器要求 |
| **2. 16 域单工具路由准确率** | 0.0% (0/4) | **100.0%** (4/4) | **+100.0%**，彻底消除跨域 Schema 键名漂移 |
| **3. 并行多工具调用准确率** | 0.0% (0/2) | **100.0%** (2/2) | **+100.0%**，精准拆解复合意图为并行 `tool_calls` 列表 |
| **4. 缺参追问与越界拒识率** | 0.0% (0/5) | **60.0%** (3/5) | **+60.0%**，建立缺参反问 (`ask_clarification`) 与越界拦截能力 |
| **5. 多轮 Observation 总结率** | 0.0% (0/10) | **100.0%** (10/10) | **+100.0%**，完整掌握"派发工具 → 读取结果 → 自然语言总结" |
| **6. 通用能力防遗忘保持率** | 100.0% (2/2) | **100.0%** (2/2) | 通用指令回放有效抑制灾难性遗忘 |
| **7. 分布外口语化 (OOD) 通过率** | 16.7% (1/6) | **66.7%** (4/6) | **+50.0%**，具备真实语义泛化而非死记硬背模板 |
| **8. 平均端到端响应延迟** | 754.6 ms | 767.6 ms | 长上下文下延迟几乎零增加（仅 +13.0 ms） |

---

### 3. AI Infra 部署形态权衡（精度-延迟-存储三维分析）

针对端侧部署的权重融合策略进行了控制变量实测（基于 `data/test.jsonl` 的 12 条 4 工具路由样本，指标口径同 `eval.py`）：

| 部署架构形态 | 权重存储精度 | 工具路由准确率 | 端到端完全匹配率 | 平均端到端延迟 | Infra 架构洞察 |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **FP16 权重融合 (`--dequantize`)** | FP16 (~6.2 GB) | **91.7%** | **83.3%** | 1384.3 ms | 权重体积膨胀 4 倍触发 **访存带宽瓶颈 (Memory-Bandwidth Bound)** |
| **4-bit 无校准重量化融合** | 4-bit (~1.8 GB) | 66.7% | 33.3% | **459.2 ms** | RTN 二次量化噪声抹平微调特征增量，导致精度大幅坍塌 |
| **4-bit 底座 + 动态挂载 Adapter (最终方案)** | 4-bit + FP16 旁路 | **91.7%** | **83.3%** | **581.1 ms** | **兼顾 2.2GB 极低内存、亚秒级延迟与无损精度**，支持多技能热切换 |

---

### 4. 异步推理网关与 Prefix KV Cache 并发压测 (`benchmark.py`)

在 FastAPI 异步网关中预计算并固化 16 工具系统提示词的 KV Cache（166 个 system token），模拟线上单并发与高并发场景：

| 压测配置模式 | 首字延迟 TTFT (Avg / P95) | 端到端延迟 (P50 / P99) | 系统总吞吐 (QPS / Tok/s) | 高并发 JSON 合法率 |
| :--- | :---: | :---: | :---: | :---: |
| **关闭 Prefix Cache (并发=1)** | 147.6 ms / 162.3 ms | 754.8 ms / 1291.3 ms | 1.21 req/s \| 46.9 tok/s | 100.0% |
| **开启 Prefix KV Cache (并发=1)** | **83.7 ms / 101.3 ms** (**-43.3%**) | **697.0 ms / 1306.4 ms** | **1.29 req/s \| 50.1 tok/s** | **100.0%** |
| **开启 Prefix KV Cache (并发=4)** | 82.2 ms / 101.0 ms | 3080.1 ms / 3764.7 ms | 1.31 req/s \| 50.9 tok/s | **100.0%** |

* **核心优化收益**：开启 Prefix KV Cache 后，重复请求无需重算系统工具列表前缀，**平均首字延迟（TTFT）直降 43.3%（进入 85ms 内）**；4 并发下服务稳定无崩溃，JSON 协议合法率始终保持 **100.0%**。
* **诚实的解读**：并发=4 时端到端 P50 从 697ms 涨到 3080ms，吞吐几乎不变（1.29 → 1.31 req/s）。这说明 MLX 侧的 Metal 队列被 `inference_lock` 串行化保护，**当前实现是"并发安全"而非"并发加速"**——真正的并行需要多进程或多模型实例。

---

## 📂 项目目录结构

    lora-lab/
    ├── README.md                  # 项目技术报告与复现文档
    ├── LICENSE                    # MIT
    ├── requirements.txt           # 已验证的依赖版本快照
    ├── .gitignore                 # 排除权重、适配器与训练日志
    │
    ├── configs/                   # 训练配置
    │   ├── config_r4.yaml         # 初版 4 工具演示 · Rank=4
    │   ├── config_r16.yaml        # 初版 4 工具演示 · Rank=16
    │   ├── config_pro_r4.yaml     # 16 工具主实验 · Rank=4
    │   └── config_pro_lora.yaml   # 16 工具主实验 · Rank=16
    │
    ├── data_pipeline.py           # 16工具+并行+多轮+防遗忘 数据合成与去重流水线
    ├── run_swanlab_exp.py         # 集成 MLX 原生回调与 SwanLab 追踪的训练评测脚本
    ├── eval.py                    # 初版 4 工具路由评测（12 条样本）
    ├── eval_pro.py                # 8 维自动化定量评测矩阵（含 OOD 泛化测试）
    ├── server.py                  # FastAPI 异步推理网关（集成 Prefix KV Cache）
    ├── benchmark.py               # 异步并发压测脚本（统计 TTFT / TPOT / P99 / QPS）
    ├── webui_arena.py             # Gradio 双引擎 A/B 同屏对比与 Agent 轨迹可视化台
    │
    ├── data/                      # 初版 4 工具数据集（12/12/4 条，供 eval.py 使用）
    ├── data_pro/                  # 主数据集（382 train / 45 valid / 23 test）
    └── docs/                      # 实验看板与 A/B 对比截图目录

**被 `.gitignore` 排除、需本地生成的产物**（体积原因，不入库）：

| 路径 | 体积 | 生成方式 |
| :--- | ---: | :--- |
| `qwen-3b/` | 1.6 GB | `mlx_lm.convert` 量化下载 |
| `qwen-3b-agent-fused/` | 5.8 GB | FP16 权重融合（实验） |
| `qwen-3b-agent-fused-4bit/`、`qwen-3b-r16-fused-4bit/` | 各 1.6 GB | 4-bit 重量化融合（实验） |
| `adapters-r4/`、`adapters-r16/` | 25 MB / 102 MB | 初版训练产出 |
| `adapters-pro-r4/`、`adapters-pro-lora/` | 79 MB / 309 MB | 主实验训练产出（含中间 checkpoint） |
| `adapters-agent/` | 51 MB | 早期 Rank=8 试验 |
| `swanlog/` | 360 KB | SwanLab 本地看板日志 |

---

## 🛠️ 快速复现指南

### 1. 环境准备

    conda create -n lora-mlx python=3.11 -y
    conda activate lora-mlx
    pip install -r requirements.txt

需 macOS + Apple Silicon。已验证环境：`mlx 0.32.4`、`mlx-lm 0.32.2`、`swanlab 0.10.6`、Python 3.11。

### 2. 下载基座模型并构建数据集

    # 下载 Qwen2.5-3B-Instruct 并转换为 4-bit 量化底座
    mlx_lm.convert --hf-path Qwen/Qwen2.5-3B-Instruct --mlx-path ./qwen-3b -q

    # 运行数据工程流水线，生成 data_pro/{train,valid,test}.jsonl
    python data_pipeline.py

### 3. 启动微调训练与 SwanLab 监控

    python run_swanlab_exp.py \
      --name "QLoRA-Rank16" \
      --config configs/config_pro_lora.yaml \
      --adapter ./adapters-pro-lora

    # 另一组消融（Rank=4）
    python run_swanlab_exp.py \
      --name "QLoRA-Rank4" \
      --config configs/config_pro_r4.yaml \
      --adapter ./adapters-pro-r4

    # 浏览器查看本地看板
    swanlab watch ./swanlog

### 4. 运行 8 维定量评测与异步网关压测

    # 8 维评测矩阵（Base vs Pro-LoRA）
    python eval_pro.py

    # 启动 FastAPI 推理网关（终端 1）
    python server.py

    # Prefix Cache 并发压测（终端 2）
    python benchmark.py

### 5. A/B 对比演示台（可选）

    python webui_arena.py     # 打开 http://127.0.0.1:7860

---

## ⚠️ 已知局限

这些是实测暴露出来的问题，也是后续迭代方向：

1. **OOD 并行调用仍会失配**：6 条 OOD 中失败 2 条。其中"多意图并行"用例输出了 `"check_inventory": {...}` 而非规范的 `tool_calls` 列表，属典型 Schema 漂移，说明并行结构的泛化仍依赖模板句式。
2. **缺参反问对口语化指令不稳定**：「把线上服务重启一下」会编造 `service_name=online_service` 而非反问，训练集里缺少"服务名模糊"的缺参样本。
3. **通用回放比例远低于设计值**：流水线本意混入 15% 通用指令，但 MD5 去重把高度相似的通用样本合并后，`data_pro` 中实际只占 3.6%（16/450）。防遗忘指标 2/2 的样本量也偏小，不足以支撑强结论。
4. **并发是安全而非加速**：见第 4 节说明，`inference_lock` 串行化了 Metal 队列。
5. **测试集偏小**：23 条测试样本，单条波动即影响 4.3 个百分点，指标置信区间较宽。

---

## 📄 License

[MIT](LICENSE)
