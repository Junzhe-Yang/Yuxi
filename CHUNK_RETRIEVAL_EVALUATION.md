# P3 块检索评估：PRIM-RAG 与 baseline

本报告使用 175 条完全相同的问题进行配对比较，包括 135 条“单文档”和
40 条“联合用药”。PRIM-RAG 另外拥有的 250 条 Beers 结果没有混入主比较。
整个评估是确定性的离线计算，不调用 LLM。

## 评估口径

1. 金标准只取答案末尾最后一个“依据清单”中去重后的手工块，不把正文中
   重复出现的引用再次加入分母。
2. 一个手工块是一个召回任务；它通过 `manual_to_auto.json` 中所有 `high`、
   `medium` 自动块别名命中。即使一个手工块映射到多个自动块，召回分母仍
   只加一。
3. 本轮不做人工作业级缩小。9 个被映射器标记为
   `requires_query_level_narrowing` 的宽手工块也保留其全部 `high`/`medium`
   别名，并在明细中显式标记。
4. 保留 Agent 自主停止前的全部成功搜索调用，不使用固定调用预算；每次调用
   只取原始顺序的 Top-5。`open` 调用不属于搜索召回，因而不计入。
5. “逐项判断”只给末尾清单中的目标分配 `core_required_chunks` 标签；末尾
   清单中其余目标为 `supporting_chunks`。正文核心引用若未进入末尾清单，
   仍按“只检测清单”规则排除。

## 指标设计

设问题 `q` 的手工目标集合为 `Gq`，目标 `g` 的自动块别名集合为 `A(g)`。
若任一实际搜索调用的 Top-5 与 `A(g)` 相交，则 `g` 被命中。

- `Macro target recall`：先计算每题命中目标数 / 目标总数，再对题目平均。
- `Micro target recall`：全体题目的命中目标总数 / 目标总数。
- `Hit rate`：至少命中一个目标的题目比例。
- `Complete coverage rate`：所有目标均被命中的题目比例。
- `Target-MRR@5`：对每个目标取它在任一次调用中的最佳局部名次 `r`，命中
  得 `1/r`，未命中得 0，再按目标和题目平均。
- `Target-nDCG@5`：同样使用最佳局部名次，目标得分为
  `1/log2(r+1)`，未命中为 0。它的理想值是所有目标都在某次调用的第 1 名。
- `Autonomous recall AUC`：每次实际调用后累计 target recall 的平均值；既
  衡量早期召回，也会反映自主停止前的无效尾部调用。
- `Productive call rate`：能新增至少一个尚未命中目标的调用数 / 搜索调用数。
- `Duplicate item rate`：`1 - 跨调用唯一自动块数 / Top-5 结果位置数`。
- `Annotated item rate`：Top-5 结果中可映射到清单目标的比例。由于清单不
  穷尽所有可能有用的块，该项不是严格 precision，只能作为诊断指标。

没有使用“把所有调用顺序拼成一个长榜单”的传统 MRR/nDCG。不同调用对应
不同子查询，把调用顺序当作全局 rank 会直接惩罚调用次数，并混淆“Agent
何时搜索”与“某次搜索内部的排序质量”。

## 175 条配对主结果

百分比指标均为题目宏平均，区间为 PRIM-RAG 减 baseline 的配对 bootstrap
95% 置信区间。

| 指标 | PRIM-RAG | baseline | 差值（百分点） | 95% CI |
|---|---:|---:|---:|---:|
| Macro target recall | 67.67% | 69.02% | -1.35 | [-5.12, 2.11] |
| Micro target recall | 61.80%（550/890） | 63.71%（567/890） | -1.91 | — |
| Hit rate | 98.29% | 99.43% | -1.14 | [-2.86, 0.00] |
| Complete coverage rate | 28.57% | 27.43% | +1.14 | [-5.14, 7.43] |
| Target-MRR@5 | 46.58% | 49.62% | -3.05 | [-6.19, 0.11] |
| Target-nDCG@5 | 51.82% | 54.45% | -2.63 | [-5.74, 0.38] |
| Any-hit MRR@5 | 91.51% | 95.38% | -3.87 | [-7.51, -0.41] |
| Autonomous recall AUC | 51.97% | 57.66% | -5.68 | [-9.18, -2.50] |
| Productive call rate | 32.48% | 40.78% | -8.29 | [-11.11, -5.50] |
| Annotated item rate（诊断） | 30.75% | 42.18% | -11.43 | [-13.63, -9.20] |

最终 target recall 的差异区间跨 0，因此不能据此断言两者的最终召回率存在
稳定显著差异；不过 PRIM-RAG 的早期命中和调用效率明显较低。它的完整覆盖率
比 baseline 高 1.14 个百分点，但差异同样不确定。

## 调用成本与检索轨迹

| 指标 | PRIM-RAG | baseline | PRIM-RAG 相对变化 |
|---|---:|---:|---:|
| 搜索调用总数 | 1,239 | 875 | +41.6% |
| 每题平均搜索调用 | 7.08 | 5.00 | +2.08 次 |
| 实际检查的 Top-5 结果位置 | 6,195 | 4,360 | +42.1% |
| 每题唯一自动块数 | 23.22 | 16.93 | +6.29 |
| Duplicate item rate | 33.84% | 31.65% | +2.19 个百分点 |
| 新增过金标目标的调用总数 | 369 | 341 | +28 次 |

PRIM-RAG 多发出了 364 次搜索，却只多得到 28 次“至少新增一个金标目标”的
调用，最终还少命中 17 个手工目标。这说明额外搜索主要扩展了检索范围，没有
有效转化为清单目标覆盖。

两套系统各返回过 1 个对齐阶段已隔离的重复损坏 OCR 块；二者都作为已知无效
块而不是正例处理。除此之外没有未知块 ID，且所有 Top-5 位置都成功解析出块 ID。

前两次调用所有 175 题都仍在检索，因此可直接比较：

| 实际调用位置 | PRIM-RAG 累计 recall | baseline 累计 recall |
|---|---:|---:|
| 第 1 次后 | 28.85% | 41.08% |
| 第 2 次后 | 40.72% | 52.41% |

第 3 次后分别为 49.12% 和 59.39%，但此时继续检索的题数已分别变为 174 和
171，后续逐调用曲线存在自主停止造成的 survivor effect；完整曲线在
`summary.json` 的 `autonomous_call_trajectories` 中保留。

## Core 与 supporting

| 范围 | 指标 | PRIM-RAG | baseline |
|---|---|---:|---:|
| Core（725 个目标） | Macro recall | 72.07% | 72.94% |
| Core | Micro recall | 66.21% | 67.86% |
| Core | Target-MRR@5 | 50.56% | 54.41% |
| Core | Target-nDCG@5 | 55.92% | 59.03% |
| Supporting（100 个适用题、165 个目标） | Macro recall | 46.33% | 47.92% |
| Supporting | Micro recall | 42.42% | 45.45% |
| Supporting | Target-MRR@5 | 24.92% | 27.16% |
| Supporting | Target-nDCG@5 | 30.19% | 32.29% |

两套系统对 supporting 块都明显弱于 core 块。PRIM-RAG 的额外调用没有改变
这一结构性短板。

## 题型与映射敏感性

| 题型 | 题数 | PRIM-RAG Macro recall | baseline Macro recall | 差值 |
|---|---:|---:|---:|---:|
| 单文档 | 135 | 71.42% | 71.91% | -0.50 |
| 联合用药 | 40 | 55.01% | 59.23% | -4.22 |

联合用药是更明显的薄弱点。其 PRIM-RAG target-MRR@5 为 34.78%，baseline
为 40.30%；PRIM-RAG 虽有 7.50% 的完整覆盖率（baseline 为 2.50%），但只有
40 题，完整覆盖差异的区间仍跨 0。

为了检查未经人工缩小的宽映射是否扭曲结论，额外做了不改变主口径的分层：

- 不含宽目标的 133 题：PRIM-RAG 69.61%，baseline 71.59%，差 -1.99 个
  百分点。
- 含宽目标的 42 题：PRIM-RAG 61.54%，baseline 60.86%，差 +0.68 个
  百分点。

因此 PRIM-RAG 总体略低并不是宽映射造成的；若只看没有该风险的题，差距反而
略有扩大。这个分层只是敏感性分析，主结果仍按约定保留全部自动映射目标。

## 结论与建议

1. 块检索主指标应使用“自主停止后的手工目标 target recall”，同时报告 macro、
   micro 和 complete coverage；不能固定调用次数，也不能把多个自动别名重复放大
   召回分母。
2. `Target-MRR@5`、`Target-nDCG@5` 和 `Autonomous recall AUC` 适合作为排序与
   过程指标；其中前两者只使用调用内局部名次。
3. 当前 PRIM-RAG 的最终召回与 baseline 大致同一量级，但没有体现出多调用应有
   的覆盖收益；它用约 42% 更多的 Top-5 检查量，最终命中目标反而少 17 个。
4. 优化重点不是限制 PRIM-RAG 的调用数，而是提高首轮查询的目标性、让后续查询
   面向尚未覆盖的 evidence obligation，并抑制重复/无新增目标的搜索。
5. `Annotated item rate` 不应升级为 precision 或 F1 主指标；未列入依据清单的块
   可能仍对回答有用，把它们一律判为 false positive 会高估错误率。

## 产物

- `evaluate_chunk_retrieval.py`：可复现的离线评估脚本。
- `test_evaluate_chunk_retrieval.py`：块级评估单元测试。
- `chunk_eval_output/summary.json`：完整定义、聚合、配对区间、分层和逐调用轨迹。
- `chunk_eval_output/summary.csv`：便于表格分析的系统/题型/范围汇总。
- `chunk_eval_output/*_paired_details.jsonl`：175 条逐题审计明细。

复现命令：

```powershell
python retrieval_eval/evaluate_chunk_retrieval.py
```
