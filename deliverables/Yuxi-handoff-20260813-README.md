# Yuxi 项目审核交接说明

本交接包用于交给独立负责人总结和审核当前项目。它保存了打包时工作区中的源码、实验脚本、设计文档、评估材料及必要的历史结果，不包含 Git 对象、本地依赖环境、缓存、日志或真实环境配置。

## 快照信息

- 打包日期：2026-08-13
- 当前分支：`sync-latest`
- 基准提交：`b7b7557dfc6f`
- 基准提交说明：`地图构建修复第六次`
- 打包前业务工作区状态：无未提交变更

压缩包基于实际工作区文件生成；如果打包时存在已修改但未提交的业务文件，也会保留其工作区版本，而不是只导出 Git 提交。

## 审核入口

建议依次阅读：

1. `README.md` 与 `ARCHITECTURE.md`：项目用途、部署方式和代码边界。
2. 根目录中的 `yuxi升级方案*.md`、`yuxi方法设计V5.md`：方法演进与实验设计。
3. `CHUNK_RETRIEVAL_EVALUATION.md`、`PRIM_DOCUMENT_CHUNK_GAP_ANALYSIS.md`、`RAG的评估.md`：现有评估及问题分析。
4. `backend/package/yuxi/agents/buildin/`：智能体和 RAG 方法实现。
5. `scripts/yuxi_batch_rag/`：批量运行、导出和评估脚本。
6. `scripts/yuxi_batch_rag/reference_file/` 与 `outputs/`：用于复核格式和实验结果的材料。

其中 `scripts/yuxi_batch_rag/reference_file/prim_rag_full_rag_records.json` 体积约 285 MB，属于历史会话审核证据，已特意保留，并非缓存。

## 已排除内容

- `.git/` 及其他版本控制内部数据；
- `.venv/`、`node_modules/` 等可重新安装的依赖；
- `__pycache__/`、`.pytest_cache/`、`.ruff_cache/`、`.uv-cache/` 等缓存；
- `dist/`、`build/`、覆盖率文件等可再生成产物；
- `saves/`、日志、临时文件和本地运行状态；
- `.env`、私钥、凭据和其他真实本地配置；模板与示例配置仍保留；
- IDE、Codex 等本地工具状态；
- `deliverables/` 中的外部交付物，避免压缩包递归包含自身。

压缩包内的 `PACKAGE_FILELIST.txt` 记录全部归档文件，外部 `SHA256SUMS.txt` 用于校验传输完整性。

## 运行说明

项目仍按仓库原有方式通过 Docker Compose 管理。审核人需要自行根据 `.env.template` 配置模型、数据库和知识库等外部服务；本包不携带远程环境凭据、模型权重、数据库卷或正在运行的容器状态。
