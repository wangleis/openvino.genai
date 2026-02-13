# Qwen2.5-VL 自动化生成迭代计划（Skill/Tools MD 驱动）

## 目标
在 `tools/` 目录内实现一条可重复流水线：
1. 从 Skill/Tools 的 MD 文档读取规范
2. 自动生成 Qwen2.5-VL 的模块骨架与 YAML pipeline
3. 保证“除 XML 模型加载外，其余逻辑与现有 `md_*` 模块实现一致”的收敛路径
4. 最终可把 Python reference 逐阶段替换为模块内实现

---

## Iteration 0（当前基线）
- 已有：
  - `tools/module_genai/qwen2_5_vl_codegen.py`
  - `tools/module_genai/README.md`
  - `.gnai/skills/model-to-module-pipeline-qwen2_5_vl.md`
- 能力：
  - 生成 YAML + skeleton + compare 脚本
- 缺口：
  - MD 内容尚未作为“机器可读规范”驱动生成

---

## Iteration 1：MD 机器可读规范化
### 交付
- 在 skill md 与 tools md 中加入 `CODEGEN_SPEC_JSON` 代码块
- 覆盖内容：
  - stage split（xml_gpu/local_logic）
  - 设备策略（hybrid）
  - parity 目标（与现有 md_* 对齐）
  - 生成路径与命名约定

### 验收
- 生成器可读取并解析 JSON block
- 解析失败时回退到默认策略

---

## Iteration 2：生成器改造为 MD 驱动
### 交付
- 新增参数：
  - `--skill-md`
  - `--tools-md`
- 生成器按优先级应用配置：
  - CLI > tools-md spec > skill-md spec > 内置默认
- 报告中输出最终生效配置与来源

### 验收
- 同一输入下可稳定复现（幂等）
- `analysis_report.json` 包含 `effective_spec`

---

## Iteration 3：按规范重生成（tools 下）
### 交付
- 输出目录：`tools/module_genai/generated_qwen2_5_vl/`
- 产物：
  - `config.generated.gpu.yaml`
  - `config.generated.cpu_compare.yaml`
  - `modules/md_generated_*.hpp/.cpp`
  - `python_ref/*`
  - `analysis_report.json`

### 验收
- 拓扑与端口和现有实现一致
- CPU compare YAML 可用于 reference 对比

---

## Iteration 4：Python reference -> 模块本地实现迁移计划
### 迁移顺序
1. `prompt_encoder`（先完全本地）
2. `embedding_merger`（完全本地）
3. `image_preprocessor`（GPU XML + 本地 glue）
4. `text_embedding`（GPU XML + 本地 glue）
5. `vision_encoder`（GPU XML + 本地 glue）
6. `llm_inference`（GPU XML + 本地 glue）

### 阶段门禁
- 功能等价：输出 shape/类型一致
- 数值一致：关键张量误差在阈值内
- 文本一致：`generated_text` 精确或归一化包含匹配

---

## Iteration 5：持续收敛
- 将 parity 测试纳入 CI（模块级 + pipeline 级）
- 对 spec 与代码生成进行版本化（`spec_version`）
- 增加失败场景策略（缺模型、输入不匹配、dtype 不兼容）
