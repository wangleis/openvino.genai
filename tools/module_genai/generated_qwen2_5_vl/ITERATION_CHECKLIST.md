# Auto Iteration Checklist

1. 更新 skill/tools md 中 CODEGEN_SPEC_JSON
2. 运行 codegen 并检查 analysis_report.effective_spec
3. 对比生成 YAML 与现有模块端口/参数一致性
4. 打开 use_python_reference，先打通 GPU XML 阶段
5. 逐阶段关闭 use_python_reference，迁移到本地模块实现