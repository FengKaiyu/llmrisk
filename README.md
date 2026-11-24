# 稳妥版部署

镜像 = 逻辑 + 双知识库；权重不进镜像，容器启动时挂载。

```bash
# 构建
docker build -t gbt-check:stable .

# 双引擎（两模型都挂载 → dual 模式自动生效）
docker run --gpus all --rm \
  -v /host/models:/models -v $PWD:/work \
  -e MODEL_27B=/models/qwen3.8-27b-bf16 \
  -e ADAPTER_27B=/models/gbt31_27b/adapter \
  -e MODEL_4B=/models/4b-finetune \
  gbt-check:stable --data /work/input.jsonl --out /work/output.jsonl

# 只挂 4B → 自动 e2 模式（4B+JSON卡两遍）
docker run --gpus all --rm -v /host/models:/models -v $PWD:/work \
  -e MODEL_4B=/models/4b-finetune \
  gbt-check:stable --data /work/input.jsonl --out /work/output.jsonl
```

降级行为：
- auto：按权重目录存在性选模式（dual > e1 > e2）
- dual 运行中 27B 加载失败 → 自动降级 e2（4B 单引擎），日志 `[degrade]`
- 输出含 mode 字段，事后可知实际用了哪条链路

## 镜像冻结工作流（推荐）

镜像建一次后不再更新，代码和权重全部挂载覆盖：

```bash
docker run --gpus all --rm \
  -v /data/models:/models:ro \
  -v /path/to/deploy:/code:ro \
  -v /data/io:/data \
  -e MODEL_27B=/models/qwen3.8-27b-bf16 \
  -e ADAPTER_27B=/models/27b-adapter \
  -e MODEL_4B=/models/4b-finetune \
  gbt-check:stable
```

- 挂载 `/code` 后用挂载的 pipeline.py/知识库，不挂则用镜像内置副本（兜底）
- 改代码/改 KB/换 adapter = 改挂载目录内容，**零重建**
- 仅两种情况需要重建镜像：新增 pip 依赖；模型架构变化要求更高版本 transformers
