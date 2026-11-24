#!/bin/bash
set -e
# ===== 启动自检：triton/fla 冒烟，失败自动禁用 fla 回退参考实现 =====
python3 /app/preflight.py || true

# ===== 代码目录：挂载 /code 优先，镜像内置 /app 兜底 =====
APP_DIR=/app
[ -f /code/pipeline.py ] && APP_DIR=/code

# ===== 模型路径（全部挂载进容器，不进镜像） =====
ARGS=()
[ -n "$MODEL_27B" ]    && ARGS+=(--e1-base "$MODEL_27B")
[ -n "$ADAPTER_27B" ]  && ARGS+=(--e1-adapter "$ADAPTER_27B")
[ -n "$MODEL_4B" ]     && ARGS+=(--e2-base "$MODEL_4B")
[ -n "$ADAPTER_4B" ]   && ARGS+=(--e2-adapter "$ADAPTER_4B")

# ===== 数据路径：CLI 参数 > 环境变量 > 默认 =====
if [ "$#" -eq 0 ]; then
  exec python3 "$APP_DIR/pipeline.py" \
    --data "${INPUT_FILE:-/data/input.jsonl}" \
    --out "${OUTPUT_FILE:-/data/output.jsonl}" \
    "${ARGS[@]}"
else
  exec python3 "$APP_DIR/pipeline.py" "$@" "${ARGS[@]}"
fi
