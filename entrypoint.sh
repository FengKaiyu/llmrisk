#!/bin/bash
set -e
# ===== 启动自检：triton/fla 冒烟，失败自动禁用 fla 回退参考实现 =====
python3 /app/preflight.py || true

# ===== 代码目录：挂载 /code 优先，镜像内置 /app 兜底 =====
APP_DIR=/app
[ -f /code/pipeline.py ] && APP_DIR=/code
echo "[entrypoint] 使用代码目录: $APP_DIR"

# ===== 模型路径（全部挂载进容器，不进镜像） =====
ARGS=()
[ -n "$MODEL_27B" ]    && ARGS+=(--e1-base "$MODEL_27B")
[ -n "$ADAPTER_27B" ]  && ARGS+=(--e1-adapter "$ADAPTER_27B")
[ -n "$MODEL_4B" ]     && ARGS+=(--e2-base "$MODEL_4B")
[ -n "$ADAPTER_4B" ]   && ARGS+=(--e2-adapter "$ADAPTER_4B")

# ===== 数据路径：CLI 显式参数优先，缺的用环境变量/默认值补齐 =====
has_data=0; has_out=0
for a in "$@"; do
  case "$a" in --data) has_data=1;; --out) has_out=1;; esac
done
EXTRA=()
[ "$has_data" -eq 0 ] && EXTRA+=(--data "${INPUT_FILE:-/data/input.jsonl}")
[ "$has_out" -eq 0 ] && EXTRA+=(--out "${OUTPUT_FILE:-/data/output.jsonl}")
exec python3 "$APP_DIR/pipeline.py" "$@" "${EXTRA[@]}" "${ARGS[@]}"
