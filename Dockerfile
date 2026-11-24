# 合规检查 · 稳妥版镜像（逻辑+知识库，权重/代码运行时挂载）
# build: docker build --platform linux/amd64 -t gbt-check:stable .
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime
WORKDIR /app
# gcc 供 triton JIT 编译（fla 加速内核需要；容器启动自检失败会自动禁用 fla）
RUN apt-get update && apt-get install -y --no-install-recommends gcc \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY pipeline.py kb_retriever.py preflight.py merge_4b.py entrypoint.sh ./
COPY kb/ kb/
COPY gbt45654_taxonomy.json gbt45654_判定规则知识库.md ./
RUN chmod +x entrypoint.sh
ENTRYPOINT ["/app/entrypoint.sh"]
