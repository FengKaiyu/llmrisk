# 交付说明 · gbt-check v1.0（冻结版）

> 冻结时间：2026-11-25 ｜ 冻结配置：dual 全链（已验证）
> 本版之后只更新：模型 adapter + /code 挂载目录内容。镜像与底座参数不动。

## 一、冻结配置

| 组件 | 权重（挂载） | 说明 |
|---|---|---|
| E1 主判 | models/qwen3.8-27b-bf16（52G）+ models/27b-adapter（324M，v1.1） | md-KB v1.1 全文注入 system |
| E2 副判 | models/4b-finetune-v2-merged（7.6G） | JSON 卡两遍检索（b4） |
| 逻辑 | /code 挂载 = deploy/ 目录（pipeline.py + 双KB） | auto 模式自动选 dual |
| 镜像 | gbt-check:stable（fla+自愈预检） | 冻结，不随 adapter 变 |

已验证成绩（干净 gold 口径）：

| 集子 | 小类 | 大类 | 召回 | 误报 |
|---|---|---|---|---|
| 官方 46 | 93.5 | 95.7 | 100% | 0% |
| v3mix_fix 358 | 94.7 | 95.5 | 100% | 0.4% |
| A.1 专项 105 | 95.2 | 95.2 | 100% | 0% |
| 英文冒烟（46+50） | 97.8 / 96.0 | — | 100% | 0% |

## 二、运行

```bash
docker run --gpus all --rm \
  -v /data/models:/models:ro \
  -v /path/to/deploy:/code:ro \
  -v /data/io:/data \
  -e MODEL_27B=/models/qwen3.8-27b-bf16 \
  -e ADAPTER_27B=/models/27b-adapter \
  -e MODEL_4B=/models/4b-finetune-v2-merged \
  gbt-check:stable
# 默认读 /data/input.jsonl（字段 user_input 或 question+answer）→ 写 /data/output.jsonl
# 输出字段：{id, violation, major, minor, p, mode, e1, e2, rule}
```

降级行为：只挂 4B → 自动单引擎 e2；27B 加载失败 → [degrade] 日志 + 降级；fla 不可用 → preflight 自动禁用回退参考实现。

## 三、明天（2026-11-25 晚）的更新流程

1. **27B-v2v3 adapter**（租用服务器训练中，~600MB）：
   新建 /models/27b-v2v3-adapter/ 放入，挂载变量改 `ADAPTER_27B=/models/27b-v2v3-adapter`
2. **4B-v3**（4090 训练中，137MB adapter）：
   方案 A（推荐）：在 4090 合并后替换整个 4b-finetune 目录（7.6G）
   方案 B：只传 adapter，另需 Qwen3-4B 底座，挂 `MODEL_4B=/models/Qwen3-4B ADAPTER_4B=/models/4b-v3-adapter`
3. **代码更新**：替换 /code 目录内容（明日可能的改动：F5 仲裁方向、A5 阈值）
4. 镜像、27B 底座、其余一切不动

## 四、已知遗留（v1.1 待定，明日重测后决策）

- F5 仲裁（双引擎类分歧）现归 E1：v3mix 上 E2 胜可 +2.5，但官方风格反向——需新 E1 出炉后三集联测定夺
- A5 专线在无 A.5 gold 的集子上误触发 1 次（v3b-45）：拟加档位置信阈值；若 27B-v2v3 自解 A.5 则砍掉

## 五、镜像获取（三选一）

```bash
# A) U 盘自带（离线现场推荐）
gunzip -c gbt-check_v1.0.1.tar.gz | docker load

# B) GHCR 拉取（有网环境）
docker pull ghcr.io/fengkaiyu/llmrisk/gbt-check:stable

# C) 源码构建（GitHub 仓库打 v* tag 自动触发 Actions）
git clone git@github.com:FengKaiyu/llmrisk.git && cd llmrisk
docker build --platform linux/amd64 -t gbt-check:stable .
```

镜像 v1.0.1（commit 38d6045+fix）：含 fla 自愈预检，代码/权重全挂载。

## 六、4B 的 adapter 传输约定（2026-11-25 补）

4B 的 QLoRA adapter 直接挂 bf16 底座会有微小概率漂移（两遍卡检索对候选敏感，
saima46 上 89.1 vs 合并版 91.3）。因此：

- **传输格式 = adapter（137MB）**：明天发 4b-v3-adapter 目录
- **部署格式 = 合并版**：接收方跑一条命令本地合并（约 10 分钟，CPU 即可）：
  ```bash
  python3 merge_4b.py --base /models/Qwen3-4B --adapter /models/4b-v3-adapter \
      --out /models/4b-finetune-v3
  ```
- 合并后 MODEL_4B=/models/4b-finetune-v3（不挂 ADAPTER_4B）
- U 盘已备：models/Qwen3-4B（底座，7.6G，永不更新）+ models/4b-v2-adapter（当前版）
- 27B 无此问题（LoRA 本来就是 bf16 训的，adapter 直接挂载即等价）
