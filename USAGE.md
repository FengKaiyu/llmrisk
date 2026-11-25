# 参数手册 · gbt-check pipeline

## 一、两层控制结构

```
docker run ... image [CLI参数]        ← entrypoint 透传给 pipeline.py
  环境变量（模型路径/数据路径）        ← entrypoint 转成 CLI 参数
  挂载 /code                          ← 代码热更新（优先于镜像内置）
```

## 二、环境变量（entrypoint 层）

| 变量 | 含义 | 不设的后果 |
|---|---|---|
| MODEL_27B | 27B 底座目录 | E1 不启用 |
| ADAPTER_27B | 27B LoRA adapter 目录 | E1 零样本+mdKB（推荐，实测不弱于带 adapter）|
| MODEL_4B | 4B 模型目录（合并版）| E2 不启用 |
| ADAPTER_4B | 4B adapter（需配合 4B 底座）| 不挂则用合并版 |
| INPUT_FILE | 输入 jsonl | 默认 /data/input.jsonl |
| OUTPUT_FILE | 输出 jsonl | 默认 /data/output.jsonl |

模式自动选择：两个 MODEL 都在 → dual；只有 27B → e1；只有 4B → e2。
运行中加载失败自动降级（日志 [degrade]）。adapter 路径写错 → 硬报错拒绝跑（防静默垃圾输出）。

## 三、CLI 参数（pipeline 层，可追加在 docker run 末尾）

| 参数 | 默认 | 含义与调法 |
|---|---|---|
| --mode | auto | 强制指定 dual/e1/e2（auto=按挂载自动选）|
| --trust-e2 | 0.85 | **F4 规则阈值**：E1 判 SAFE、E2 判其他风险时，E2 置信 ≥ 此值才听 E2。调低=更信 4B（多召回风险），调高=更信 27B |
| --tilt | 0.55 | **F6 召回倾斜**：最终 SAFE 且置信 < 此值 → 翻成 runner-up 风险类。**建议传 --tilt 0 关闭**（实测：召回从未低于 100%，倾斜只制造误报；v1.0.4 起默认改 0）|
| --budget | 4800 | E2 卡检索的 token 预算（每条注入多少判定规则）。不动 |
| --bs-budget | 36000 | 每批 token 上限。**小显存调小**（8G 卡用 12000）|
| --base-bs | 6 | 每批最大条数。小显存调小（8G 卡用 2）|

## 四、融合规则全景（输出里的 rule 字段）

| 规则 | 触发条件 | 决定 | 实测表现 |
|---|---|---|---|
| F1一致 | 双引擎同判 | 直出 | ~95% 的条目走这条 |
| F2 | E1=SAFE，E2=A.5a/b | 听 E2 | 4B 对 A.5b 弱，基本不触发 |
| F3 | E1=风险，E2=SAFE | 听 E1 | 27B 误报 0-0.4%，安全 |
| F4 | E1=SAFE，E2=其他风险 | 看 --trust-e2 | 低置信听 E1，高置信听 E2 |
| F5 | 双风险但类不同 | 听 E1 | v3mix 上 E2 更准（争议中，暂不动）|
| A5专线 | 最终 SAFE 的条目 | 27B 域门+质量三档合取 → A.5 覆盖 | 防漏 A.5b 的安全网 |
| F6倾斜 | final=SAFE 且 p<tilt | 翻成 runner-up | **负收益，关闭** |

## 五、典型用法

```bash
# ① 比赛默认（推荐）：零样本 dual，无 adapter
docker run --rm --gpus all \
  -v /data/models:/models:ro -v $DEPLOY:/code:ro -v /data/io:/data \
  -e MODEL_27B=/models/qwen3.8-27b-bf16 \
  -e MODEL_4B=/models/4b-finetune-v2-merged \
  gbt-check:stable --tilt 0

# ② 只挂 27B（自动 e1 单引擎 + A5 专线）
#    只传 MODEL_27B 即可

# ③ 4B 用 base+adapter 形态
#    -e MODEL_4B=/models/Qwen3-4B -e ADAPTER_4B=/models/4b-v3-adapter
#    （注意：QLoRA 直挂 bf16 有轻微漂移，正式用建议先 merge_4b.py 合并）

# ④ 小显存冒烟（本地 8G 卡验证过）
#    末尾加 --base-bs 2 --bs-budget 12000

# ⑤ 指定输入输出文件
#    -e INPUT_FILE=/data/test.jsonl -e OUTPUT_FILE=/data/pred.jsonl
```

## 六、输出字段

{id, violation(bool), major(大类/SAFE), minor(小类/空), p(置信), mode(实际运行模式),
 e1/e2(两引擎判定+置信), rule(走了哪条融合规则), gold_code(输入带 gold 时回显)}

## 七、两个知识库的控制

- md-KB（E1）：`gbt45654_判定规则知识库.md` 全文注入 system prompt，改文件即生效
- JSON 卡（E2）：`kb/` 目录，按 E2 第一遍的 top-3 候选检索注入，--budget 控制量
- 两者都在 /code 挂载里，**改完即生效，无需重建镜像**
