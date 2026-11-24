#!/usr/bin/env python3
"""容器启动自检：triton/fla 可用性冒烟。失败则自动禁用 fla（重命名包目录），
transformers 退回参考实现——保证任何环境下容器都能跑起来。"""
import os, shutil, site, sys

def disable_fla(reason):
    print(f"[preflight] 禁用 fla（{reason}），回退参考实现（慢但稳）", flush=True)
    for sp in site.getsitepackages() + [site.getusersitepackages()]:
        fla = os.path.join(sp, "fla")
        if os.path.isdir(fla):
            try:
                shutil.move(fla, fla + "_disabled")
                print(f"[preflight] 已重命名 {fla}", flush=True)
            except OSError as e:
                print(f"[preflight] 重命名失败 {e}", flush=True)

try:
    import torch, triton, triton.language as tl

    @triton.jit
    def add_one(x_ptr, BLOCK: tl.constexpr):
        offs = tl.arange(0, BLOCK)
        tl.store(x_ptr + offs, tl.load(x_ptr + offs) + 1)

    x = torch.zeros(64, device="cuda")
    add_one[(1,)](x, BLOCK=64)
    assert x[0].item() == 1.0
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule  # noqa
    print("[preflight] triton + fla 可用，启用加速内核", flush=True)
except SystemExit:
    raise
except Exception as e:
    disable_fla(repr(e)[:200])
    sys.exit(0)  # 禁用不是错误
