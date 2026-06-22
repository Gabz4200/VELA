import functools
import importlib
import math
import os
import platform
from enum import Enum as _EnumType

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.utils._pytree as _pytree
from pytorch_lightning.strategies import DeepSpeedStrategy
from pytorch_lightning.utilities import rank_zero_info, rank_zero_warn
from torch.nn import functional as F

# Patch pytree.register_constant to skip Enum subclasses (native in torch.compile since PyTorch 2.14+)
# Prevents upstream torchao deprecation triggered by deepspeed import chain.
_orig_reg = _pytree.register_constant
_pytree.register_constant = lambda cls: (
    cls if isinstance(cls, type) and issubclass(cls, _EnumType)
    else _orig_reg(cls)
)

if importlib.util.find_spec("deepspeed"):
    import deepspeed
    from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam

from .dataset import IGNORE_INDEX, IMAGE_TOKEN_INDEX, STOP_TOKEN_INDEX
from .siglino import load_siglino_from_hub
from .utils import compress_parameter_names


def __nop(ob):
    return ob


MyModule = nn.Module
MyFunction = __nop
if os.environ.get("RWKV_JIT_ON", "0") == "1":
    MyModule = torch.jit.ScriptModule
    MyFunction = torch.jit.script_method


HEAD_SIZE = int(os.environ.get("RWKV_HEAD_SIZE_A", "64"))
CHUNK_LEN = 16


def wind_backstepping_ref_forward(w, q, k, v, z, b):
    B, T, H, C = w.shape
    device = w.device
    dtype = w.dtype

    w_f = w.float()
    q_f = q.float()
    k_f = k.float()
    v_f = v.float()
    z_f = z.float()
    b_f = b.float()

    state = torch.zeros(B, H, C, C, device=device)

    y = torch.empty(B, T, H, C, dtype=dtype, device=device)
    sa = torch.empty(B, T, H, C, dtype=torch.float32, device=device)
    s_chunk = torch.empty(B, H, T // CHUNK_LEN, C, C, dtype=torch.float32, device=device)

    for t in range(T):
        q_t = q_f[:, t, :, :]
        w_t = torch.exp(-torch.exp(w_f[:, t, :, :]))
        k_t = k_f[:, t, :, :]
        v_t = v_f[:, t, :, :]
        z_t = z_f[:, t, :, :]
        b_t = b_f[:, t, :, :]

        sa_t = torch.matmul(z_t.unsqueeze(-2), state).squeeze(-2)
        sa[:, t, :, :] = sa_t

        state = (
            w_t.unsqueeze(-1) * state
            + torch.matmul(b_t.unsqueeze(-1), sa_t.unsqueeze(-2))
            + torch.matmul(k_t.unsqueeze(-1), v_t.unsqueeze(-2))
        )

        y_t = torch.matmul(q_t.unsqueeze(-2), state).squeeze(-2)
        y[:, t, :, :] = y_t.to(dtype)

        if (t + 1) % CHUNK_LEN == 0:
            s_chunk[:, :, (t + 1) // CHUNK_LEN - 1, :, :] = state

    return y, s_chunk, sa


def wind_backstepping_ref_backward(w, q, k, v, z, b, dy, s_chunk, sa):
    B, T, H, C = w.shape
    device = w.device
    dtype = w.dtype

    w_f = w.float()
    q_f = q.float()
    k_f = k.float()
    v_f = v.float()
    z_f = z.float()
    b_f = b.float()
    dy_f = dy.float()

    dw = torch.empty_like(w)
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    dz = torch.empty_like(z)
    db = torch.empty_like(b)

    dstate = torch.zeros(B, H, C, C, device=device)
    dstateT = torch.zeros(B, H, C, C, device=device)
    stateT = torch.zeros(B, H, C, C, device=device)

    for t in range(T - 1, -1, -1):
        q_t = q_f[:, t, :, :]
        w_val = w_f[:, t, :, :]
        w_fac_t = -torch.exp(w_val)
        w_t = torch.exp(w_fac_t)
        k_t = k_f[:, t, :, :]
        z_t = z_f[:, t, :, :]
        b_t = b_f[:, t, :, :]
        v_t = v_f[:, t, :, :]
        dy_t = dy_f[:, t, :, :]
        sa_t = sa[:, t, :, :]

        if (t + 1) % CHUNK_LEN == 0:
            stateT = s_chunk[:, :, (t + 1) // CHUNK_LEN - 1, :, :]

        dq[:, t, :, :] = torch.matmul(stateT, dy_t.unsqueeze(-1)).squeeze(-1).to(dtype)

        stateT = (
            stateT
            - torch.matmul(k_t.unsqueeze(-1), v_t.unsqueeze(-2))
            - torch.matmul(b_t.unsqueeze(-1), sa_t.unsqueeze(-2))
        ) / w_t.unsqueeze(-1)

        dstate = dstate + torch.matmul(dy_t.unsqueeze(-1), q_t.unsqueeze(-2))
        dstateT = dstateT + torch.matmul(q_t.unsqueeze(-1), dy_t.unsqueeze(-2))

        dw_val = (dstateT * stateT).sum(dim=-1)
        dk_val = (dstateT * v_t.unsqueeze(-2)).sum(dim=-1)
        dv_val = (dstate * k_t.unsqueeze(-2)).sum(dim=-1)
        dSb_t = (dstate * b_t.unsqueeze(-2)).sum(dim=-1)
        db_val = (dstateT * sa_t.unsqueeze(-2)).sum(dim=-1)

        dw[:, t, :, :] = (dw_val * w_t * w_fac_t).to(dtype)
        dk[:, t, :, :] = dk_val.to(dtype)
        dv[:, t, :, :] = dv_val.to(dtype)
        db[:, t, :, :] = db_val.to(dtype)

        dz[:, t, :, :] = torch.matmul(stateT, dSb_t.unsqueeze(-1)).squeeze(-1).to(dtype)

        dstate = dstate * w_t.unsqueeze(-2) + torch.matmul(dSb_t.unsqueeze(-1), z_t.unsqueeze(-2))
        dstateT = dstateT * w_t.unsqueeze(-1) + torch.matmul(z_t.unsqueeze(-1), dSb_t.unsqueeze(-2))

    return dw, dq, dk, dv, dz, db


@functools.cache
def detect_gpu_backend() -> tuple[str, str]:
    """Probe torch to determine the active GPU backend + arch identifier.

    Returns ``(backend, arch)``:
        - ``("cuda", "smXY")`` for NVIDIA (cubin is sm-locked).
        - ``("rocm", "gfxNNN")`` for AMD (HSACO is gfx-locked; we
          strip the ``:sramecc+:xnack-`` suffix that ``gcnArchName``
          appends because those modes don't affect the cached binary
          for our kernels).
        - ``("metal", "macosNN")`` for Apple. Metal AIR is largely
          forward-compatible across chip generations within a macOS
          major; we key on the macOS major version because system
          updates have invalidated cached AIR in the past.

    Cached for the process — switching GPUs mid-process isn't
    supported anyway. Raises ``RuntimeError`` if no GPU backend is
    available.
    """
    import torch  # noqa: PLC0415  — torch is heavy; defer until needed

    if torch.cuda.is_available():
        if getattr(torch.version, "hip", None) is not None:
            raw = torch.cuda.get_device_properties(0).gcnArchName
            arch = raw.split(":")[0]  # drop sramecc/xnack mode suffix
            if not arch.startswith("gfx") or arch == "gfx":
                # Empty or malformed gfxNNN → silently keying on "gfx"
                # would risk sharing a cache slot across truly-different
                # ROCm targets. Fail loudly.
                raise RuntimeError(
                    f"ROCm device reports an unrecognised gcnArchName "
                    f"({raw!r}); refusing to share a cache slot across "
                    f"unknown AMD GPU targets."
                )
            return ("rocm", arch)
        major, minor = torch.cuda.get_device_capability(0)
        return ("cuda", f"sm{major}{minor}")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        macos_major = (platform.mac_ver()[0] or "").split(".")[0]
        if not macos_major:
            raise RuntimeError("Could not detect macOS major version via `platform.mac_ver()`.")
        return ("metal", f"macos{macos_major}")
    raise RuntimeError("no GPU backend available — install torch with cuda / rocm / mps support")

HAS_CPP_EXT = False
try:
    from torch.utils.cpp_extension import load

    # Copy exact same pattern causal-conv1d-mojo uses to decide between CUDA or CPU
    use_gpu = False
    try:
        backend, arch = detect_gpu_backend()
        if backend in ("cuda", "rocm"):
            use_gpu = True
    except RuntimeError:
        use_gpu = False

    model_dir = os.path.dirname(__file__)
    if use_gpu:
        flags = [
            "-res-usage",
            f"-D_C_={HEAD_SIZE}",
            f"-D_CHUNK_LEN_={CHUNK_LEN}",
            "--use_fast_math",
            "-O3",
            "-Xptxas -O3",
            "--extra-device-vectorization",
        ]
        load(
            name="wind_backstepping",
            sources=[
                os.path.join(model_dir, "..", "cuda", "wkv7_cuda.cu"),
                os.path.join(model_dir, "..", "cuda", "wkv7_op.cpp"),
            ],
            is_python_module=False,
            verbose=True,
            extra_cflags=["-DUSE_CUDA"],
            extra_cuda_cflags=flags + ["-DUSE_CUDA"],
        )
    else:
        load(
            name="wind_backstepping",
            sources=[os.path.join(model_dir, "..", "cuda", "wkv7_op.cpp")],
            is_python_module=False,
            verbose=True,
            extra_cflags=["-DNO_CUDA"],
        )
    HAS_CPP_EXT = True
except Exception as e:
    print(
        f"Failed to load or compile C++/CUDA wind_backstepping extension: {e}. Falling back to pure PyTorch reference implementation."
    )
    HAS_CPP_EXT = False


class WindBackstepping(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w, q, k, v, z, b):
        B, T, H, C = w.shape
        assert T % CHUNK_LEN == 0
        assert all(i.dtype == torch.bfloat16 for i in [w, q, k, v, z, b])
        assert all(i.is_contiguous() for i in [w, q, k, v, z, b])

        if HAS_CPP_EXT:
            y = torch.empty_like(v)
            s = torch.empty(B, H, T // CHUNK_LEN, C, C, dtype=torch.float32, device=w.device)
            sa = torch.empty(B, T, H, C, dtype=torch.float32, device=w.device)
            torch.ops.wind_backstepping.forward(w, q, k, v, z, b, y, s, sa)
            ctx.save_for_backward(w, q, k, v, z, b, s, sa)
            return y
        else:
            y, s, sa = wind_backstepping_ref_forward(w, q, k, v, z, b)
            ctx.save_for_backward(w, q, k, v, z, b, s, sa)
            return y

    @staticmethod
    def backward(ctx, *grad_outputs):
        dy = grad_outputs[0].contiguous()
        w, q, k, v, z, b, s, sa = ctx.saved_tensors

        if HAS_CPP_EXT:
            dw, dq, dk, dv, dz, db = [torch.empty_like(x) for x in [w, q, k, v, z, b]]
            torch.ops.wind_backstepping.backward(
                w, q, k, v, z, b, dy, s, sa, dw, dq, dk, dv, dz, db
            )
            return dw, dq, dk, dv, dz, db
        else:
            return wind_backstepping_ref_backward(w, q, k, v, z, b, dy, s, sa)


def RUN_CUDA_RWKV7g(q, w, k, v, a, b):
    B, T, HC = q.shape
    q, w, k, v, a, b = [i.view(B, T, HC // 64, 64) for i in [q, w, k, v, a, b]]
    return WindBackstepping.apply(w, q, k, v, a, b).view(B, T, HC)


def rmsnorm(x, eps=1e-6):
    orig_dtype = x.dtype
    x_f = x.float()
    norm = x_f * torch.rsqrt(x_f.pow(2).mean(-1, keepdim=True) + eps)
    return norm.to(dtype=orig_dtype)


def Sinkhorn_Knopp(X, tmax=20, eps=1e-12):
    orig_dtype = X.dtype
    X_f = X.float()
    X_f = X_f - X_f.max(dim=-1, keepdim=True)[0]
    M = torch.exp(X_f)
    for _ in range(tmax):
        M = M / (M.sum(dim=-1, keepdim=True) + eps)
        M = M / (M.sum(dim=-2, keepdim=True) + eps)
    return M.to(dtype=orig_dtype)



class RWKV_Tmix_x070(nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id

        self.head_size = args.head_size_a
        self.n_head = args.dim_att // self.head_size
        assert args.dim_att % self.n_head == 0
        H = self.n_head
        N = self.head_size
        C = args.n_embd

        with torch.no_grad():
            ratio_0_to_1 = layer_id / (args.n_layer - 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, C)
            for i in range(C):
                ddd[0, 0, i] = i / C

            self.x_r = nn.Parameter(1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0))
            self.x_w = nn.Parameter(1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0))
            self.x_k = nn.Parameter(
                1.0 - (torch.pow(ddd, 0.9 * ratio_1_to_almost0) + 0.4 * ratio_0_to_1)
            )
            self.x_v = nn.Parameter(
                1.0 - (torch.pow(ddd, 0.4 * ratio_1_to_almost0) + 0.6 * ratio_0_to_1)
            )
            self.x_a = nn.Parameter(1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0))
            self.x_g = nn.Parameter(1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0))

            def ortho_init(x, scale):
                with torch.no_grad():
                    shape = x.shape
                    if len(shape) == 2:
                        gain = math.sqrt(shape[0] / shape[1]) if shape[0] > shape[1] else 1
                        nn.init.orthogonal_(x, gain=gain * scale)
                    elif len(shape) == 3:
                        gain = math.sqrt(shape[1] / shape[2]) if shape[1] > shape[2] else 1
                        for i in range(shape[0]):
                            nn.init.orthogonal_(x[i], gain=gain * scale)
                    else:
                        assert False
                    return x

            D_DECAY_LORA = max(32, int(round((1.8 * (C**0.5)) / 32) * 32))  # suggestion
            self.w1 = nn.Parameter(torch.zeros(C, D_DECAY_LORA))
            self.w2 = nn.Parameter(ortho_init(torch.zeros(D_DECAY_LORA, C), 0.1))
            decay_speed = torch.ones(C)
            for n in range(C):
                decay_speed[n] = -7 + 5 * (n / (C - 1)) ** (0.85 + 1.0 * ratio_0_to_1**0.5)
            self.w0 = nn.Parameter(
                decay_speed.reshape(1, 1, C) + 0.5
            )  # !!! 0.5 comes from F.softplus !!!

            D_AAA_LORA = max(32, int(round((1.8 * (C**0.5)) / 32) * 32))  # suggestion
            self.a1 = nn.Parameter(torch.zeros(C, D_AAA_LORA))
            self.a2 = nn.Parameter(ortho_init(torch.zeros(D_AAA_LORA, C), 0.1))
            self.a0 = nn.Parameter(torch.zeros(1, 1, C))

            D_MV_LORA = max(32, int(round((1.3 * (C**0.5)) / 32) * 32))  # suggestion
            if self.layer_id != 0:  # not needed for the first layer
                self.v1 = nn.Parameter(torch.zeros(C, D_MV_LORA))
                self.v2 = nn.Parameter(ortho_init(torch.zeros(D_MV_LORA, C), 0.1))
                self.v0 = nn.Parameter(torch.zeros(1, 1, C) + 1.0)

            D_GATE_LORA = max(32, int(round((0.6 * (C**0.8)) / 32) * 32))  # suggestion
            # Note: for some data, you can reduce D_GATE_LORA or even remove this gate
            self.g1 = nn.Parameter(torch.zeros(C, D_GATE_LORA))
            self.g2 = nn.Parameter(ortho_init(torch.zeros(D_GATE_LORA, C), 0.1))

            self.k_k = nn.Parameter(torch.ones(1, 1, C) * 0.85)
            self.k_a = nn.Parameter(torch.ones(1, 1, C))
            self.r_k = nn.Parameter(torch.zeros(H, N))

            self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
            self.receptance = nn.Linear(C, C, bias=False)
            self.key = nn.Linear(C, C, bias=False)
            self.value = nn.Linear(C, C, bias=False)
            self.output = nn.Linear(C, C, bias=False)
            self.ln_x = nn.GroupNorm(
                H, C, eps=(1e-5) * (args.head_size_divisor**2)
            )  # !!! notice eps value !!!

            self.receptance.weight.data.uniform_(-0.5 / (C**0.5), 0.5 / (C**0.5))
            self.key.weight.data.uniform_(-0.05 / (C**0.5), 0.05 / (C**0.5))
            self.value.weight.data.uniform_(-0.5 / (C**0.5), 0.5 / (C**0.5))
            self.output.weight.data.zero_()

    def forward(self, x, v_first):
        B, T, C = x.size()
        H = self.n_head
        xx = self.time_shift(x) - x

        xr = x + xx * self.x_r
        xw = x + xx * self.x_w
        xk = x + xx * self.x_k
        xv = x + xx * self.x_v
        xa = x + xx * self.x_a
        xg = x + xx * self.x_g

        r = self.receptance(xr)
        w = (
            -F.softplus(-(self.w0 + torch.tanh(xw @ self.w1) @ self.w2)) - 0.5
        )  # soft-clamp to (-inf, -0.5)
        k = self.key(xk)
        v = self.value(xv)
        if self.layer_id == 0:
            v_first = v  # store the v of the first layer
        else:
            v = v + (v_first - v) * torch.sigmoid(
                self.v0 + (xv @ self.v1) @ self.v2
            )  # add value residual
        a = torch.sigmoid(self.a0 + (xa @ self.a1) @ self.a2)  # a is "in-context learning rate"
        g = torch.sigmoid(xg @ self.g1) @ self.g2

        kk = k * self.k_k
        kk = F.normalize(kk.view(B, T, H, -1), dim=-1, p=2.0).view(B, T, C)
        k = k * (1 + (a - 1) * self.k_a)

        x = RUN_CUDA_RWKV7g(r, w, k, v, -kk, kk * a)
        x = self.ln_x(x.view(B * T, C)).view(B, T, C)

        x = x + (
            (r.view(B, T, H, -1) * k.view(B, T, H, -1) * self.r_k).sum(dim=-1, keepdim=True)
            * v.view(B, T, H, -1)
        ).view(B, T, C)
        pre_out = x * g
        x = self.output(pre_out)
        return x, v_first, pre_out


class RWKV_CMix_x070(nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        with torch.no_grad():
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, args.n_embd)
            for i in range(args.n_embd):
                ddd[0, 0, i] = i / args.n_embd
            self.x_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0**4))

        self.key = nn.Linear(args.n_embd, args.n_embd * 4, bias=False)
        self.value = nn.Linear(args.n_embd * 4, args.n_embd, bias=False)

        self.key.weight.data.uniform_(-0.5 / (args.n_embd**0.5), 0.5 / (args.n_embd**0.5))
        self.value.weight.data.zero_()

    def forward(self, x):
        xx = self.time_shift(x) - x

        k = x + xx * self.x_k
        k = torch.relu(self.key(k)) ** 2

        return self.value(k)



def block_attn_res(V_blocks, partial_block, proj, norm):
    V = torch.cat([V_blocks, partial_block.unsqueeze(0)], dim=0)
    K = norm(V)
    logits = torch.einsum("d, n b t d -> n b t", proj.weight.squeeze(0), K)
    h = torch.einsum("n b t, n b t d -> b t d", logits.softmax(0), V)
    return h


class Block(nn.Module):
    def __init__(self, args, layer_id, is_vtc=False):
        super().__init__()
        self.args = args
        self.layer_id = layer_id

        if self.layer_id == 0:
            self.ln0 = nn.LayerNorm(args.n_embd)  # only used in block 0, should be fused with emb
        self.ln1 = nn.LayerNorm(args.n_embd)
        self.ln2 = nn.LayerNorm(args.n_embd)

        self.att = RWKV_Tmix_x070(args, layer_id)
        self.ffn = RWKV_CMix_x070(args, layer_id)

        if not hasattr(args, "n_attnres_blocks"):
            args.n_attnres_blocks = 8
        n_layers = args.n_vtc_layer if (is_vtc and hasattr(args, "n_vtc_layer")) else args.n_layer
        self.layers_per_block = max(1, n_layers // args.n_attnres_blocks)

        self.attn_res_proj = nn.Linear(args.n_embd, 1, bias=False)
        self.attn_res_norm = nn.LayerNorm(args.n_embd)
        self.mlp_res_proj = nn.Linear(args.n_embd, 1, bias=False)
        self.mlp_res_norm = nn.LayerNorm(args.n_embd)

    def forward(self, V_blocks, partial_block, v_first):
        if self.layer_id == 0:
            partial_block = self.ln0(partial_block)

        h = block_attn_res(V_blocks, partial_block, self.attn_res_proj, self.attn_res_norm)

        if self.layer_id % self.layers_per_block == 0:
            V_blocks = torch.cat([V_blocks, partial_block.unsqueeze(0)], dim=0)
            partial_block = None

        xx, v_first, _ = self.att(self.ln1(h), v_first)
        partial_block = partial_block + xx if partial_block is not None else xx

        h = block_attn_res(V_blocks, partial_block, self.mlp_res_proj, self.mlp_res_norm)
        xx_mlp = self.ffn(self.ln2(h))
        partial_block = partial_block + xx_mlp

        return V_blocks, partial_block, v_first


class MHCBlock(Block):
    def __init__(self, args, layer_id):
        super().__init__(args, layer_id)
        if hasattr(self, "ffn"):
            del self.ffn
        C = args.n_embd
        self.experts = nn.ModuleList([RWKV_CMix_x070(args, layer_id) for _ in range(4)])

        self.phi_pre_att = nn.Parameter(torch.empty(C, 4))
        self.phi_post_att = nn.Parameter(torch.empty(C, 4))
        self.phi_res_att = nn.Parameter(torch.empty(C, 16))
        self.b_pre_att = nn.Parameter(torch.empty(4))
        self.b_post_att = nn.Parameter(torch.empty(4))
        self.b_res_att = nn.Parameter(torch.empty(16))

        self.alpha_pre_att = nn.Parameter(torch.tensor(1.0))
        self.alpha_post_att = nn.Parameter(torch.tensor(1.0))
        self.alpha_res_att = nn.Parameter(torch.tensor(1.0))

        nn.init.normal_(self.phi_pre_att, std=0.01)
        nn.init.normal_(self.phi_post_att, std=0.01)
        nn.init.normal_(self.phi_res_att, std=0.01)
        nn.init.normal_(self.b_pre_att, std=0.01)
        nn.init.normal_(self.b_post_att, std=0.01)
        nn.init.normal_(self.b_res_att, std=0.01)

        self.phi_pre_ffn = nn.Parameter(torch.empty(C, 4))
        self.phi_post_ffn = nn.Parameter(torch.empty(C, 4))
        self.phi_res_ffn = nn.Parameter(torch.empty(C, 16))
        self.b_pre_ffn = nn.Parameter(torch.empty(4))
        self.b_post_ffn = nn.Parameter(torch.empty(4))
        self.b_res_ffn = nn.Parameter(torch.empty(16))

        self.alpha_pre_ffn = nn.Parameter(torch.tensor(1.0))
        self.alpha_post_ffn = nn.Parameter(torch.tensor(1.0))
        self.alpha_res_ffn = nn.Parameter(torch.tensor(1.0))

        nn.init.normal_(self.phi_pre_ffn, std=0.01)
        nn.init.normal_(self.phi_post_ffn, std=0.01)
        nn.init.normal_(self.phi_res_ffn, std=0.01)
        nn.init.normal_(self.b_pre_ffn, std=0.01)
        nn.init.normal_(self.b_post_ffn, std=0.01)
        nn.init.normal_(self.b_res_ffn, std=0.01)

    def forward(self, V_blocks, partial_block, v_first):
        if self.layer_id == 0:
            partial_block = self.ln0(partial_block)

        h = block_attn_res(V_blocks, partial_block, self.attn_res_proj, self.attn_res_norm)

        if self.layer_id % self.layers_per_block == 0:
            V_blocks = torch.cat([V_blocks, partial_block.unsqueeze(0)], dim=0)
            partial_block = None

        B, T = h.shape[0], h.shape[1]
        x_norm = rmsnorm(h)
        tilde_H_pre_att = self.alpha_pre_att * (x_norm @ self.phi_pre_att) + self.b_pre_att
        tilde_H_post_att = self.alpha_post_att * (x_norm @ self.phi_post_att) + self.b_post_att
        tilde_H_res_att = self.alpha_res_att * (x_norm @ self.phi_res_att) + self.b_res_att
        tilde_H_res_att = tilde_H_res_att.view(B, T, 4, 4)
        H_pre_att = torch.sigmoid(tilde_H_pre_att)
        H_post_att = 2 * torch.sigmoid(tilde_H_post_att)
        H_res_att = Sinkhorn_Knopp(tilde_H_res_att)

        s_att = torch.einsum("b t e, b t c -> b t c", H_pre_att, h)
        xx, v_first, pre_out = self.att(self.ln1(s_att), v_first)

        Z_att = H_post_att.permute(2, 0, 1).unsqueeze(-1) * xx + torch.einsum(
            "b t i j, b t c -> i b t c", H_res_att, h
        )

        pre_out_norm = rmsnorm(pre_out)
        tilde_H_pre_ffn = self.alpha_pre_ffn * (pre_out_norm @ self.phi_pre_ffn) + self.b_pre_ffn
        tilde_H_post_ffn = (
            self.alpha_post_ffn * (pre_out_norm @ self.phi_post_ffn) + self.b_post_ffn
        )
        tilde_H_res_ffn = self.alpha_res_ffn * (pre_out_norm @ self.phi_res_ffn) + self.b_res_ffn
        tilde_H_res_ffn = tilde_H_res_ffn.view(B, T, 4, 4)
        H_pre_ffn = torch.sigmoid(tilde_H_pre_ffn)
        H_post_ffn = 2 * torch.sigmoid(tilde_H_post_ffn)
        H_res_ffn = Sinkhorn_Knopp(tilde_H_res_ffn)

        s_ffn = torch.einsum("b t e, e b t c -> b t c", H_pre_ffn, Z_att)
        h_ffn = block_attn_res(V_blocks, s_ffn, self.mlp_res_proj, self.mlp_res_norm)

        t_ffn = torch.stack(
            [self.experts[e](self.ln2(H_pre_ffn[..., e].unsqueeze(-1) * h_ffn)) for e in range(4)],
            dim=0,
        )

        H_post_ffn_reshaped = H_post_ffn.permute(2, 0, 1).unsqueeze(-1)
        Z_out = H_post_ffn_reshaped * t_ffn + torch.einsum(
            "b t i j, j b t c -> i b t c", H_res_ffn, Z_att
        )

        partial_block = Z_out.mean(dim=0)
        return V_blocks, partial_block, v_first


class L2Wrap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, loss, y):
        ctx.save_for_backward(y)
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        y = ctx.saved_tensors[0]
        # to encourage the logits to be close to 0
        factor = 1e-4 / (y.shape[0] * y.shape[1])
        maxx, ids = torch.max(y, -1, keepdim=True)
        gy = torch.zeros_like(y)
        gy.scatter_(-1, ids, maxx * factor)
        return (grad_output, gy)


class RWKV(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.emb = nn.Embedding(args.vocab_size, args.n_embd)
        self.blocks = nn.ModuleList(
            [MHCBlock(args, i) if i < 4 else Block(args, i) for i in range(args.n_layer)]
        )
        self.ln_out = nn.LayerNorm(args.n_embd)
        self.head = nn.Linear(args.n_embd, args.vocab_size, bias=False)

        if args.dropout > 0:
            self.drop0 = nn.Dropout(p=args.dropout)

    def pad_left(self, x, num_tokens_to_pad):
        # pad left with eos token embedding
        if num_tokens_to_pad != 0:
            eos_idx = torch.full(
                (x.size(0), num_tokens_to_pad),
                STOP_TOKEN_INDEX,
                dtype=torch.long,
                device=x.device,
            )
            eos_emb = self.emb(eos_idx)
            x = torch.cat((eos_emb, x), dim=1)
        return x

    def unpad(self, x, num_tokens_to_pad):
        # unpad
        if num_tokens_to_pad > 0:
            x = x[:, num_tokens_to_pad:]
        return x

    def forward(self, x):
        args = self.args

        num_tokens_to_pad = CHUNK_LEN - x.size(1) % CHUNK_LEN if x.size(1) % CHUNK_LEN != 0 else 0
        x = self.pad_left(x, num_tokens_to_pad)
        if args.dropout > 0:
            x = self.drop0(x)

        V_blocks = torch.empty(0, x.size(0), x.size(1), x.size(2), dtype=x.dtype, device=x.device)
        partial_block = x
        v_first = torch.empty_like(x)
        for block in self.blocks:
            if args.grad_cp == 1:
                V_blocks, partial_block, v_first = deepspeed.checkpointing.checkpoint(
                    block, V_blocks, partial_block, v_first
                )
            else:
                V_blocks, partial_block, v_first = block(V_blocks, partial_block, v_first)

        x = self.ln_out(partial_block)
        x = self.head(x)
        return self.unpad(x, num_tokens_to_pad)


class VisualTokenCompressor(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.blocks = nn.ModuleList([Block(args, i, is_vtc=True) for i in range(args.n_vtc_layer)])
        self.ln_out = nn.LayerNorm(args.n_embd)

    def pad_left(self, x, num_tokens_to_pad):
        # pad left with eos token embedding
        if num_tokens_to_pad != 0:
            # left padding by add zero emb at the beginning
            eos_emb = torch.zeros(
                x.size(0), num_tokens_to_pad, x.size(2), dtype=x.dtype, device=x.device
            )
            x = torch.cat((eos_emb, x), dim=1)
        return x

    def unpad(self, x, num_tokens_to_pad):
        # unpad
        if num_tokens_to_pad > 0:
            x = x[:, num_tokens_to_pad:]
        return x

    def forward(self, x):
        args = self.args

        num_tokens_to_pad = CHUNK_LEN - x.size(1) % CHUNK_LEN if x.size(1) % CHUNK_LEN != 0 else 0
        x = self.pad_left(x, num_tokens_to_pad)

        V_blocks = torch.empty(0, x.size(0), x.size(1), x.size(2), dtype=x.dtype, device=x.device)
        partial_block = x
        v_first = torch.empty_like(x)
        for i, block in enumerate(self.blocks):
            do_reverse = i % 2 == 1
            if do_reverse:  # reverse
                V_blocks = V_blocks.flip(2)
                partial_block = partial_block.flip(1)
                v_first = v_first.flip(1)

            if args.grad_cp == 1:
                V_blocks, partial_block, v_first = deepspeed.checkpointing.checkpoint(
                    block, V_blocks, partial_block, v_first
                )
            else:
                V_blocks, partial_block, v_first = block(V_blocks, partial_block, v_first)

            if do_reverse:  # reverse back
                V_blocks = V_blocks.flip(2)
                partial_block = partial_block.flip(1)
                v_first = v_first.flip(1)

        x = self.ln_out(partial_block)
        return self.unpad(x, num_tokens_to_pad)


class MLPWithContextGating(nn.Module):
    def __init__(self, in_dim, n_embd):
        super().__init__()
        self.gate = nn.Linear(in_dim, in_dim, bias=False)
        self.o_proj = nn.Linear(in_dim, n_embd, bias=False)
        self.ln_v = nn.LayerNorm(n_embd)

    def forward(self, x):
        # x: [B, T, D]
        gating = torch.sigmoid(self.gate(x))
        return self.ln_v(self.o_proj(x * gating))


class VELA(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.rwkv = RWKV(args)
        if len(args.load_model) > 0:
            self.load_rwkv_from_pretrained(args.load_model)

        # Determine dtype: FP16 on CUDA, FP32 on CPU
        if torch.cuda.is_available():
            self.vit_dtype = torch.float16
        else:
            self.vit_dtype = torch.float32

        # Load SigLino from HuggingFace Hub using our local vendored code
        self.vit, self.vit_processor = load_siglino_from_hub(
            repo_id=args.vision_tower_path,
            device="cpu",  # always load on CPU first; move after
            dtype=self.vit_dtype,
        )
        # Read dim before torch.compile potentially wraps self.vit
        hidden_size: int = self.vit.args.dim
        # Move to the right device (CUDA if available)
        if torch.cuda.is_available():
            self.vit = self.vit.cuda()

        # Tier-2 CPU optimisation: torch.compile cannot be used here because
        # the inductor will trace into SigLino's attention and attempt to lower
        # flex_attention (which requires CUDA) even with our runtime guards.
        # CPU inference runs in eager mode; it's still fast enough for inference.
        # CUDA path: model is already in FP16 on device, inference is fast.
        self.freeze_vit()

        self.proj = MLPWithContextGating(hidden_size, args.n_embd)
        self.vtc = VisualTokenCompressor(args)

    def init_vtc_weights(self):
        # Copy weights from rwkv to vtc
        self.vtc.ln_out.load_state_dict(self.rwkv.ln_out.state_dict())
        for i in range(self.args.n_vtc_layer):
            vtc_block = self.vtc.blocks[i]
            rwkv_block = self.rwkv.blocks[i]
            vtc_block.load_state_dict(rwkv_block.state_dict())

    def load_rwkv_from_pretrained(self, path):
        self.rwkv.load_state_dict(
            torch.load(path, map_location="cpu", weights_only=True), strict=False
        )
        rank_zero_info(f"Loaded pretrained RWKV from {path}")

    @property
    def deepspeed_offload(self) -> bool:
        strategy = self.trainer.strategy
        if isinstance(strategy, DeepSpeedStrategy):
            cfg = strategy.config["zero_optimization"]
            return cfg.get("offload_optimizer") or cfg.get("offload_param")
        return False

    def freeze_vit(self):
        self.vit.requires_grad_(False)

    def freeze_rwkv(self, num_layers_to_freeze):
        # freeze all layers including embedding and lm head
        if num_layers_to_freeze == self.args.n_layer:
            self.rwkv.requires_grad_(False)
        for i, block in enumerate(self.rwkv.blocks):
            block.requires_grad_(i >= num_layers_to_freeze)

    def freeze_emb(self):
        self.rwkv.emb.requires_grad_(False)

    def freeze_proj(self):
        self.proj.requires_grad_(False)

    def configure_optimizers(self):
        zero_weight_decay_group = [
            p for p in self.parameters() if len(p.squeeze().shape) < 2 and p.requires_grad
        ]
        # add weight decay to len(p.squeeze().shape) >= 2
        weight_decay_group = [
            p for p in self.parameters() if len(p.squeeze().shape) >= 2 and p.requires_grad
        ]

        name_of_trainable_params = [n for n, p in self.named_parameters() if p.requires_grad]
        compressed_name_of_trainable_params = compress_parameter_names(name_of_trainable_params)
        rank_zero_info(
            f"Name of trainable parameters in optimizers: {compressed_name_of_trainable_params}"
        )
        rank_zero_info(
            f"Number of trainable parameters in optimizers: {len(name_of_trainable_params)}"
        )
        optim_groups = []
        if zero_weight_decay_group:
            optim_groups += [{"params": zero_weight_decay_group, "weight_decay": 0.0}]
        if weight_decay_group:
            if self.args.weight_decay > 0:
                optim_groups += [
                    {"params": weight_decay_group, "weight_decay": self.args.weight_decay}
                ]
                rank_zero_info(
                    f"Number of parameters with weight decay: {len(weight_decay_group)}, with value: {self.args.weight_decay}"
                )
            else:
                optim_groups += [{"params": weight_decay_group, "weight_decay": 0.0}]
        if self.deepspeed_offload:
            return DeepSpeedCPUAdam(
                optim_groups,
                lr=self.args.lr_init,
                betas=self.args.betas,
                eps=self.args.adam_eps,
                bias_correction=True,
                adamw_mode=True,
                amsgrad=False,
            )
        return FusedAdam(
            optim_groups,
            lr=self.args.lr_init,
            betas=self.args.betas,
            eps=self.args.adam_eps,
            bias_correction=True,
            adam_w_mode=True,
            amsgrad=False,
        )

    def forward(self, samples):
        x, targets = self.preparing_embedding(samples)
        # unidirectional forward
        logits = self.rwkv(x)
        return logits, targets

    def training_step(self, batch, batch_idx):
        logits, targets = self(batch)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = targets[..., 1:].contiguous()
        # calculate valid length for each sample
        valid_lengths = (shift_labels != IGNORE_INDEX).sum(1)  # [B, T] -> [B]
        # if valid length is 0, set it to 1, to avoid division by zero
        valid_lengths = torch.max(valid_lengths, torch.ones_like(valid_lengths))
        # calculate loss， loss of IGNORE_INDEX will be set to 0
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=IGNORE_INDEX,
            reduction="none",
        )
        # Average the loss by valid label length
        loss = loss.view(shift_labels.size()).sum(1) / valid_lengths  # [B*T] -> [B, T] -> [B]
        loss = loss.mean()  # average over batch
        return L2Wrap.apply(loss, logits)

    def training_step_end(self, batch_parts):
        if pl.__version__[0] != "2":
            all = self.all_gather(batch_parts)
            if self.trainer.is_global_zero:
                self.trainer.my_loss_all = all

    def encode_images(self, images, spatial_shapes=None, padding_mask=None):
        if images.dim() == 5:
            # Traditional format: [B, N, C, H, W] — pixel images, patchify first
            B, N, C, H, W = images.shape
            images = images.view(B * N, C, H, W)
        elif images.dim() == 4:
            # Patchified format from SigLinoImageProcessor: [B, N, L, D]
            B, N, L, D = images.shape
            images = images.view(B * N, L, D)
            if spatial_shapes is not None and spatial_shapes.dim() == 3:
                spatial_shapes = spatial_shapes.view(B * N, -1)
            if padding_mask is not None and padding_mask.dim() == 3:
                padding_mask = padding_mask.view(B * N, -1)
        else:
            raise ValueError(f"Unexpected images shape: {images.shape}")

        # Call SigLino directly; compile=False disables internal flex compilation
        # (we use torch.compile at the module level on CPU instead)
        vit_outputs = self.vit(
            pixel_values=images,
            spatial_shapes=spatial_shapes,
            padding_mask=padding_mask,
            compile=torch.cuda.is_available(),  # flex compile only on CUDA
        )

        # SigLino always returns a dict with patch_features
        image_features = vit_outputs["patch_features"]["siglino"]  # (B*N, L, dim)

        _, L, D = image_features.shape
        image_features = image_features.view(B, N, L, D)
        return self.proj(image_features)  # (B, N, L, n_embd)

    def compress_visual_tokens(self, image_features, reduction="pool"):
        # image_features: [B, NL, D]
        B, N, L, D = image_features.shape
        image_features = image_features.view(B, N * L, D)  # global
        image_features = self.vtc(image_features)  # [B, N*L, D]
        if reduction == "step":
            step = L // self.args.num_token_per_image
            return image_features[:, ::step, :]
        elif reduction == "pool":
            output_length = self.args.num_token_per_image * N
            pool = nn.AdaptiveAvgPool1d(output_length)
            image_features = image_features.permute(0, 2, 1)  # [B, D, N*L]
            image_features = pool(image_features)  # [B, D, num_token_per_image*N]
            return image_features.permute(0, 2, 1)  # [B, num_token_per_image*N, D]

    def preparing_embedding(self, samples):
        if "images" not in samples:
            return self.rwkv.emb(samples["input_ids"]), samples["labels"]
        spatial_shapes = samples.get("spatial_shapes", None)
        padding_mask = samples.get("padding_mask", None)
        image_features = self.encode_images(
            samples["images"], spatial_shapes=spatial_shapes, padding_mask=padding_mask
        )
        image_features = image_features.to(dtype=self.rwkv.emb.weight.dtype)
        image_features = self.compress_visual_tokens(image_features)
        B_IMG, L_IMG, D_IMG = image_features.shape
        image_features = image_features.view(-1, D_IMG)
        input_embeds = self.rwkv.emb(samples["input_ids"])
        B, L, D = input_embeds.shape
        input_embeds = input_embeds.view(B * L, D)
        input_ids = samples["input_ids"].view(B * L)
        selected = input_ids == IMAGE_TOKEN_INDEX
        selected_sum = selected.sum()
        if selected_sum != B_IMG * L_IMG:
            # truncate the image_features, wrong way to handle this, but it is fine for now
            image_features = image_features[:selected_sum]
            sample_id = ":::".join(samples.get("sample_id", []))
            rank_zero_warn(
                f"\nsample_id: {sample_id}, image tokens: {selected_sum}, but image features: {B_IMG * L_IMG}\n"
            )
        # fill the image features to the input_embeds
        input_embeds[selected] = image_features
        return input_embeds.view(B, L, D), samples["labels"]

    def generate(
        self, input_ids, images, do_sample, temperature, top_p, max_new_tokens, stop_token_idx
    ) -> tuple[list[int], list[float], list[float]]:
        """Generate tokens one at a time (greedy only); single sample.

        Args:
            input_ids: [1, seq_len]
            images: dict of vision features, each [1, 3, H, W]
            do_sample: bool
            temperature: float
            top_p: float
            max_new_tokens: int
        """
        samples = {
            "input_ids": input_ids,
            "labels": torch.full_like(input_ids, IGNORE_INDEX),
        }
        if images is not None:
            samples["images"] = images
        x, _ = self.preparing_embedding(samples)
        generated_tokens = []
        generated_token_logits = []
        generated_token_probs = []
        for i in range(max_new_tokens):
            logits = self.rwkv(x)[:, -1, :]
            if do_sample:
                raise NotImplementedError
            else:  # greedy
                # [1, vocab_size] -> [1, 1]
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
                next_token_logit = logits.gather(-1, next_token)
                probs = torch.softmax(logits, dim=-1)
                next_token_prob = probs.gather(-1, next_token)
            generated_tokens.append(next_token.item())
            generated_token_logits.append(next_token_logit.item())
            generated_token_probs.append(next_token_prob.item())
            if generated_tokens[-1] == stop_token_idx:
                break
            x = torch.cat((x, self.rwkv.emb(next_token)), dim=-2)
            x = x[:, -self.args.ctx_len :, :]  # truncate
        return generated_tokens, generated_token_logits, generated_token_probs
