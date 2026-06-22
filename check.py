import torch
from Vela7.src.model import WindBackstepping, wind_backstepping_ref_forward

B, T, H, C = 2, 16, 4, 64
CHUNK_LEN = 16
torch.manual_seed(123)
w = (torch.randn(B, T, H, C, dtype=torch.bfloat16) * 0.3).detach().requires_grad_(True)
q = (torch.randn(B, T, H, C, dtype=torch.bfloat16) * 0.05).detach().requires_grad_(True)
k = (torch.randn(B, T, H, C, dtype=torch.bfloat16) * 0.05).detach().requires_grad_(True)
v = (torch.randn(B, T, H, C, dtype=torch.bfloat16) * 0.05).detach().requires_grad_(True)
z = (torch.randn(B, T, H, C, dtype=torch.bfloat16) * 0.05).detach().requires_grad_(True)
bb = (torch.randn(B, T, H, C, dtype=torch.bfloat16) * 0.05).detach().requires_grad_(True)

w_ref, q_ref, k_ref, v_ref, z_ref, b_ref = [t.clone().detach().requires_grad_(True) for t in [w, q, k, v, z, bb]]

y_cpp = WindBackstepping.apply(w, q, k, v, z, bb)
y_ref, s_ref, sa_ref = wind_backstepping_ref_forward(w_ref, q_ref, k_ref, v_ref, z_ref, b_ref)

y_cpp.sum().backward()
grads_cpp = [w.grad.clone(), q.grad.clone(), k.grad.clone(), v.grad.clone(), z.grad.clone(), bb.grad.clone()]

from Vela7.src.model import wind_backstepping_ref_backward
dy = torch.ones_like(y_ref)
grads_ref = wind_backstepping_ref_backward(w_ref, q_ref, k_ref, v_ref, z_ref, b_ref, dy, s_ref, sa_ref)

for gc, gr, name in zip(grads_cpp, grads_ref, ["w", "q", "k", "v", "z", "bb"]):
    print(f"{name} grad cpp vs ref max diff:", (gc - gr).abs().max().item())
