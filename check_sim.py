import torch
import math

B, T, H, C = 2, 16, 4, 64
CHUNK_LEN = 16
torch.manual_seed(123)
w = torch.randn(B, T, H, C, dtype=torch.float64) * 0.3
q = torch.randn(B, T, H, C, dtype=torch.float64) * 0.05
k = torch.randn(B, T, H, C, dtype=torch.float64) * 0.05
v = torch.randn(B, T, H, C, dtype=torch.float64) * 0.05
z = torch.randn(B, T, H, C, dtype=torch.float64) * 0.05
bb = torch.randn(B, T, H, C, dtype=torch.float64) * 0.05

w_ref, q_ref, k_ref, v_ref, z_ref, b_ref = [t.clone().detach().requires_grad_(True) for t in [w, q, k, v, z, bb]]

from Vela7.src.model import wind_backstepping_ref_forward
y_ref, s_chunk, sa = wind_backstepping_ref_forward(w_ref, q_ref, k_ref, v_ref, z_ref, b_ref)
y_ref.sum().backward()

w_f = w.tolist()
q_f = q.tolist()
k_f = k.tolist()
v_f = v.tolist()
z_f = z.tolist()
b_f = bb.tolist()
dy_f = torch.ones_like(y_ref).tolist()
sa_f = sa.tolist()
s_chunk_f = s_chunk.tolist()

dw_sim = torch.empty_like(w)
dq_sim = torch.empty_like(q)
dk_sim = torch.empty_like(k)
dv_sim = torch.empty_like(v)
dz_sim = torch.empty_like(z)
db_sim = torch.empty_like(bb)

for b_idx in range(B):
    for h_idx in range(H):
        dstate = [[0.0] * C for _ in range(C)]
        dstateT = [[0.0] * C for _ in range(C)]
        stateT = [[0.0] * C for _ in range(C)]
        
        for t in range(T - 1, -1, -1):
            q_t = q_f[b_idx][t][h_idx]
            w_val = w_f[b_idx][t][h_idx]
            w_fac_t = [-math.exp(x) for x in w_val]
            w_t = [math.exp(x) for x in w_fac_t]
            k_t = k_f[b_idx][t][h_idx]
            z_t = z_f[b_idx][t][h_idx]
            b_t = b_f[b_idx][t][h_idx]
            v_t = v_f[b_idx][t][h_idx]
            dy_t = dy_f[b_idx][t][h_idx]
            sa_t = sa_f[b_idx][t][h_idx]
            
            if (t + 1) % CHUNK_LEN == 0:
                for i in range(C):
                    for j in range(C):
                        stateT[i][j] = s_chunk_f[b_idx][h_idx][(t + 1) // CHUNK_LEN - 1][i][j]
            
            # dq_val
            dq_val = [0.0] * C
            for i in range(C):
                val = 0.0
                for j in range(C):
                    val += stateT[i][j] * dy_t[j]
                dq_val[i] = val
            dq_sim[b_idx, t, h_idx, :] = torch.tensor(dq_val, dtype=torch.float64)
            
            # stateT update
            for i in range(C):
                iwi = 1.0 / w_t[i]
                for j in range(C):
                    stateT[i][j] = (stateT[i][j] - k_t[i] * v_t[j] - b_t[i] * sa_t[j]) * iwi
                    dstate[i][j] += dy_t[i] * q_t[j]
                    dstateT[i][j] += q_t[i] * dy_t[j]
                    
            # dw_val, dk_val, dv_val, dSb, db_val
            dw_val = [0.0] * C
            dk_val = [0.0] * C
            dv_val = [0.0] * C
            dSb = [0.0] * C
            db_val = [0.0] * C
            
            for i in range(C):
                for j in range(C):
                    dw_val[i] += dstateT[i][j] * stateT[i][j]
                    dk_val[i] += dstateT[i][j] * v_t[j]
                    dv_val[i] += dstate[i][j] * k_t[j]
                    dSb[i] += dstate[i][j] * b_t[j]
                    db_val[i] += dstateT[i][j] * sa_t[j]
                    
            for i in range(C):
                dw_sim[b_idx, t, h_idx, i] = torch.tensor(dw_val[i] * w_t[i] * w_fac_t[i], dtype=torch.float64)
                dk_sim[b_idx, t, h_idx, i] = torch.tensor(dk_val[i], dtype=torch.float64)
                dv_sim[b_idx, t, h_idx, i] = torch.tensor(dv_val[i], dtype=torch.float64)
                db_sim[b_idx, t, h_idx, i] = torch.tensor(db_val[i], dtype=torch.float64)
                
            # dz_val
            dz_val = [0.0] * C
            for i in range(C):
                for j in range(C):
                    dz_val[i] += stateT[i][j] * dSb[j]
            dz_sim[b_idx, t, h_idx, :] = torch.tensor(dz_val, dtype=torch.float64)
            
            # dstate, dstateT update
            for i in range(C):
                for j in range(C):
                    dstate[i][j] = dstate[i][j] * w_t[j] + dSb[i] * z_t[j]
                    dstateT[i][j] = dstateT[i][j] * w_t[i] + z_t[i] * dSb[j]

from Vela7.src.model import wind_backstepping_ref_backward
dw_ref, dq_ref_val, dk_ref, dv_ref, dz_ref, db_ref = wind_backstepping_ref_backward(w_ref, q_ref, k_ref, v_ref, z_ref, b_ref, torch.ones_like(y_ref), s_chunk, sa)

print("--- Sim vs Ref ---")
print("q max diff:", (dq_sim - q_ref.grad).abs().max().item())
print("w max diff:", (dw_sim - w_ref.grad).abs().max().item())
print("z max diff:", (dz_sim - z_ref.grad).abs().max().item())

print("--- Ref Backward vs Autograd ---")
print("q max diff:", (dq_ref_val - q_ref.grad).abs().max().item())
print("w max diff:", (dw_ref - w_ref.grad).abs().max().item())
print("z max diff:", (dz_ref - z_ref.grad).abs().max().item())
