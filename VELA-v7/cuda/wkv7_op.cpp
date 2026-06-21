#include <torch/extension.h>

#ifdef USE_CUDA
#include <cuda_bf16.h>
using bf = __nv_bfloat16;

void cuda_forward(int B, int T, int H, bf*w, bf*q, bf*k, bf*v, bf*z, bf*a, bf*y, float*s, float*sa);
void cuda_backward(int B, int T, int H, bf*w, bf*q, bf*k, bf*v, bf*z, bf*a, bf*dy, float*s, float*sa, bf*dw, bf*dq, bf*dk, bf*dv, bf*dz, bf*da);

void forward_cuda(torch::Tensor &w, torch::Tensor &q, torch::Tensor &k, torch::Tensor &v, torch::Tensor &z, torch::Tensor &a, torch::Tensor &y, torch::Tensor &s, torch::Tensor &sa) {
    int B = w.sizes()[0], T = w.sizes()[1], H = w.sizes()[2];
    cuda_forward(B, T, H, (bf*)w.data_ptr(), (bf*)q.data_ptr(), (bf*)k.data_ptr(), (bf*)v.data_ptr(), (bf*)z.data_ptr(), (bf*)a.data_ptr(), (bf*)y.data_ptr(), (float*)s.data_ptr(), (float*)sa.data_ptr());
}

void backward_cuda(torch::Tensor &w, torch::Tensor &q, torch::Tensor &k, torch::Tensor &v, torch::Tensor &z, torch::Tensor &a, torch::Tensor &dy,
        torch::Tensor &s, torch::Tensor &sa, torch::Tensor &dw, torch::Tensor &dq, torch::Tensor &dk, torch::Tensor &dv, torch::Tensor &dz, torch::Tensor &da) {
    int B = w.sizes()[0], T = w.sizes()[1], H = w.sizes()[2];
    cuda_backward(B, T, H, (bf*)w.data_ptr(), (bf*)q.data_ptr(), (bf*)k.data_ptr(), (bf*)v.data_ptr(), (bf*)z.data_ptr(), (bf*)a.data_ptr(), (bf*)dy.data_ptr(), 
            (float*)s.data_ptr(), (float*)sa.data_ptr(), (bf*)dw.data_ptr(), (bf*)dq.data_ptr(), (bf*)dk.data_ptr(), (bf*)dv.data_ptr(), (bf*)dz.data_ptr(), (bf*)da.data_ptr());
}
#endif

// CPU implementation (always compiled)
#include <cmath>
#include <vector>
#include <ATen/Parallel.h>

void forward_cpu(torch::Tensor &w, torch::Tensor &q, torch::Tensor &k, torch::Tensor &v, torch::Tensor &z, torch::Tensor &a, torch::Tensor &y, torch::Tensor &s, torch::Tensor &sa) {
    int B = w.sizes()[0];
    int T = w.sizes()[1];
    int H = w.sizes()[2];
    int C = w.sizes()[3];
    int CHUNK_LEN = 16;

    auto w_data = w.data_ptr<c10::BFloat16>();
    auto q_data = q.data_ptr<c10::BFloat16>();
    auto k_data = k.data_ptr<c10::BFloat16>();
    auto v_data = v.data_ptr<c10::BFloat16>();
    auto z_data = z.data_ptr<c10::BFloat16>();
    auto a_data = a.data_ptr<c10::BFloat16>();
    auto y_data = y.data_ptr<c10::BFloat16>();
    auto s_data = s.data_ptr<float>();
    auto sa_data = sa.data_ptr<float>();

    at::parallel_for(0, B * H, 1, [&](int64_t start, int64_t end) {
        for (int64_t bh = start; bh < end; bh++) {
            int b = bh / H;
            int h = bh % H;

            std::vector<float> state(C * C, 0.0f);
            std::vector<float> q_local(C);
            std::vector<float> k_local(C);
            std::vector<float> w_local(C);
            std::vector<float> z_local(C);
            std::vector<float> a_local(C);
            std::vector<float> v_local(C);
            std::vector<float> sa_local(C);

            for (int t = 0; t < T; t++) {
                int ind_base = b * T * H * C + t * H * C + h * C;

                for (int i = 0; i < C; i++) {
                    int ind = ind_base + i;
                    q_local[i] = float(q_data[ind]);
                    w_local[i] = std::exp(-std::exp(float(w_data[ind])));
                    k_local[i] = float(k_data[ind]);
                    z_local[i] = float(z_data[ind]);
                    a_local[i] = float(a_data[ind]);
                    v_local[i] = float(v_data[ind]);
                }

                for (int i = 0; i < C; i++) {
                    float sa_val = 0.0f;
                    for (int j = 0; j < C; j++) {
                        sa_val += z_local[j] * state[j * C + i];
                    }
                    sa_local[i] = sa_val;
                    sa_data[ind_base + i] = sa_val;
                }

                for (int i = 0; i < C; i++) {
                    float y_val = 0.0f;
                    for (int j = 0; j < C; j++) {
                        float &s_val = state[j * C + i];
                        s_val = s_val * w_local[j] + sa_local[i] * a_local[j] + k_local[j] * v_local[i];
                        y_val += s_val * q_local[j];
                    }
                    y_data[ind_base + i] = c10::BFloat16(y_val);
                }

                if ((t + 1) % CHUNK_LEN == 0) {
                    int base = bh * (T / CHUNK_LEN) * C * C + (t / CHUNK_LEN) * C * C;
                    for (int j = 0; j < C; j++) {
                        for (int i = 0; i < C; i++) {
                            s_data[base + j * C + i] = state[j * C + i];
                        }
                    }
                }
            }
        }
    });
}

void backward_cpu(torch::Tensor &w, torch::Tensor &q, torch::Tensor &k, torch::Tensor &v, torch::Tensor &z, torch::Tensor &a, torch::Tensor &dy,
        torch::Tensor &s, torch::Tensor &sa, torch::Tensor &dw, torch::Tensor &dq, torch::Tensor &dk, torch::Tensor &dv, torch::Tensor &dz, torch::Tensor &da) {
    int B = w.sizes()[0];
    int T = w.sizes()[1];
    int H = w.sizes()[2];
    int C = w.sizes()[3];
    int CHUNK_LEN = 16;

    auto w_data = w.data_ptr<c10::BFloat16>();
    auto q_data = q.data_ptr<c10::BFloat16>();
    auto k_data = k.data_ptr<c10::BFloat16>();
    auto v_data = v.data_ptr<c10::BFloat16>();
    auto z_data = z.data_ptr<c10::BFloat16>();
    auto a_data = a.data_ptr<c10::BFloat16>();
    auto dy_data = dy.data_ptr<c10::BFloat16>();
    auto s_data = s.data_ptr<float>();
    auto sa_data = sa.data_ptr<float>();

    auto dw_data = dw.data_ptr<c10::BFloat16>();
    auto dq_data = dq.data_ptr<c10::BFloat16>();
    auto dk_data = dk.data_ptr<c10::BFloat16>();
    auto dv_data = dv.data_ptr<c10::BFloat16>();
    auto dz_data = dz.data_ptr<c10::BFloat16>();
    auto da_data = da.data_ptr<c10::BFloat16>();

    at::parallel_for(0, B * H, 1, [&](int64_t start, int64_t end) {
        for (int64_t bh = start; bh < end; bh++) {
            int b = bh / H;
            int h = bh % H;

            std::vector<float> dstate(C * C, 0.0f);
            std::vector<float> dstateT(C * C, 0.0f);
            std::vector<float> stateT(C * C, 0.0f);

            std::vector<float> q_local(C);
            std::vector<float> w_local(C);
            std::vector<float> w_fac_local(C);
            std::vector<float> k_local(C);
            std::vector<float> z_local(C);
            std::vector<float> a_local(C);
            std::vector<float> v_local(C);
            std::vector<float> dy_local(C);
            std::vector<float> sa_local(C);
            std::vector<float> dSb_local(C);

            for (int t = T - 1; t >= 0; t--) {
                int ind_base = b * T * H * C + t * H * C + h * C;

                for (int i = 0; i < C; i++) {
                    int ind = ind_base + i;
                    q_local[i] = float(q_data[ind]);
                    float w_val = float(w_data[ind]);
                    w_fac_local[i] = -std::exp(w_val);
                    w_local[i] = std::exp(w_fac_local[i]);
                    k_local[i] = float(k_data[ind]);
                    z_local[i] = float(z_data[ind]);
                    a_local[i] = float(a_data[ind]);
                    v_local[i] = float(v_data[ind]);
                    dy_local[i] = float(dy_data[ind]);
                    sa_local[i] = sa_data[ind];
                }

                if ((t + 1) % CHUNK_LEN == 0) {
                    int base = bh * (T / CHUNK_LEN) * C * C + (t / CHUNK_LEN) * C * C;
                    for (int i = 0; i < C; i++) {
                        for (int j = 0; j < C; j++) {
                            stateT[i * C + j] = s_data[base + i * C + j];
                        }
                    }
                }

                for (int i = 0; i < C; i++) {
                    float dq_val = 0.0f;
                    for (int j = 0; j < C; j++) {
                        dq_val += stateT[i * C + j] * dy_local[j];
                    }
                    dq_data[ind_base + i] = c10::BFloat16(dq_val);
                }

                for (int i = 0; i < C; i++) {
                    float iwi = 1.0f / w_local[i];
                    for (int j = 0; j < C; j++) {
                        stateT[i * C + j] = (stateT[i * C + j] - k_local[i] * v_local[j] - a_local[i] * sa_local[j]) * iwi;
                        dstate[i * C + j] += dy_local[i] * q_local[j];
                        dstateT[i * C + j] += q_local[i] * dy_local[j];
                    }
                }

                for (int i = 0; i < C; i++) {
                    float dw_val = 0.0f;
                    float dk_val = 0.0f;
                    float dv_val = 0.0f;
                    float dSb_val = 0.0f;
                    float db_val = 0.0f;
                    for (int j = 0; j < C; j++) {
                        dw_val += dstateT[i * C + j] * stateT[i * C + j];
                        dk_val += dstateT[i * C + j] * v_local[j];
                        dv_val += dstate[i * C + j] * k_local[j];
                        dSb_val += dstate[i * C + j] * a_local[j];
                        db_val += dstateT[i * C + j] * sa_local[j];
                    }
                    dw_data[ind_base + i] = c10::BFloat16(dw_val * w_local[i] * w_fac_local[i]);
                    dk_data[ind_base + i] = c10::BFloat16(dk_val);
                    dv_data[ind_base + i] = c10::BFloat16(dv_val);
                    da_data[ind_base + i] = c10::BFloat16(db_val);
                    dSb_local[i] = dSb_val;
                }

                for (int i = 0; i < C; i++) {
                    float da_val = 0.0f;
                    for (int j = 0; j < C; j++) {
                        da_val += stateT[i * C + j] * dSb_local[j];
                    }
                    dz_data[ind_base + i] = c10::BFloat16(da_val);
                }

                for (int i = 0; i < C; i++) {
                    for (int j = 0; j < C; j++) {
                        dstate[i * C + j] = dstate[i * C + j] * w_local[i] + dSb_local[i] * z_local[j];
                        dstateT[i * C + j] = dstateT[i * C + j] * w_local[j] + z_local[i] * dSb_local[j];
                    }
                }
            }
        }
    });
}

TORCH_LIBRARY(wind_backstepping, m) {
    m.def("forward(Tensor w, Tensor q, Tensor k, Tensor v, Tensor z, Tensor a, Tensor(a!) y, Tensor(b!) s, Tensor(c!) sa) -> ()");
    m.def("backward(Tensor w, Tensor q, Tensor k, Tensor v, Tensor z, Tensor a, Tensor dy, Tensor s, Tensor sa, Tensor(a!) dw, Tensor(b!) dq, Tensor(c!) dk, Tensor(d!) dv, Tensor(e!) dz, Tensor(f!) da) -> ()");
}

#ifdef USE_CUDA
TORCH_LIBRARY_IMPL(wind_backstepping, CUDA, m) {
    m.impl("forward", &forward_cuda);
    m.impl("backward", &backward_cuda);
}
#endif

TORCH_LIBRARY_IMPL(wind_backstepping, CPU, m) {
    m.impl("forward", &forward_cpu);
    m.impl("backward", &backward_cpu);
}
