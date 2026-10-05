#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

static std::vector<char> pack(ggml_type type, int64_t row, int64_t rows, uint32_t seed) {
    std::vector<float> data(row*rows);
    for (float & x : data) {
        seed = seed*1664525u + 1013904223u;
        x = (int(seed >> 8) - 8388608)/8388608.0f;
    }
    std::vector<char> result(ggml_row_size(type, row)*rows);
    std::vector<float> imatrix(row, 1.0f);
    ggml_quantize_chunk(type, data.data(), result.data(), 0, rows, row, imatrix.data());
    return result;
}

static void set(ggml_tensor * tensor, const std::vector<char> & data) {
    ggml_backend_tensor_set(tensor, data.data(), 0, data.size());
}

static void check_columns(ggml_backend_t backend, ggml_tensor * out, int width,
        std::vector<float> & reference, const char * label) {
    std::vector<float> data(ggml_nelements(out));
    ggml_backend_tensor_get(out, data.data(), 0, data.size()*sizeof(float));
    const size_t column = data.size()/width;
    for (float x : data) {
        if (!std::isfinite(x)) { throw std::runtime_error("non-finite output"); }
    }
    if (width == 1) { reference = data; }
    for (int j = 0; j < width; ++j) {
        if (memcmp(reference.data(), data.data() + j*column, column*sizeof(float)) != 0) {
            fprintf(stderr, "FAIL: %s, width=%d, column=%d\n", label, width, j);
            throw std::runtime_error("decode/verification output mismatch");
        }
    }
}

static void attention(ggml_backend_t backend) {
    constexpr int heads = 3, gqa = 4;
    for (int d : {64, 128, 256}) {
        for (int nk : {256, 8192}) {
            for (ggml_type type : {GGML_TYPE_F16, GGML_TYPE_Q8_0, GGML_TYPE_Q4_0,
                    GGML_TYPE_Q4_1, GGML_TYPE_Q5_0, GGML_TYPE_Q5_1}) {
                const auto keys = pack(type, d, heads*(nk+17), 719);
                const auto values = pack(type, d, heads*(nk+17), 721);
                std::vector<float> reference;
                char label[128];
                snprintf(label, sizeof(label), "attention d=%d kv=%d type=%s", d, nk, ggml_type_name(type));
                for (int width = 1; width <= 8; ++width) {
                    auto ctx = ggml_init({4*1024*1024, nullptr, true});
                    auto q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, d, width, heads*gqa, 1);
                    auto kb = ggml_new_tensor_3d(ctx, type, d, heads, nk+17);
                    auto vb = ggml_new_tensor_3d(ctx, type, d, heads, nk+17);
                    auto k = ggml_permute(ctx, ggml_view_3d(ctx, kb, d, heads, nk, kb->nb[1], kb->nb[2], 0), 0, 2, 1, 3);
                    auto v = ggml_permute(ctx, ggml_view_3d(ctx, vb, d, heads, nk, vb->nb[1], vb->nb[2], 0), 0, 2, 1, 3);
                    auto mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, nk, width);
                    auto out = ggml_flash_attn_ext(ctx, q, k, v, mask, 1.0f/sqrtf(float(d)), 0, 0);
                    ggml_prec_set_acc(out, GGML_PREC_F32);
                    auto graph = ggml_new_graph(ctx);
                    ggml_build_forward_expand(graph, out);
                    auto buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
                    if (!buffer) { throw std::runtime_error("allocation failed"); }
                    set(kb, keys);
                    set(vb, values);
                    std::vector<float> queries(ggml_nelements(q));
                    for (size_t i = 0; i < queries.size(); ++i) {
                        queries[i] = sinf(float(i%d + 13*(i/(d*width))));
                    }
                    ggml_backend_tensor_set(q, queries.data(), 0, queries.size()*sizeof(float));
                    std::vector<ggml_fp16_t> masks(nk*width);
                    for (size_t i = 0; i < masks.size(); ++i) {
                        const int j = i % nk;
                        const bool visible = j <= nk-8 && j % 31 != 7 && !(j >= 64 && j < 128);
                        masks[i] = ggml_fp32_to_fp16(visible ? 0.0f : -INFINITY);
                    }
                    ggml_backend_tensor_set(mask, masks.data(), 0, masks.size()*sizeof(ggml_fp16_t));
                    if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
                        throw std::runtime_error("attention graph failed");
                    }
                    check_columns(backend, out, width, reference, label);
                    ggml_backend_buffer_free(buffer);
                    ggml_free(ctx);
                }
                printf("PASS: %s, widths=1..8\n", label);
            }
        }
    }
}

static void matvec(ggml_backend_t backend) {
    constexpr int rows = 33, experts = 4, used = 2;
    for (ggml_type type : {GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_BF16, GGML_TYPE_Q8_0,
            GGML_TYPE_Q4_K, GGML_TYPE_Q6_K, GGML_TYPE_IQ3_S, GGML_TYPE_IQ4_NL}) {
        for (int k : {256, 512, 4096}) {
            for (int mode : {0, 1, 2}) {
                const bool moe = mode == 1;
                const auto weights = pack(type, k, rows*(moe ? experts : 1), 719);
                const auto gates = mode == 2 ? pack(type, k, rows, 723) : std::vector<char>{};
                std::vector<float> reference;
                char label[128];
                snprintf(label, sizeof(label), "matvec k=%d mode=%d type=%s", k, mode, ggml_type_name(type));
                for (int width = 1; width <= 8; ++width) {
                    auto ctx = ggml_init({4*1024*1024, nullptr, true});
                    auto x = ggml_new_tensor_3d(ctx, type, k, rows, moe ? experts : 1);
                    auto y = moe ? ggml_new_tensor_3d(ctx, GGML_TYPE_F32, k, used, width)
                                 : ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, width);
                    auto ids = moe ? ggml_new_tensor_2d(ctx, GGML_TYPE_I32, used, width) : nullptr;
                    auto out = moe ? ggml_mul_mat_id(ctx, x, y, ids) : ggml_mul_mat(ctx, x, y);
                    ggml_tensor * gate = nullptr;
                    if (mode == 2) {
                        gate = ggml_new_tensor_2d(ctx, type, k, rows);
                        out = ggml_swiglu_split(ctx, ggml_mul_mat(ctx, gate, y), out);
                    }
                    auto graph = ggml_new_graph(ctx);
                    ggml_build_forward_expand(graph, out);
                    auto buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
                    if (!buffer) { throw std::runtime_error("allocation failed"); }
                    set(x, weights);
                    if (gate) { set(gate, gates); }
                    std::vector<float> input(ggml_nelements(y));
                    for (size_t i = 0; i < input.size(); ++i) { input[i] = sinf(float(i % (moe ? k*used : k))); }
                    ggml_backend_tensor_set(y, input.data(), 0, input.size()*sizeof(float));
                    if (ids) {
                        std::vector<int32_t> indices(used*width);
                        for (size_t i = 0; i < indices.size(); ++i) { indices[i] = (i % used)*2; }
                        ggml_backend_tensor_set(ids, indices.data(), 0, indices.size()*sizeof(int32_t));
                    }
                    if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
                        throw std::runtime_error("matvec graph failed");
                    }
                    check_columns(backend, out, width, reference, label);
                    ggml_backend_buffer_free(buffer);
                    ggml_free(ctx);
                }
                printf("PASS: %s, widths=1..8\n", label);
            }
        }
    }
}

int main() {
    ggml_backend_load_all();
    auto backend = ggml_backend_init_by_name("ROCm0", nullptr);
    if (!backend) { fprintf(stderr, "ROCm0 unavailable\n"); return 1; }
    try {
        attention(backend);
        matvec(backend);
    } catch (const std::exception & e) {
        fprintf(stderr, "%s\n", e.what());
        ggml_backend_free(backend);
        return 1;
    }
    ggml_backend_free(backend);
    puts("PASS: decode/verification consistency");
}
