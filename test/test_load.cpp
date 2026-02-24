// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause
//
// Smoke test: dlopen the plugin, resolve lux_backend_init, query interface,
// then on macOS hardware actually create a device, allocate a buffer, run a
// minimal Metal compute kernel and verify the result.

#include <lux/accel/backend_api.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>

namespace {

void log_info(const char* msg)  { std::fprintf(stderr, "[info]  %s\n", msg); }
void log_warn(const char* msg)  { std::fprintf(stderr, "[warn]  %s\n", msg); }
void log_error(const char* msg) { std::fprintf(stderr, "[error] %s\n", msg); }
void log_debug(const char* msg) { std::fprintf(stderr, "[debug] %s\n", msg); }

void* core_alloc(size_t n) { return std::malloc(n); }
void  core_free(void* p)   { std::free(p); }

const void* get_kernel_bundle(const char*, size_t* sz) { if (sz) *sz = 0; return nullptr; }
const char* get_kernel_source(const char*) { return nullptr; }

// Simple Metal compute kernel: out[i] = a[i] + b[i]
const char* kAddKernel = R"METAL(
#include <metal_stdlib>
using namespace metal;
kernel void vec_add(device const float* a [[buffer(0)]],
                    device const float* b [[buffer(1)]],
                    device float* c       [[buffer(2)]],
                    constant uint& n      [[buffer(3)]],
                    uint gid              [[thread_position_in_grid]]) {
    if (gid < n) c[gid] = a[gid] + b[gid];
}
)METAL";

} // namespace

int main(int argc, char** argv) {
    const char* path = argc > 1 ? argv[1] : "./lux_metal.plugin";

    void* handle = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (!handle) {
        std::fprintf(stderr, "dlopen failed: %s\n", dlerror());
        return 1;
    }

    auto init = reinterpret_cast<lux_backend_init_fn>(
        dlsym(handle, LUX_BACKEND_INIT_SYMBOL));
    if (!init) {
        std::fprintf(stderr, "dlsym(%s) failed: %s\n",
                     LUX_BACKEND_INIT_SYMBOL, dlerror());
        dlclose(handle);
        return 1;
    }

    lux_core_api_t core = {};
    core.api_version       = LUX_BACKEND_API_VERSION;
    core.log_debug         = log_debug;
    core.log_info          = log_info;
    core.log_warn          = log_warn;
    core.log_error         = log_error;
    core.alloc             = core_alloc;
    core.free              = core_free;
    core.get_kernel_bundle = get_kernel_bundle;
    core.get_kernel_source = get_kernel_source;

    lux_backend_interface_t* iface = init(&core);
    if (!iface) { std::fprintf(stderr, "init returned null\n"); dlclose(handle); return 1; }

    if (iface->api_version != LUX_BACKEND_API_VERSION ||
        iface->type != LUX_BACKEND_TYPE_METAL ||
        !iface->name || std::strcmp(iface->name, "metal") != 0) {
        std::fprintf(stderr, "interface identity mismatch\n");
        dlclose(handle); return 1;
    }

    if (!iface->is_available || !iface->is_available()) {
        std::fprintf(stderr, "metal not available on this system; identity-only OK\n");
        dlclose(handle);
        return 0;
    }

    if (!iface->init || !iface->init()) {
        std::fprintf(stderr, "init() failed\n");
        dlclose(handle); return 1;
    }

    int nDev = iface->get_device_count();
    if (nDev <= 0) {
        std::fprintf(stderr, "no metal devices\n");
        iface->shutdown();
        dlclose(handle); return 1;
    }

    lux_device_caps_t caps = {};
    iface->get_device_caps(0, &caps);
    std::fprintf(stderr, "metal device 0: %s (vendor=%s, total_mem=%llu)\n",
                 caps.name ? caps.name : "?",
                 caps.vendor ? caps.vendor : "?",
                 (unsigned long long)caps.total_memory);

    void* dev = iface->create_device(0);
    if (!dev) { std::fprintf(stderr, "create_device failed\n"); dlclose(handle); return 1; }

    void* queue = iface->create_queue(dev);
    if (!queue) { std::fprintf(stderr, "create_queue failed\n"); dlclose(handle); return 1; }

    constexpr uint32_t N = 1024;
    float a[N], b[N];
    for (uint32_t i = 0; i < N; i++) { a[i] = (float)i; b[i] = (float)(2*i); }

    lux_buffer_desc_t desc_a = { N * sizeof(float),
        LUX_BUFFER_USAGE_STORAGE | LUX_BUFFER_USAGE_COPY_SRC, nullptr };
    lux_buffer_desc_t desc_b = desc_a;
    lux_buffer_desc_t desc_c = { N * sizeof(float),
        LUX_BUFFER_USAGE_STORAGE | LUX_BUFFER_USAGE_MAP_READ, nullptr };

    void* bufA = iface->create_buffer_with_data(dev, &desc_a, a);
    void* bufB = iface->create_buffer_with_data(dev, &desc_b, b);
    void* bufC = iface->create_buffer(dev, &desc_c);
    if (!bufA || !bufB || !bufC) {
        std::fprintf(stderr, "create_buffer failed\n");
        dlclose(handle); return 1;
    }

    void* kernel = iface->create_kernel_from_source(dev, kAddKernel, "vec_add");
    if (!kernel) { std::fprintf(stderr, "kernel compile failed\n"); dlclose(handle); return 1; }

    iface->kernel_set_buffer(kernel, 0, bufA, 0);
    iface->kernel_set_buffer(kernel, 1, bufB, 0);
    iface->kernel_set_buffer(kernel, 2, bufC, 0);
    iface->kernel_set_bytes (kernel, 3, &N, sizeof(N));
    iface->kernel_set_workgroup_size(kernel, 64, 1, 1);

    lux_dispatch_desc_t disp = { (N + 63) / 64, 1, 1, 64, 1, 1 };
    if (!iface->dispatch(queue, kernel, &disp)) {
        std::fprintf(stderr, "dispatch failed\n"); return 1;
    }
    if (!iface->queue_wait(queue)) {
        std::fprintf(stderr, "queue_wait failed\n"); return 1;
    }

    void* mapped = iface->map_buffer(bufC);
    if (!mapped) { std::fprintf(stderr, "map_buffer failed\n"); return 1; }
    const float* out = static_cast<const float*>(mapped);
    int errors = 0;
    for (uint32_t i = 0; i < N; i++) {
        float expect = a[i] + b[i];
        if (out[i] != expect) {
            if (errors < 4) std::fprintf(stderr, "mismatch %u: %f != %f\n",
                                         i, (double)out[i], (double)expect);
            errors++;
        }
    }
    iface->unmap_buffer(bufC);

    iface->destroy_kernel(kernel);
    iface->destroy_buffer(bufA);
    iface->destroy_buffer(bufB);
    iface->destroy_buffer(bufC);
    iface->destroy_queue(queue);
    iface->destroy_device(dev);
    iface->shutdown();

    dlclose(handle);
    if (errors) {
        std::fprintf(stderr, "FAIL: %d/%u mismatches\n", errors, N);
        return 1;
    }
    std::fprintf(stderr, "OK: %u-element vector add via Metal verified\n", N);
    return 0;
}
