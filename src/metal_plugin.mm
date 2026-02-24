// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause
//
// Metal backend plugin for lux-accel

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <QuartzCore/QuartzCore.h>

#include <lux/accel/backend_api.h>

#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

// Core API provided by lux-accel
const lux_core_api_t* g_core_api = nullptr;

// Cached devices
NSArray<id<MTLDevice>>* g_devices = nil;

// =============================================================================
// Metal Buffer wrapper
// =============================================================================

struct MetalBuffer {
    id<MTLBuffer> buffer;
    size_t size;
    uint32_t usage;
};

// =============================================================================
// Metal Kernel wrapper
// =============================================================================

struct MetalKernel {
    id<MTLComputePipelineState> pipeline;
    std::string name;
    MTLSize workgroup_size = MTLSizeMake(256, 1, 1);
    std::unordered_map<uint32_t, std::pair<id<MTLBuffer>, size_t>> buffer_args;
    std::unordered_map<uint32_t, std::vector<uint8_t>> byte_args;
};

// =============================================================================
// Metal Queue wrapper
// =============================================================================

struct MetalQueue {
    id<MTLCommandQueue> queue;
    id<MTLCommandBuffer> current_buffer;
};

// =============================================================================
// Metal Device wrapper
// =============================================================================

struct MetalDevice {
    id<MTLDevice> device;
    std::string name;
    std::string vendor;
};

// =============================================================================
// Backend interface implementation
// =============================================================================

int metal_is_available() {
    if (!g_devices) {
        g_devices = MTLCopyAllDevices();
    }
    return g_devices.count > 0 ? 1 : 0;
}

int metal_init() {
    if (!g_devices) {
        g_devices = MTLCopyAllDevices();
    }
    return g_devices.count > 0 ? 1 : 0;
}

void metal_shutdown() {
    g_devices = nil;
}

int metal_get_device_count() {
    return g_devices ? static_cast<int>(g_devices.count) : 0;
}

int metal_get_device_caps(int index, lux_device_caps_t* caps) {
    if (!g_devices || index < 0 || index >= static_cast<int>(g_devices.count)) {
        return 0;
    }

    id<MTLDevice> dev = g_devices[index];

    // Store name in static buffer (simplification - should use proper storage)
    static thread_local std::string name_storage;
    name_storage = dev.name.UTF8String;

    caps->name = name_storage.c_str();
    caps->vendor = "Apple";
    caps->is_discrete = dev.isLowPower ? 0 : 1;
    caps->is_unified_memory = dev.hasUnifiedMemory ? 1 : 0;

    if (@available(macOS 10.15, iOS 13.0, *)) {
        caps->total_memory = dev.recommendedMaxWorkingSetSize;
        caps->max_buffer_size = dev.maxBufferLength;
    } else {
        caps->total_memory = 0;
        caps->max_buffer_size = 256 * 1024 * 1024;  // 256MB default
    }

    caps->max_workgroup_size = 1024;
    caps->simd_width = 32;
    caps->supports_fp16 = 1;
    caps->supports_subgroups = 1;

    return 1;
}

void* metal_create_device(int index) {
    if (!g_devices || index < 0 || index >= static_cast<int>(g_devices.count)) {
        return nullptr;
    }

    auto* dev = new MetalDevice();
    dev->device = g_devices[index];
    dev->name = dev->device.name.UTF8String;
    dev->vendor = "Apple";
    return dev;
}

void metal_destroy_device(void* device) {
    delete static_cast<MetalDevice*>(device);
}

void* metal_create_queue(void* device) {
    auto* dev = static_cast<MetalDevice*>(device);
    if (!dev) return nullptr;

    id<MTLCommandQueue> mtl_queue = [dev->device newCommandQueue];
    if (!mtl_queue) return nullptr;

    auto* queue = new MetalQueue();
    queue->queue = mtl_queue;
    queue->current_buffer = [mtl_queue commandBuffer];
    return queue;
}

void metal_destroy_queue(void* queue) {
    auto* q = static_cast<MetalQueue*>(queue);
    if (q) {
        q->current_buffer = nil;
        q->queue = nil;
        delete q;
    }
}

int metal_queue_submit(void* queue) {
    auto* q = static_cast<MetalQueue*>(queue);
    if (!q) return 0;

    [q->current_buffer commit];
    q->current_buffer = [q->queue commandBuffer];
    return 1;
}

int metal_queue_wait(void* queue) {
    auto* q = static_cast<MetalQueue*>(queue);
    if (!q) return 0;

    [q->current_buffer commit];
    [q->current_buffer waitUntilCompleted];
    q->current_buffer = [q->queue commandBuffer];
    return 1;
}

void* metal_create_buffer(void* device, const lux_buffer_desc_t* desc) {
    auto* dev = static_cast<MetalDevice*>(device);
    if (!dev || !desc) return nullptr;

    MTLResourceOptions options = MTLResourceStorageModeShared;
    id<MTLBuffer> mtl_buffer = [dev->device newBufferWithLength:desc->size options:options];
    if (!mtl_buffer) return nullptr;

    auto* buf = new MetalBuffer();
    buf->buffer = mtl_buffer;
    buf->size = desc->size;
    buf->usage = desc->usage;
    return buf;
}

void* metal_create_buffer_with_data(void* device, const lux_buffer_desc_t* desc, const void* data) {
    auto* dev = static_cast<MetalDevice*>(device);
    if (!dev || !desc || !data) return nullptr;

    id<MTLBuffer> mtl_buffer = [dev->device newBufferWithBytes:data
                                                         length:desc->size
                                                        options:MTLResourceStorageModeShared];
    if (!mtl_buffer) return nullptr;

    auto* buf = new MetalBuffer();
    buf->buffer = mtl_buffer;
    buf->size = desc->size;
    buf->usage = desc->usage;
    return buf;
}

void metal_destroy_buffer(void* buffer) {
    auto* buf = static_cast<MetalBuffer*>(buffer);
    if (buf) {
        buf->buffer = nil;
        delete buf;
    }
}

void* metal_map_buffer(void* buffer) {
    auto* buf = static_cast<MetalBuffer*>(buffer);
    if (!buf) return nullptr;
    return buf->buffer.contents;
}

void metal_unmap_buffer(void* buffer) {
    auto* buf = static_cast<MetalBuffer*>(buffer);
    if (buf && (buf->usage & LUX_BUFFER_USAGE_MAP_WRITE)) {
        [buf->buffer didModifyRange:NSMakeRange(0, buf->size)];
    }
}

void* metal_create_kernel_from_source(void* device, const char* source, const char* entry_point) {
    auto* dev = static_cast<MetalDevice*>(device);
    if (!dev || !source || !entry_point) return nullptr;

    @autoreleasepool {
        NSError* error = nil;
        NSString* src = [NSString stringWithUTF8String:source];
        MTLCompileOptions* opts = [[MTLCompileOptions alloc] init];
        opts.fastMathEnabled = YES;

        id<MTLLibrary> lib = [dev->device newLibraryWithSource:src options:opts error:&error];
        if (error || !lib) {
            if (g_core_api && g_core_api->log_error) {
                g_core_api->log_error(error ? error.localizedDescription.UTF8String : "Compilation failed");
            }
            return nullptr;
        }

        NSString* entry = [NSString stringWithUTF8String:entry_point];
        id<MTLFunction> fn = [lib newFunctionWithName:entry];
        if (!fn) {
            if (g_core_api && g_core_api->log_error) {
                g_core_api->log_error("Entry point not found");
            }
            return nullptr;
        }

        id<MTLComputePipelineState> pipeline = [dev->device newComputePipelineStateWithFunction:fn error:&error];
        if (error || !pipeline) {
            if (g_core_api && g_core_api->log_error) {
                g_core_api->log_error(error ? error.localizedDescription.UTF8String : "Pipeline creation failed");
            }
            return nullptr;
        }

        auto* kernel = new MetalKernel();
        kernel->pipeline = pipeline;
        kernel->name = entry_point;
        return kernel;
    }
}

void* metal_create_kernel_from_bundle(void* device, const void* bundle_data,
                                       size_t bundle_size, const char* entry_point) {
    auto* dev = static_cast<MetalDevice*>(device);
    if (!dev || !bundle_data || !entry_point) return nullptr;

    @autoreleasepool {
        NSError* error = nil;
        dispatch_data_t data = dispatch_data_create(bundle_data, bundle_size,
                                                     dispatch_get_main_queue(),
                                                     DISPATCH_DATA_DESTRUCTOR_DEFAULT);

        id<MTLLibrary> lib = [dev->device newLibraryWithData:data error:&error];
        if (error || !lib) {
            if (g_core_api && g_core_api->log_error) {
                g_core_api->log_error(error ? error.localizedDescription.UTF8String : "Bundle load failed");
            }
            return nullptr;
        }

        NSString* entry = [NSString stringWithUTF8String:entry_point];
        id<MTLFunction> fn = [lib newFunctionWithName:entry];
        if (!fn) {
            if (g_core_api && g_core_api->log_error) {
                g_core_api->log_error("Entry point not found");
            }
            return nullptr;
        }

        id<MTLComputePipelineState> pipeline = [dev->device newComputePipelineStateWithFunction:fn error:&error];
        if (error || !pipeline) {
            if (g_core_api && g_core_api->log_error) {
                g_core_api->log_error(error ? error.localizedDescription.UTF8String : "Pipeline creation failed");
            }
            return nullptr;
        }

        auto* kernel = new MetalKernel();
        kernel->pipeline = pipeline;
        kernel->name = entry_point;
        return kernel;
    }
}

void metal_destroy_kernel(void* kernel) {
    auto* k = static_cast<MetalKernel*>(kernel);
    if (k) {
        k->pipeline = nil;
        delete k;
    }
}

void metal_kernel_set_buffer(void* kernel, uint32_t index, void* buffer, size_t offset) {
    auto* k = static_cast<MetalKernel*>(kernel);
    auto* buf = static_cast<MetalBuffer*>(buffer);
    if (k && buf) {
        k->buffer_args[index] = {buf->buffer, offset};
    }
}

void metal_kernel_set_bytes(void* kernel, uint32_t index, const void* data, size_t size) {
    auto* k = static_cast<MetalKernel*>(kernel);
    if (k && data) {
        k->byte_args[index].assign(static_cast<const uint8_t*>(data),
                                    static_cast<const uint8_t*>(data) + size);
    }
}

void metal_kernel_set_workgroup_size(void* kernel, uint32_t x, uint32_t y, uint32_t z) {
    auto* k = static_cast<MetalKernel*>(kernel);
    if (k) {
        k->workgroup_size = MTLSizeMake(x, y, z);
    }
}

int metal_dispatch(void* queue, void* kernel, const lux_dispatch_desc_t* desc) {
    auto* q = static_cast<MetalQueue*>(queue);
    auto* k = static_cast<MetalKernel*>(kernel);
    if (!q || !k || !desc) return 0;

    id<MTLComputeCommandEncoder> encoder = [q->current_buffer computeCommandEncoder];
    [encoder setComputePipelineState:k->pipeline];

    // Set buffer arguments
    for (const auto& [idx, arg] : k->buffer_args) {
        [encoder setBuffer:arg.first offset:arg.second atIndex:idx];
    }

    // Set byte arguments
    for (const auto& [idx, data] : k->byte_args) {
        [encoder setBytes:data.data() length:data.size() atIndex:idx];
    }

    // Dispatch
    MTLSize grid = MTLSizeMake(desc->grid_x, desc->grid_y, desc->grid_z);
    [encoder dispatchThreadgroups:grid threadsPerThreadgroup:k->workgroup_size];
    [encoder endEncoding];

    return 1;
}

int metal_copy_buffer(void* queue, void* src, size_t src_offset,
                       void* dst, size_t dst_offset, size_t size) {
    auto* q = static_cast<MetalQueue*>(queue);
    auto* src_buf = static_cast<MetalBuffer*>(src);
    auto* dst_buf = static_cast<MetalBuffer*>(dst);
    if (!q || !src_buf || !dst_buf) return 0;

    id<MTLBlitCommandEncoder> encoder = [q->current_buffer blitCommandEncoder];
    [encoder copyFromBuffer:src_buf->buffer sourceOffset:src_offset
                   toBuffer:dst_buf->buffer destinationOffset:dst_offset
                       size:size];
    [encoder endEncoding];
    return 1;
}

int metal_fill_buffer(void* queue, void* buffer, size_t offset, size_t size, uint8_t value) {
    auto* q = static_cast<MetalQueue*>(queue);
    auto* buf = static_cast<MetalBuffer*>(buffer);
    if (!q || !buf) return 0;

    id<MTLBlitCommandEncoder> encoder = [q->current_buffer blitCommandEncoder];
    [encoder fillBuffer:buf->buffer range:NSMakeRange(offset, size) value:value];
    [encoder endEncoding];
    return 1;
}

// =============================================================================
// Backend interface structure
// =============================================================================

static lux_backend_interface_t g_metal_interface = {
    .type = LUX_BACKEND_TYPE_METAL,
    .name = "metal",
    .api_version = LUX_BACKEND_API_VERSION,

    .is_available = metal_is_available,
    .init = metal_init,
    .shutdown = metal_shutdown,

    .get_device_count = metal_get_device_count,
    .get_device_caps = metal_get_device_caps,

    .create_device = metal_create_device,
    .destroy_device = metal_destroy_device,

    .create_queue = metal_create_queue,
    .destroy_queue = metal_destroy_queue,
    .queue_submit = metal_queue_submit,
    .queue_wait = metal_queue_wait,

    .create_buffer = metal_create_buffer,
    .create_buffer_with_data = metal_create_buffer_with_data,
    .destroy_buffer = metal_destroy_buffer,
    .map_buffer = metal_map_buffer,
    .unmap_buffer = metal_unmap_buffer,

    .create_kernel_from_source = metal_create_kernel_from_source,
    .create_kernel_from_bundle = metal_create_kernel_from_bundle,
    .destroy_kernel = metal_destroy_kernel,
    .kernel_set_buffer = metal_kernel_set_buffer,
    .kernel_set_bytes = metal_kernel_set_bytes,
    .kernel_set_workgroup_size = metal_kernel_set_workgroup_size,

    .dispatch = metal_dispatch,
    .copy_buffer = metal_copy_buffer,
    .fill_buffer = metal_fill_buffer,
};

} // anonymous namespace

// =============================================================================
// Plugin entry point
// =============================================================================

extern "C" {

LUX_PLUGIN_EXPORT lux_backend_interface_t* lux_backend_init(const lux_core_api_t* core_api) {
    g_core_api = core_api;

    if (core_api && core_api->log_info) {
        core_api->log_info("Metal backend plugin loaded");
    }

    return &g_metal_interface;
}

} // extern "C"
