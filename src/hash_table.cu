// src/hash_table.cu
#include "hash_table.hpp"
#include <cuda_runtime.h>
#include <cstring>

namespace dart {

// GPU hash function
__device__ uint32_t hash_function(uint64_t key, uint32_t table_size) {
    key = (~key) + (key << 21); // key = (key << 21) - key - 1;
    key = key ^ (key >> 24);
    key = (key + (key << 3)) + (key << 8); // key * 265
    key = key ^ (key >> 14);
    key = (key + (key << 2)) + (key << 4); // key * 21
    key = key ^ (key >> 28);
    key = key + (key << 31);
    return key % table_size;
}

// GPU kernel to insert keys into hash table
__global__ void HashTableInsertKernel(
    const uint128_t* keys,
    const uint32_t* values,
    uint32_t num_entries,
    HashEntry* hash_table,
    uint32_t table_size
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_entries) return;

    uint128_t key = keys[tid];
    uint32_t value = values[tid];

    // Compute hash index
    uint32_t hash_idx = hash_function(key.low ^ key.high, table_size);

    // Linear probing for collision resolution
    for (uint32_t probe = 0; probe < table_size; ++probe) {
        uint32_t idx = (hash_idx + probe) % table_size;

        // Try to insert using atomic compare-and-swap
        uint64_t old_low = atomicCAS(
            reinterpret_cast<unsigned long long*>(&hash_table[idx].key.low),
            0ULL,
            key.low
        );

        if (old_low == 0ULL || (old_low == key.low && hash_table[idx].key.high == key.high)) {
            // Successfully inserted or key already exists
            hash_table[idx].key = key;
            hash_table[idx].value = value;
            hash_table[idx].occupied = true;
            return;
        }
    }

    // Table is full - this shouldn't happen with proper sizing
}

// GPU kernel to lookup keys in hash table
__global__ void HashTableLookupKernel(
    const uint128_t* keys,
    uint32_t num_queries,
    const HashEntry* hash_table,
    uint32_t table_size,
    uint32_t* results,           // Output: values for each key
    bool* found                  // Output: whether key was found
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_queries) return;

    uint128_t key = keys[tid];
    uint32_t hash_idx = hash_function(key.low ^ key.high, table_size);

    // Linear probing to find key
    for (uint32_t probe = 0; probe < table_size; ++probe) {
        uint32_t idx = (hash_idx + probe) % table_size;

        if (!hash_table[idx].occupied) {
            // Empty slot - key not found
            found[tid] = false;
            return;
        }

        if (hash_table[idx].key.low == key.low &&
            hash_table[idx].key.high == key.high) {
            // Found the key
            results[tid] = hash_table[idx].value;
            found[tid] = true;
            return;
        }
    }

    // Not found after full scan
    found[tid] = false;
}

// GPU kernel to clear hash table
__global__ void HashTableClearKernel(
    HashEntry* hash_table,
    uint32_t table_size
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= table_size) return;

    hash_table[tid].key.low = 0;
    hash_table[tid].key.high = 0;
    hash_table[tid].value = 0;
    hash_table[tid].occupied = false;
}

// GPUHashTable implementation
GPUHashTable::GPUHashTable(uint32_t capacity)
    : capacity_(capacity)
    , table_size_(capacity * 2)  // Load factor of 0.5
    , d_hash_table_(nullptr)
    , num_entries_(0)
{
    // Allocate hash table on GPU
    size_t table_bytes = table_size_ * sizeof(HashEntry);
    cudaMalloc(&d_hash_table_, table_bytes);

    // Initialize to empty
    Clear();
}

GPUHashTable::~GPUHashTable() {
    if (d_hash_table_) {
        cudaFree(d_hash_table_);
    }
}

void GPUHashTable::Insert(
    const uint128_t* d_keys,
    const uint32_t* d_values,
    uint32_t num_entries,
    cudaStream_t stream
) {
    if (num_entries_ + num_entries > capacity_) {
        throw std::runtime_error("Hash table capacity exceeded");
    }

    int block_size = 256;
    int grid_size = (num_entries + block_size - 1) / block_size;

    HashTableInsertKernel<<<grid_size, block_size, 0, stream>>>(
        d_keys,
        d_values,
        num_entries,
        d_hash_table_,
        table_size_
    );

    num_entries_ += num_entries;
}

void GPUHashTable::Lookup(
    const uint128_t* d_keys,
    uint32_t num_queries,
    uint32_t* d_results,
    bool* d_found,
    cudaStream_t stream
) {
    int block_size = 256;
    int grid_size = (num_queries + block_size - 1) / block_size;

    HashTableLookupKernel<<<grid_size, block_size, 0, stream>>>(
        d_keys,
        num_queries,
        d_hash_table_,
        table_size_,
        d_results,
        d_found
    );
}

void GPUHashTable::Clear(cudaStream_t stream) {
    int block_size = 256;
    int grid_size = (table_size_ + block_size - 1) / block_size;

    HashTableClearKernel<<<grid_size, block_size, 0, stream>>>(
        d_hash_table_,
        table_size_
    );

    num_entries_ = 0;
}

uint32_t GPUHashTable::GetNumEntries() const {
    return num_entries_;
}

float GPUHashTable::GetLoadFactor() const {
    return static_cast<float>(num_entries_) / static_cast<float>(table_size_);
}

} // namespace dart
