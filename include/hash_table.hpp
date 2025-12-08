// include/hash_table.hpp
#pragma once

#include "dart_config.hpp"
#include <cstdint>
#include <cuda_runtime.h>

namespace dart {

// Hash table entry structure
struct HashEntry {
    uint128_t key;
    uint32_t value;
    bool occupied;
};

// GPU-accelerated hash table with linear probing
class GPUHashTable {
public:
    explicit GPUHashTable(uint32_t capacity);
    ~GPUHashTable();

    // Insert key-value pairs into hash table
    void Insert(
        const uint128_t* d_keys,
        const uint32_t* d_values,
        uint32_t num_entries,
        cudaStream_t stream = nullptr
    );

    // Lookup keys in hash table
    void Lookup(
        const uint128_t* d_keys,
        uint32_t num_queries,
        uint32_t* d_results,     // Output: values
        bool* d_found,           // Output: whether key found
        cudaStream_t stream = nullptr
    );

    // Clear hash table
    void Clear(cudaStream_t stream = nullptr);

    // Get number of entries
    uint32_t GetNumEntries() const;

    // Get load factor
    float GetLoadFactor() const;

private:
    HashEntry* d_hash_table_;
    uint32_t capacity_;
    uint32_t table_size_;
    uint32_t num_entries_;
};

} // namespace dart
