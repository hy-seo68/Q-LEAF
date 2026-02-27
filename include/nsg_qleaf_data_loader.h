#ifndef NSG_QLEAF_DATA_LOADER_H
#define NSG_QLEAF_DATA_LOADER_H

#include "aligned_allocator.h"
#include <vector>
#include <string>
#include <memory>
#include <unordered_map>
#include <cstdint>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

// EFANNA NSG headers
#include <efanna2e/index_nsg.h>

/**
  * @brief .fvecs file reader using mmap (zero-copy, shared memory)
  */
class MmapVectorReader {
private:
    int fd_;
    void* mmap_addr_;
    size_t file_size_;
    size_t num_vectors_;
    size_t dim_;
    size_t record_size_;   // (dim + 1) * sizeof(float) - size of one record

public:
    /**
     * @brief Open .fvecs file with mmap
     */
    MmapVectorReader(const char* filename);

    /**
     * @brief Destructor: release mmap and close file
     */
    ~MmapVectorReader();

    // Prevent copying (mmap is a unique resource)
    MmapVectorReader(const MmapVectorReader&) = delete;
    MmapVectorReader& operator=(const MmapVectorReader&) = delete;

    /**
     * @brief Return vector pointer for a specific index (zero-copy)
     */
    inline const float* get_vector(uint32_t index) const {
        if (index >= num_vectors_) {
            throw std::out_of_range("Vector index out of range: " +
                std::to_string(index) + " >= " + std::to_string(num_vectors_));
        }
        char* base = static_cast<char*>(mmap_addr_);

        return reinterpret_cast<const float*>(
            base + index * record_size_ + sizeof(int));
    }

    /**
     * @brief Set access pattern hint for mmap region
     */
    void set_access_pattern(int advice);

    size_t num_vectors() const { return num_vectors_; }
    size_t dimension() const { return dim_; }
    size_t file_size() const { return file_size_; }
};

/**
 * @brief Query information structure
 */
struct Query {
    std::vector<float> vector;
    std::string category;  
    int category_id;       
};

/**
 * @brief NSG subgraph information (Global Pool + Mapping structure)
 */
struct NSGSubgraph {
    std::string category;
    std::shared_ptr<efanna2e::IndexNSG> index;

    // Global Pool approach: safe vector access through MmapVectorReader
    std::shared_ptr<const MmapVectorReader> reader;   
    std::vector<uint32_t> id_mapping;   

    AlignedVector32<float> opt_graph_data;

    std::vector<uint32_t> entry_points;   
    AlignedVector32<float> entry_points_data;   

    size_t dim;   
    size_t num_vectors;   

    /**
     * @brief Get vector pointer by local ID (inline, zero-copy)
     */
    inline const float* get_vector(uint32_t local_id) const {
        uint32_t global_id = id_mapping[local_id];
        return reader->get_vector(global_id);
    }
};

/**
 * @brief Time measurement structure for each phase
 */
struct PhaseTimings {
    double phase1_subgraph_selection_ms;
    double phase2_entry_point_selection_ms;
    double phase3_nsg_search_ms;
    double total_time_ms;
};

/**
 * @brief Per-query performance information
 */
struct QueryPerformance {
    PhaseTimings timings;
    uint64_t search_dist_count;
    std::vector<uint32_t> result_global_ids;   // Result global IDs
};

/**
 * @brief Load .fvecs file
 */
std::vector<float> load_fvecs(const std::string& filename, size_t& num_vectors, size_t& dim);

/**
 * @brief Load Ground Truth (.ivecs format)
 */
std::vector<uint32_t> load_ground_truth(const std::string& filename, size_t& num_queries, size_t& dim);

/**
 * @brief Convert label to category name (e.g., "2 outdoor night" -> "2_outdoor_night")
 */
std::string convert_label_to_category(const std::string& label);

/**
 * @brief Load queries
 * @param category_to_id Category string to Integer ID mapping
 */
std::vector<Query> load_queries(const std::string& query_vec_file,
                                const std::string& query_label_file,
                                size_t dim,
                                const std::unordered_map<std::string, int>& category_to_id);

/**
 * @brief Load NSG subgraph (Global Pool + Mapping approach)
 */
NSGSubgraph load_nsg_subgraph(const std::string& category,
                              const std::string& graph_file,
                              const std::string& mapping_file,
                              const std::string& entry_points_file,
                              std::shared_ptr<const MmapVectorReader> reader,
                              size_t global_dim);

/**
 * @brief Save .idx file
 */
void save_label_indices(const char* filename, const std::vector<uint32_t>& indices);

/**
 * @brief Load .idx file
 */
std::vector<uint32_t> load_label_indices(const char* filename);

#endif // NSG_QLEAF_DATA_LOADER_H