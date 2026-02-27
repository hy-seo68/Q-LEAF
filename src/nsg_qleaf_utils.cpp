#include "nsg_qleaf_utils.h"
#include <fstream>
#include <stdexcept>
#include <algorithm>
#include <iostream>
#include <limits>
#include <set>
#include <faiss/utils/distances.h>  // fvec_L2sqr_ny_nearest, knn_L2sqr

std::vector<uint32_t> load_entry_points(const std::string& filename) {
    std::ifstream in(filename);
    if (!in.is_open()) {
        throw std::runtime_error("Cannot open entry points file: " + filename);
    }

    std::vector<uint32_t> entry_points;
    std::string line;

    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }

        try {
            uint32_t ep_id = std::stoul(line);
            entry_points.push_back(ep_id);
        } catch (const std::exception& e) {
            std::cerr << "Warning: Failed to parse line: " << line << std::endl;
            continue;
        }
    }

    in.close();

    if (entry_points.empty()) {
        throw std::runtime_error("No entry points loaded from file: " + filename);
    }

    std::cout << "Loaded " << entry_points.size() << " entry points from " << filename << std::endl;

    return entry_points;
}


std::pair<uint32_t, float> select_best_entry_point_fast(
    const float* query,
    const std::vector<uint32_t>& entry_points,
    const float* entry_points_data,
    size_t num_eps,
    size_t dim,
    uint64_t& ep_dist_counter,
    float* distances_buffer
) {
    if (entry_points.empty() || num_eps == 0) {
        throw std::runtime_error("Entry points vector is empty");
    }

    size_t nearest_idx = faiss::fvec_L2sqr_ny_nearest(
        distances_buffer,
        query,
        entry_points_data,
        dim,
        num_eps
    );

    ep_dist_counter += num_eps;

    float min_distance_squared = distances_buffer[nearest_idx];

    uint32_t best_ep_id = entry_points[nearest_idx];

    return {best_ep_id, min_distance_squared};
}


std::vector<std::pair<uint32_t, float>> select_top_entry_points_fast(
    const float* query,
    const std::vector<uint32_t>& entry_points,
    const float* entry_points_data,
    size_t num_eps,
    size_t dim,
    int num_selected,
    uint64_t& ep_dist_counter,
    float* distances_buffer,
    int64_t* indexes_buffer
) {
    if (entry_points.empty() || num_eps == 0) {
        throw std::runtime_error("Entry points vector is empty");
    }

    // Determine actual number to select
    int actual_num_selected = std::min(num_selected, static_cast<int>(num_eps));

    faiss::knn_L2sqr(
        query,                  
        entry_points_data,      
        dim,                    
        1,                      
        num_eps,                
        actual_num_selected,    
        distances_buffer,       
        indexes_buffer          
    );

    ep_dist_counter += num_eps;

    std::vector<std::pair<uint32_t, float>> selected;
    selected.reserve(actual_num_selected);

    for (int i = 0; i < actual_num_selected; i++) {
        int64_t buffer_idx = indexes_buffer[i];
        uint32_t original_ep_id = entry_points[buffer_idx];
        float distance = distances_buffer[i];
        selected.push_back({original_ep_id, distance});
    }

    return selected;
}

/**
 * @brief Calculate Recall
 */
float calculate_recall(const std::vector<std::vector<uint32_t>>& results,
                      const std::vector<std::vector<uint32_t>>& id_mappings,
                      const std::vector<uint32_t>& ground_truth,
                      size_t gt_dim, int k) {
    int total_correct = 0;
    int denominator_per_query = std::min(k, static_cast<int>(gt_dim));

    for (size_t i = 0; i < results.size(); i++) {
        const auto& result = results[i];
        const auto& mapping = id_mappings[i];

        if (result.empty()) {
            continue;
        }

        std::set<uint32_t> retrieved_set;
        for (int j = 0; j < k && j < static_cast<int>(result.size()); j++) {
            uint32_t local_id = result[j];
            if (local_id < mapping.size()) {
                retrieved_set.insert(mapping[local_id]);
            }
        }

        // Create Ground Truth set (use only top k)
        std::set<uint32_t> gt_set;
        size_t gt_k = std::min(static_cast<size_t>(k), gt_dim);
        for (size_t gt_idx = 0; gt_idx < gt_k; gt_idx++) {
            gt_set.insert(ground_truth[i * gt_dim + gt_idx]);
        }

        std::vector<uint32_t> intersection;
        std::set_intersection(retrieved_set.begin(), retrieved_set.end(),
                             gt_set.begin(), gt_set.end(),
                             std::back_inserter(intersection));

        total_correct += intersection.size();
    }

    int total_denominator = results.size() * denominator_per_query;
    return total_denominator > 0 ? static_cast<float>(total_correct) / total_denominator : 0.0f;
}

/**
 * @brief Calculate Recall (optimized version - Flat Array + Pointer Array)
 */
float calculate_recall_flat(
    const uint32_t* results_flat,
    const std::vector<const std::vector<uint32_t>*>& id_mapping_ptrs,
    size_t num_queries,
    const std::vector<uint32_t>& ground_truth,
    size_t gt_dim, int k) {

    int total_correct = 0;
    int denominator_per_query = std::min(k, static_cast<int>(gt_dim));

    for (size_t i = 0; i < num_queries; i++) {
        const uint32_t* result = results_flat + i * k;
        const std::vector<uint32_t>* mapping_ptr = id_mapping_ptrs[i];

        if (mapping_ptr == nullptr) {
            continue;
        }

        const std::vector<uint32_t>& mapping = *mapping_ptr;

        // Convert Retrieved Set to Global ID and create set
        std::set<uint32_t> retrieved_set;
        for (int j = 0; j < k; j++) {
            uint32_t local_id = result[j];
            if (local_id < mapping.size()) {
                retrieved_set.insert(mapping[local_id]);
            }
        }

        // Create Ground Truth set (use only top k)
        std::set<uint32_t> gt_set;
        size_t gt_k = std::min(static_cast<size_t>(k), gt_dim);
        for (size_t gt_idx = 0; gt_idx < gt_k; gt_idx++) {
            gt_set.insert(ground_truth[i * gt_dim + gt_idx]);
        }

        // Calculate intersection
        std::vector<uint32_t> intersection;
        std::set_intersection(retrieved_set.begin(), retrieved_set.end(),
                             gt_set.begin(), gt_set.end(),
                             std::back_inserter(intersection));

        total_correct += intersection.size();
    }

    int total_denominator = num_queries * denominator_per_query;
    return total_denominator > 0 ? static_cast<float>(total_correct) / total_denominator : 0.0f;
}