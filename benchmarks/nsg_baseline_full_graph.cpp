/**
 * @file nsg_baseline_full_graph.cpp
 * @brief NSG Baseline Benchmark (Case A: Full Graph + Post-filtering)
 *
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <memory>
#include <cstring>
#include <stdexcept>

// EFANNA NSG headers
#include <efanna2e/index_nsg.h>
#include <efanna2e/util.h>

/**
 * @brief Query information structure
 */
struct Query {
    std::vector<float> vector;
    std::string label;
};

/**
 * @brief Phase-wise time measurement structure
 */
struct PhaseTimings {
    double phase1_nsg_search_ms;
    double phase2_post_filtering_ms;
    double total_time_ms;
};

/**
 * @brief Per-query performance information
 */
struct QueryPerformance {
    PhaseTimings timings;
    uint64_t search_dist_count;
    int pre_filter_results;
    int post_filter_results;
};

/**
 * @brief Single search_L experiment result structure
 */
struct ExperimentResult {
    int search_L;
    size_t total_queries;
    double wall_clock_time_seconds;
    double pure_algorithm_time_seconds;
    double wall_clock_qps;
    double pure_algorithm_qps;
    double overhead_percentage;
    float recall;
    uint64_t search_distance_calculations;
    double avg_phase1_ms;
    double avg_phase2_ms;
    double phase1_percentage;
    double phase2_percentage;
    double avg_pre_filter_results;
    double avg_post_filter_results;
    double filtering_ratio;
};

/**
 * @brief Load .fvecs file
 */
std::vector<float> load_fvecs(const std::string& filename, size_t& num_vectors, size_t& dim) {
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) {
        throw std::runtime_error("Cannot open fvecs file: " + filename);
    }

    in.seekg(0, std::ios::end);
    size_t file_size = in.tellg();
    in.seekg(0, std::ios::beg);

    int d;
    in.read(reinterpret_cast<char*>(&d), sizeof(int));
    dim = static_cast<size_t>(d);

    num_vectors = file_size / ((dim + 1) * sizeof(float));
    std::vector<float> data(num_vectors * dim);

    in.seekg(0, std::ios::beg);
    for (size_t i = 0; i < num_vectors; i++) {
        in.read(reinterpret_cast<char*>(&d), sizeof(int));
        in.read(reinterpret_cast<char*>(&data[i * dim]), dim * sizeof(float));
    }

    in.close();
    return data;
}

/**
 * @brief Load label file (text format)
 */
std::vector<std::string> load_labels(const std::string& filename) {
    std::ifstream in(filename);
    if (!in.is_open()) {
        throw std::runtime_error("Cannot open label file: " + filename);
    }

    std::vector<std::string> labels;
    std::string line;

    while (std::getline(in, line)) {
        if (!line.empty()) {
            labels.push_back(line);
        }
    }

    in.close();
    return labels;
}

/**
 * @brief Load queries
 */
std::vector<Query> load_queries(const std::string& query_vec_file,
                                const std::string& query_label_file,
                                size_t dim) {
    size_t num_queries, query_dim;
    auto query_vectors = load_fvecs(query_vec_file, num_queries, query_dim);

    if (query_dim != dim) {
        throw std::runtime_error("Query dimension mismatch: expected " +
                                std::to_string(dim) + ", got " + std::to_string(query_dim));
    }

    auto labels = load_labels(query_label_file);

    if (labels.size() != num_queries) {
        std::cerr << "Warning: Label count (" << labels.size()
                  << ") differs from query count (" << num_queries << ")" << std::endl;
    }

    std::vector<Query> queries;
    queries.reserve(num_queries);

    for (size_t i = 0; i < num_queries; i++) {
        Query q;
        q.vector.assign(query_vectors.begin() + i * dim,
                       query_vectors.begin() + (i + 1) * dim);
        q.label = (i < labels.size()) ? labels[i] : "";
        queries.push_back(q);
    }

    return queries;
}

/**
 * @brief Load Ground Truth (.ivecs format)
 */
std::vector<uint32_t> load_ground_truth(const std::string& filename, size_t& num_queries, size_t& dim) {
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) {
        throw std::runtime_error("Cannot open ground truth file: " + filename);
    }

    in.seekg(0, std::ios::end);
    size_t file_size = in.tellg();
    in.seekg(0, std::ios::beg);

    int d;
    in.read(reinterpret_cast<char*>(&d), sizeof(int));
    dim = static_cast<size_t>(d);

    num_queries = file_size / ((dim + 1) * sizeof(int));
    std::vector<uint32_t> data(num_queries * dim);

    in.seekg(0, std::ios::beg);
    for (size_t i = 0; i < num_queries; i++) {
        in.read(reinterpret_cast<char*>(&d), sizeof(int));
        in.read(reinterpret_cast<char*>(&data[i * dim]), dim * sizeof(int));
    }

    in.close();
    return data;
}

/**
 * @brief Calculate Recall with Post-filtering
 */
float calculate_recall_with_postfilter(
    const std::vector<std::vector<uint32_t>>& all_results,
    const std::vector<std::string>& query_labels,
    const std::vector<std::string>& base_labels,
    const std::vector<uint32_t>& ground_truth,
    size_t gt_dim,
    int k
) {
    int total_correct = 0;
    int total_comparisons = 0;

    for (size_t i = 0; i < all_results.size(); i++) {
        const auto& raw_results = all_results[i];
        const std::string& query_label = query_labels[i];

        std::vector<uint32_t> filtered_results;
        for (uint32_t result_id : raw_results) {
            if (result_id < base_labels.size() && base_labels[result_id] == query_label) {
                filtered_results.push_back(result_id);
                if (filtered_results.size() >= static_cast<size_t>(k)) {
                    break;
                }
            }
        }

        for (size_t j = 0; j < filtered_results.size() && j < static_cast<size_t>(k); j++) {
            uint32_t result_id = filtered_results[j];

            for (size_t gt_idx = 0; gt_idx < gt_dim && gt_idx < static_cast<size_t>(k); gt_idx++) {
                if (ground_truth[i * gt_dim + gt_idx] == result_id) {
                    total_correct++;
                    break;
                }
            }
            total_comparisons++;
        }
    }

    return total_comparisons > 0 ? static_cast<float>(total_correct) / total_comparisons : 0.0f;
}

/**
 * @brief Parse search_L values function
 */
std::vector<int> parse_search_L_values(const std::string& search_L_str) {
    std::vector<int> search_L_values;
    std::stringstream ss(search_L_str);
    std::string token;

    while (std::getline(ss, token, ',')) {
        size_t start = token.find_first_not_of(" \t\r\n");
        size_t end = token.find_last_not_of(" \t\r\n");
        if (start != std::string::npos && end != std::string::npos) {
            token = token.substr(start, end - start + 1);
        }

        try {
            int value = std::stoi(token);
            if (value > 0) {
                search_L_values.push_back(value);
            }
        } catch (const std::exception& e) {
            std::cerr << "Warning: Invalid search_L value: " << token << std::endl;
        }
    }

    return search_L_values;
}

/**
 * @brief Save all experiment results to a single JSON file
 */
void save_all_results_json(
    const std::string& output_file,
    int k,
    const std::vector<ExperimentResult>& results) {

    std::ofstream out(output_file);
    if (!out.is_open()) {
        std::cerr << "Warning: Cannot save results to " << output_file << std::endl;
        return;
    }

    auto now = std::chrono::system_clock::now();
    auto time_t_now = std::chrono::system_clock::to_time_t(now);
    auto tm_now = *std::localtime(&time_t_now);

    out << "{\n";
    out << "  \"timestamp\": \"" << time_t_now << "\",\n";
    out << "  \"datetime\": \"" << std::put_time(&tm_now, "%Y-%m-%d %H:%M:%S") << "\",\n";
    out << "  \"framework\": \"NSG_Baseline_FullGraph\",\n";
    out << "  \"variant\": \"case_a_postfilter\",\n";
    out << "  \"description\": \"Case A: Full graph with NSG default EP and post-filtering (Pure Baseline)\",\n";
    out << "  \"ablation_case\": \"A\",\n";
    out << "  \"entry_point_optimization\": false,\n";
    out << "  \"subgraph_partitioning\": false,\n";
    out << "  \"parameters\": {\n";
    out << "    \"k\": " << k << ",\n";
    out << "    \"search_L_values\": [";
    for (size_t i = 0; i < results.size(); i++) {
        out << results[i].search_L;
        if (i < results.size() - 1) out << ", ";
    }
    out << "]\n";
    out << "  },\n";
    out << "  \"experiments\": [\n";

    for (size_t exp_idx = 0; exp_idx < results.size(); exp_idx++) {
        const auto& res = results[exp_idx];

        out << "    {\n";
        out << "      \"search_L\": " << res.search_L << ",\n";
        out << "      \"results\": {\n";
        out << "        \"total_queries\": " << res.total_queries << ",\n";
        out << "        \"wall_clock_time_seconds\": " << res.wall_clock_time_seconds << ",\n";
        out << "        \"pure_algorithm_time_seconds\": " << res.pure_algorithm_time_seconds << ",\n";
        out << "        \"wall_clock_qps\": " << res.wall_clock_qps << ",\n";
        out << "        \"pure_algorithm_qps\": " << res.pure_algorithm_qps << ",\n";
        out << "        \"overhead_percentage\": " << res.overhead_percentage << ",\n";
        out << "        \"recall@" << k << "\": " << res.recall << "\n";
        out << "      },\n";
        out << "      \"filtering_stats\": {\n";
        out << "        \"avg_pre_filter_results\": " << res.avg_pre_filter_results << ",\n";
        out << "        \"avg_post_filter_results\": " << res.avg_post_filter_results << ",\n";
        out << "        \"filtering_ratio\": " << res.filtering_ratio << "\n";
        out << "      },\n";
        out << "      \"efficiency_breakdown\": {\n";
        out << "        \"search_distance_calculations\": " << res.search_distance_calculations << ",\n";
        out << "        \"avg_search_distance_per_query\": "
            << static_cast<double>(res.search_distance_calculations) / res.total_queries << "\n";
        out << "      },\n";
        out << "      \"phase_timing\": {\n";
        out << "        \"avg_phase1_nsg_search_ms\": " << res.avg_phase1_ms << ",\n";
        out << "        \"avg_phase2_post_filtering_ms\": " << res.avg_phase2_ms << ",\n";
        out << "        \"phase1_percentage\": " << res.phase1_percentage << ",\n";
        out << "        \"phase2_percentage\": " << res.phase2_percentage << "\n";
        out << "      }\n";
        out << "    }";
        if (exp_idx < results.size() - 1) out << ",";
        out << "\n";
    }

    out << "  ]\n";
    out << "}\n";

    out.close();
    std::cout << "\nAll results saved to: " << output_file << std::endl;
}

/**
 * @brief Main function
 */
int main(int argc, char** argv) {
    if (argc < 8) {
        std::cerr << "========================================" << std::endl;
        std::cerr << "NSG Baseline Full Graph Benchmark (Case A)" << std::endl;
        std::cerr << "========================================" << std::endl;
        std::cerr << "Usage: " << argv[0]
                  << " <nsg_graph_file> <base_data_file> <base_label_file>"
                  << " <query_vec_file> <query_label_file> <k> <search_L_values> <ground_truth_file>"
                  << std::endl;
        std::cerr << std::endl;
        std::cerr << "Arguments:" << std::endl;
        std::cerr << "  nsg_graph_file     : Full NSG graph file (.graph)" << std::endl;
        std::cerr << "  base_data_file     : Base vectors (.fvecs)" << std::endl;
        std::cerr << "  base_label_file    : Base labels (.txt)" << std::endl;
        std::cerr << "  query_vec_file     : Query vectors (.fvecs)" << std::endl;
        std::cerr << "  query_label_file   : Query labels (.txt)" << std::endl;
        std::cerr << "  k                  : Number of nearest neighbors" << std::endl;
        std::cerr << "  search_L_values    : Comma-separated search_L values (e.g., \"50,100,150,200\")" << std::endl;
        std::cerr << "  ground_truth_file  : Ground truth (.ivecs)" << std::endl;
        std::cerr << std::endl;
        std::cerr << "Note: This is Case A (Pure Baseline) for Ablation Study." << std::endl;
        std::cerr << "      - Uses full graph (no subgraph partitioning)" << std::endl;
        std::cerr << "      - Uses NSG default entry point (no Q-leaf EP optimization)" << std::endl;
        std::cerr << "      - Uses post-filtering for label constraints" << std::endl;
        std::cerr << "========================================" << std::endl;
        return -1;
    }

    std::string nsg_graph_file = argv[1];
    std::string base_data_file = argv[2];
    std::string base_label_file = argv[3];
    std::string query_vec_file = argv[4];
    std::string query_label_file = argv[5];
    int k = std::stoi(argv[6]);
    std::string search_L_str = argv[7];
    std::string ground_truth_file = argv[8];

    try {
        // Parse search_L values
        std::vector<int> search_L_values = parse_search_L_values(search_L_str);

        if (search_L_values.empty()) {
            throw std::runtime_error("No valid search_L values provided");
        }

        std::cout << "\n========================================" << std::endl;
        std::cout << "NSG Baseline Full Graph Benchmark (Case A)" << std::endl;
        std::cout << "Ablation Study: Pure Baseline" << std::endl;
        std::cout << "========================================" << std::endl;
        std::cout << "Entry Point Optimization: DISABLED (NSG default EP)" << std::endl;
        std::cout << "Subgraph Partitioning: DISABLED (Full graph)" << std::endl;
        std::cout << "Label Filtering: Post-filtering" << std::endl;
        std::cout << "========================================" << std::endl;
        std::cout << "k=" << k << ", search_L values: ";
        for (size_t i = 0; i < search_L_values.size(); i++) {
            std::cout << search_L_values[i];
            if (i < search_L_values.size() - 1) std::cout << ", ";
        }
        std::cout << std::endl;
        std::cout << "Number of experiments: " << search_L_values.size() << std::endl;
        std::cout << "========================================\n" << std::endl;

        // Load base data
        std::cout << "=== Loading Base Data ===" << std::endl;
        size_t num_vectors, dim;
        auto base_data = load_fvecs(base_data_file, num_vectors, dim);
        std::cout << "Loaded " << num_vectors << " vectors, dim=" << dim << std::endl;

        // Load base labels
        std::cout << "\n=== Loading Base Labels ===" << std::endl;
        auto base_labels = load_labels(base_label_file);
        std::cout << "Loaded " << base_labels.size() << " labels" << std::endl;

        if (base_labels.size() != num_vectors) {
            std::cerr << "Warning: Label count (" << base_labels.size()
                      << ") differs from vector count (" << num_vectors << ")" << std::endl;
        }

        // Load NSG graph
        std::cout << "\n=== Loading NSG Graph ===" << std::endl;
        auto index = std::make_shared<efanna2e::IndexNSG>(dim, num_vectors, efanna2e::L2, nullptr);
        index->Load(nsg_graph_file.c_str());
        std::cout << "Loaded NSG graph from " << nsg_graph_file << std::endl;

        // Call OptimizeGraph
        index->OptimizeGraph(base_data.data());
        std::cout << "Optimized graph for fast search" << std::endl;

        // Load queries
        std::cout << "\n=== Loading Queries ===" << std::endl;
        auto queries = load_queries(query_vec_file, query_label_file, dim);
        std::cout << "Loaded " << queries.size() << " queries" << std::endl;

        // Load Ground Truth
        std::cout << "\n=== Loading Ground Truth ===" << std::endl;
        size_t gt_num, gt_dim;
        auto ground_truth = load_ground_truth(ground_truth_file, gt_num, gt_dim);
        std::cout << "Loaded ground truth: " << gt_num << " queries, " << gt_dim << " neighbors" << std::endl;

        // Warm-up
        std::cout << "\n=== Warm-up ===" << std::endl;
        efanna2e::Parameters warmup_params;
        warmup_params.Set<unsigned>("L_search", search_L_values[0]);

        int warmup_count = std::min(100, static_cast<int>(queries.size()));
        std::cout << "Warm-up queries: " << warmup_count << " (using search_L=" << search_L_values[0] << ")" << std::endl;

        for (int i = 0; i < warmup_count; i++) {
            std::vector<unsigned> result(search_L_values[0]);
            // Case A: Use NSG default EP (no SetEntryPoint call)
            index->SearchWithOptGraph(queries[i].vector.data(), search_L_values[0], warmup_params, result.data());

            if ((i + 1) % 20 == 0) {
                std::cout << "  Warm-up progress: " << (i + 1) << " / " << warmup_count << std::endl;
            }
        }
        std::cout << "Warm-up completed" << std::endl;

        // Run benchmark for all search_L values
        std::vector<ExperimentResult> all_results;

        for (size_t exp_num = 0; exp_num < search_L_values.size(); exp_num++) {
            int search_L = search_L_values[exp_num];

            std::cout << "\n========================================" << std::endl;
            std::cout << "Experiment " << (exp_num + 1) << "/" << search_L_values.size()
                      << ": search_L = " << search_L << std::endl;
            std::cout << "========================================" << std::endl;

            ExperimentResult exp_result;
            exp_result.search_L = search_L;
            exp_result.total_queries = queries.size();

            // Vector for storing results
            std::vector<std::vector<uint32_t>> all_raw_results(queries.size());
            std::vector<std::string> query_labels;

            // For phase-wise time measurement
            std::vector<double> phase1_times(queries.size());
            std::vector<double> phase2_times(queries.size());
            std::vector<int> pre_filter_counts(queries.size());
            std::vector<int> post_filter_counts(queries.size());

            efanna2e::Parameters params;
            params.Set<unsigned>("L_search", search_L);

            // Initialize distance calculation counter
            uint64_t search_dist_before_loop = efanna2e::IndexNSG::distance_computation_counter;

            // Start wall-clock measurement
            auto benchmark_start = std::chrono::high_resolution_clock::now();

            for (size_t i = 0; i < queries.size(); i++) {
                const auto& q = queries[i];

                // Phase 1: NSG search (full graph, using default EP)
                auto phase1_start = std::chrono::high_resolution_clock::now();

                std::vector<unsigned> result(search_L);
                // Case A: Use NSG default Entry Point - no SetEntryPoint call!
                index->SearchWithOptGraph(q.vector.data(), search_L, params, result.data());

                auto phase1_end = std::chrono::high_resolution_clock::now();
                phase1_times[i] = std::chrono::duration<double, std::milli>(phase1_end - phase1_start).count();

                pre_filter_counts[i] = result.size();

                // Phase 2: Post-filtering (label constraints)
                auto phase2_start = std::chrono::high_resolution_clock::now();

                int post_count = 0;
                for (unsigned result_id : result) {
                    if (result_id < base_labels.size() && base_labels[result_id] == q.label) {
                        post_count++;
                    }
                }
                post_filter_counts[i] = post_count;

                auto phase2_end = std::chrono::high_resolution_clock::now();
                phase2_times[i] = std::chrono::duration<double, std::milli>(phase2_end - phase2_start).count();

                // Store results (for Recall calculation)
                all_raw_results[i].assign(result.begin(), result.end());
                query_labels.push_back(q.label);
            }

            auto benchmark_end = std::chrono::high_resolution_clock::now();

            // Read distance calculation counter
            uint64_t search_dist_after_loop = efanna2e::IndexNSG::distance_computation_counter;
            exp_result.search_distance_calculations = search_dist_after_loop - search_dist_before_loop;

            // Calculate time
            exp_result.wall_clock_time_seconds =
                std::chrono::duration<double>(benchmark_end - benchmark_start).count();

            exp_result.pure_algorithm_time_seconds = 0.0;
            for (size_t i = 0; i < queries.size(); i++) {
                exp_result.pure_algorithm_time_seconds +=
                    (phase1_times[i] + phase2_times[i]) / 1000.0;
            }

            exp_result.wall_clock_qps = queries.size() / exp_result.wall_clock_time_seconds;
            exp_result.pure_algorithm_qps = queries.size() / exp_result.pure_algorithm_time_seconds;
            exp_result.overhead_percentage =
                ((exp_result.wall_clock_time_seconds - exp_result.pure_algorithm_time_seconds) /
                 exp_result.wall_clock_time_seconds) * 100.0;

            // Calculate Recall
            exp_result.recall = calculate_recall_with_postfilter(
                all_raw_results, query_labels, base_labels,
                ground_truth, gt_dim, k);

            // Calculate average time per phase
            exp_result.avg_phase1_ms = 0.0;
            exp_result.avg_phase2_ms = 0.0;
            double total_pre_filter = 0.0;
            double total_post_filter = 0.0;

            for (size_t i = 0; i < queries.size(); i++) {
                exp_result.avg_phase1_ms += phase1_times[i];
                exp_result.avg_phase2_ms += phase2_times[i];
                total_pre_filter += pre_filter_counts[i];
                total_post_filter += post_filter_counts[i];
            }

            exp_result.avg_phase1_ms /= queries.size();
            exp_result.avg_phase2_ms /= queries.size();
            exp_result.avg_pre_filter_results = total_pre_filter / queries.size();
            exp_result.avg_post_filter_results = total_post_filter / queries.size();
            exp_result.filtering_ratio = (total_pre_filter > 0) ?
                (total_post_filter / total_pre_filter) : 0.0;

            double total_phase_time = exp_result.avg_phase1_ms + exp_result.avg_phase2_ms;
            exp_result.phase1_percentage = (exp_result.avg_phase1_ms / total_phase_time) * 100.0;
            exp_result.phase2_percentage = (exp_result.avg_phase2_ms / total_phase_time) * 100.0;

            // Output results
            std::cout << "search_L=" << search_L << " Results:" << std::endl;
            std::cout << "  Wall-clock time: " << exp_result.wall_clock_time_seconds << " seconds" << std::endl;
            std::cout << "  Pure algorithm time: " << exp_result.pure_algorithm_time_seconds << " seconds" << std::endl;
            std::cout << "  Wall-clock QPS: " << exp_result.wall_clock_qps << std::endl;
            std::cout << "  Pure algorithm QPS: " << exp_result.pure_algorithm_qps << std::endl;
            std::cout << "  Recall@" << k << ": " << exp_result.recall << std::endl;
            std::cout << "  Search distance calculations: " << exp_result.search_distance_calculations << std::endl;
            std::cout << "  Filtering ratio: " << (exp_result.filtering_ratio * 100) << "%" << std::endl;

            all_results.push_back(std::move(exp_result));
        }

        // Output results summary
        std::cout << "\n========================================" << std::endl;
        std::cout << "Summary of All Experiments (Case A Baseline)" << std::endl;
        std::cout << "========================================" << std::endl;
        std::cout << std::setw(10) << "search_L"
                  << std::setw(15) << "Recall@" + std::to_string(k)
                  << std::setw(15) << "Wall QPS"
                  << std::setw(15) << "Pure QPS"
                  << std::setw(15) << "Search Dist"
                  << std::setw(15) << "Filter Ratio"
                  << std::endl;
        std::cout << "--------------------------------------------------------------------------------" << std::endl;

        for (const auto& res : all_results) {
            std::cout << std::setw(10) << res.search_L
                      << std::setw(15) << std::fixed << std::setprecision(4) << res.recall
                      << std::setw(15) << std::fixed << std::setprecision(2) << res.wall_clock_qps
                      << std::setw(15) << std::fixed << std::setprecision(2) << res.pure_algorithm_qps
                      << std::setw(15) << res.search_distance_calculations
                      << std::setw(14) << std::fixed << std::setprecision(2) << (res.filtering_ratio * 100) << "%"
                      << std::endl;
        }

        // Save JSON results
        std::ostringstream timestamp_stream;
        auto now = std::chrono::system_clock::now();
        auto time_t_now = std::chrono::system_clock::to_time_t(now);
        auto tm_now = *std::localtime(&time_t_now);
        timestamp_stream << std::put_time(&tm_now, "%Y%m%d_%H%M%S");

        std::string output_file = "nsg_baseline_full_graph_k" + std::to_string(k) +
                                 "_multi_L_" + timestamp_stream.str() + ".json";

        save_all_results_json(output_file, k, all_results);

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}
