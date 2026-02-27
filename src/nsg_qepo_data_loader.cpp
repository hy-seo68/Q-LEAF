#include "nsg_qepo_data_loader.h"
#include "nsg_qepo_utils.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <set>
#include <iomanip>
#include <chrono>
#include <cstring>
#include <stdexcept>
#include <cerrno>

// EFANNA NSG headers
#include <efanna2e/util.h>

/**
 * @brief Load .fvecs file (optimized: large buffer read + memcpy)
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

    size_t record_size = (dim + 1) * sizeof(float);
    num_vectors = file_size / record_size;

    in.seekg(0, std::ios::beg);
    std::vector<char> raw_buffer(file_size);
    in.read(raw_buffer.data(), file_size);
    in.close();

    if (!in.good() && !in.eof()) {
        throw std::runtime_error("Failed to read fvecs file: " + filename);
    }

    std::vector<float> data(num_vectors * dim);
    const char* src = raw_buffer.data();

    for (size_t i = 0; i < num_vectors; i++) {
        src += sizeof(int);
        std::memcpy(data.data() + i * dim, src, dim * sizeof(float));
        src += dim * sizeof(float);
    }

    return data;
}


/**
 * @brief Load Ground Truth (optimized: large buffer read + memcpy)
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

    size_t record_size = (dim + 1) * sizeof(int);
    num_queries = file_size / record_size;

    in.seekg(0, std::ios::beg);
    std::vector<char> raw_buffer(file_size);
    in.read(raw_buffer.data(), file_size);
    in.close();

    if (!in.good() && !in.eof()) {
        throw std::runtime_error("Failed to read ground truth file: " + filename);
    }

    std::vector<uint32_t> data(num_queries * dim);
    const char* src = raw_buffer.data();

    for (size_t i = 0; i < num_queries; i++) {
        src += sizeof(int);
        std::memcpy(data.data() + i * dim, src, dim * sizeof(int));
        src += dim * sizeof(int);
    }

    return data;
}

/**
 * @brief Convert label to category name (e.g., "2 outdoor night" -> "2_outdoor_night")
 */
std::string convert_label_to_category(const std::string& label) {
    std::string category = label;
    std::replace(category.begin(), category.end(), ' ', '_');
    return category;
}


std::vector<Query> load_queries(const std::string& query_vec_file,
                                const std::string& query_label_file,
                                size_t dim,
                                const std::unordered_map<std::string, int>& category_to_id) {
    // Load query vectors
    size_t num_queries, query_dim;
    auto query_vectors = load_fvecs(query_vec_file, num_queries, query_dim);

    if (query_dim != dim) {
        throw std::runtime_error("Query dimension mismatch");
    }

    // Load query labels
    std::ifstream label_in(query_label_file);
    if (!label_in.is_open()) {
        throw std::runtime_error("Cannot open query label file");
    }

    std::vector<std::string> labels;
    std::string line;

    // Skip header
    if (std::getline(label_in, line)) {}

    while (std::getline(label_in, line)) {
        if (!line.empty()) {
            labels.push_back(line);
        }
    }

    if (labels.size() != num_queries) {
        throw std::runtime_error("Mismatch between query vectors and labels");
    }

    // Create Query structures
    std::vector<Query> queries;
    queries.reserve(num_queries);

    for (size_t i = 0; i < num_queries; i++) {
        Query q;
        q.vector.assign(query_vectors.begin() + i * dim,
                       query_vectors.begin() + (i + 1) * dim);
        q.category = convert_label_to_category(labels[i]);

        auto it = category_to_id.find(q.category);
        if (it != category_to_id.end()) {
            q.category_id = it->second;
        } else {
            throw std::runtime_error("Unknown category: " + q.category);
        }

        queries.push_back(q);
    }

    return queries;
}

/**
 * @brief Load NSG subgraph (Global Pool + Mapping method)
 */
NSGSubgraph load_nsg_subgraph(const std::string& category,
                              const std::string& graph_file,
                              const std::string& mapping_file,
                              const std::string& entry_points_file,
                              std::shared_ptr<const MmapVectorReader> reader,
                              size_t global_dim) {
    NSGSubgraph subgraph;
    subgraph.category = category;
    subgraph.dim = global_dim;
    subgraph.reader = reader;

    std::cout << "Loading NSG subgraph (Global Pool mode): " << category << std::endl;

    // 1. Load mapping file (local ID -> global ID)
    subgraph.id_mapping = load_label_indices(mapping_file.c_str());
    subgraph.num_vectors = subgraph.id_mapping.size();
    std::cout << "  - Loaded mapping: " << subgraph.num_vectors << " local IDs → global IDs"
              << " (" << (subgraph.num_vectors * sizeof(uint32_t) / 1024.0) << " KB)" << std::endl;

    // 2. Load NSG index
    subgraph.index = std::make_shared<efanna2e::IndexNSG>(subgraph.dim, subgraph.num_vectors,
                                                          efanna2e::L2, nullptr);
    subgraph.index->Load(graph_file.c_str());
    std::cout << "  - Loaded NSG graph from " << graph_file << std::endl;

    // 3. Call OptimizeGraph - Copy subgraph vectors from Global Pool to contiguous memory
    subgraph.opt_graph_data.resize(subgraph.num_vectors * subgraph.dim);
    for (size_t local_id = 0; local_id < subgraph.num_vectors; local_id++) {
        uint32_t global_id = subgraph.id_mapping[local_id];
        const float* src = reader->get_vector(global_id); // MmapVectorReader computes correct offset
        float* dst = subgraph.opt_graph_data.data() + local_id * subgraph.dim;
        std::memcpy(dst, src, subgraph.dim * sizeof(float));
    }
    subgraph.index->OptimizeGraph(subgraph.opt_graph_data.data());
    std::cout << "  - Optimized graph for fast search (data persisted in subgraph)" << std::endl;

    // 4. Load Entry Points
    subgraph.entry_points = load_entry_points(entry_points_file);
    std::cout << "  - Loaded " << subgraph.entry_points.size() << " entry points" << std::endl;

    // 5. Copy Entry Point vectors to contiguous memory (Cache-friendly optimization)
    size_t num_eps = subgraph.entry_points.size();
    subgraph.entry_points_data.resize(num_eps * subgraph.dim);
    for (size_t i = 0; i < num_eps; i++) {
        int64_t local_ep_id = subgraph.entry_points[i];
        uint32_t global_id = subgraph.id_mapping[local_ep_id];
        const float* src = reader->get_vector(global_id); // MmapVectorReader computes correct offset
        float* dst = subgraph.entry_points_data.data() + i * subgraph.dim;
        std::memcpy(dst, src, subgraph.dim * sizeof(float));
    }
    std::cout << "  - Copied " << num_eps << " entry point vectors to contiguous buffer ("
              << (num_eps * subgraph.dim * sizeof(float) / 1024.0) << " KB)" << std::endl;

    // Calculate memory usage
    size_t mapping_bytes = subgraph.num_vectors * sizeof(uint32_t);
    size_t opt_graph_bytes = subgraph.opt_graph_data.size() * sizeof(float);
    size_t entry_points_bytes = num_eps * subgraph.dim * sizeof(float);
    size_t total_bytes = mapping_bytes + opt_graph_bytes + entry_points_bytes;

    std::cout << "  - Memory usage breakdown:" << std::endl;
    std::cout << "    - Mapping: " << (mapping_bytes / 1024.0 / 1024.0) << " MB" << std::endl;
    std::cout << "    - OptimizeGraph data: " << (opt_graph_bytes / 1024.0 / 1024.0) << " MB" << std::endl;
    std::cout << "    - Entry points: " << (entry_points_bytes / 1024.0) << " KB" << std::endl;
    std::cout << "  - Total subgraph memory: " << (total_bytes / 1024.0 / 1024.0) << " MB" << std::endl;

    return subgraph;
}


MmapVectorReader::MmapVectorReader(const char* filename)
    : fd_(-1), mmap_addr_(nullptr), file_size_(0), num_vectors_(0), dim_(0), record_size_(0) {

    // 1. Open file (read-only)
    fd_ = open(filename, O_RDONLY);
    if (fd_ < 0) {
        throw std::runtime_error("Cannot open file: " + std::string(filename) +
                               " (errno: " + std::to_string(errno) + ")");
    }

    // 2. Check file size
    struct stat sb;
    if (fstat(fd_, &sb) < 0) {
        close(fd_);
        throw std::runtime_error("Cannot stat file: " + std::string(filename));
    }
    file_size_ = sb.st_size;

    if (file_size_ == 0) {
        close(fd_);
        throw std::runtime_error("File is empty: " + std::string(filename));
    }

    // 3. Map the entire file to memory using mmap
    mmap_addr_ = mmap(nullptr, file_size_, PROT_READ, MAP_SHARED, fd_, 0);
    if (mmap_addr_ == MAP_FAILED) {
        close(fd_);
        throw std::runtime_error("mmap failed for file: " + std::string(filename) +
                               " (errno: " + std::to_string(errno) +
                               "). Check memory availability.");
    }

    // 4. Validate .fvecs file format and read dimension
    if (file_size_ < sizeof(int)) {
        munmap(mmap_addr_, file_size_);
        close(fd_);
        throw std::runtime_error("File too small to contain dimension: " +
                               std::string(filename));
    }

    dim_ = *static_cast<int*>(mmap_addr_);
    if (dim_ <= 0 || dim_ > 10000) {
        munmap(mmap_addr_, file_size_);
        close(fd_);
        throw std::runtime_error("Invalid dimension: " + std::to_string(dim_) +
                               ". File may be corrupted or not in .fvecs format.");
    }

    // 5. Calculate record size and number of vectors
    record_size_ = (dim_ + 1) * sizeof(float); // dimension + vector

    if (file_size_ % record_size_ != 0) {
        munmap(mmap_addr_, file_size_);
        close(fd_);
        std::ostringstream oss;
        oss << "File size mismatch!\n"
            << "  File: " << filename << "\n"
            << "  File size: " << file_size_ << " bytes\n"
            << "  Expected record size: " << record_size_ << " bytes\n"
            << "  Dimension: " << dim_ << "\n"
            << "  Remainder: " << (file_size_ % record_size_) << " bytes\n"
            << "  File may be corrupted or incomplete.";
        throw std::runtime_error(oss.str());
    }

    num_vectors_ = file_size_ / record_size_;

    // 6. Set default access pattern hint (MADV_RANDOM)
    madvise(mmap_addr_, file_size_, MADV_RANDOM);

    std::cout << "[MmapVectorReader] Loaded file: " << filename << std::endl;
    std::cout << "  - Vectors: " << num_vectors_ << std::endl;
    std::cout << "  - Dimension: " << dim_ << std::endl;
    std::cout << "  - File size: " << (file_size_ / 1024.0 / 1024.0) << " MB" << std::endl;
    std::cout << "  - Mapped with MAP_SHARED (page cache shared)" << std::endl;
}

MmapVectorReader::~MmapVectorReader() {
    if (mmap_addr_ != nullptr && mmap_addr_ != MAP_FAILED) {
        munmap(mmap_addr_, file_size_);
    }
    if (fd_ >= 0) {
        close(fd_);
    }
}

void MmapVectorReader::set_access_pattern(int advice) {
    if (mmap_addr_ != nullptr && mmap_addr_ != MAP_FAILED) {
        if (madvise(mmap_addr_, file_size_, advice) != 0) {
            std::cerr << "Warning: madvise failed (errno: " << errno << ")" << std::endl;
        } else {
            const char* advice_str = "UNKNOWN";
            switch (advice) {
                case MADV_NORMAL: advice_str = "NORMAL"; break;
                case MADV_RANDOM: advice_str = "RANDOM"; break;
                case MADV_SEQUENTIAL: advice_str = "SEQUENTIAL"; break;
                case MADV_WILLNEED: advice_str = "WILLNEED"; break;
                case MADV_DONTNEED: advice_str = "DONTNEED"; break;
            }
            std::cout << "[MmapVectorReader] Access pattern set to: " << advice_str << std::endl;
        }
    }
}


void save_label_indices(const char* filename, const std::vector<uint32_t>& indices) {
    std::ofstream out(filename, std::ios::binary);
    if (!out.is_open()) {
        throw std::runtime_error("Cannot open file for writing: " + std::string(filename));
    }

    uint32_t num_indices = static_cast<uint32_t>(indices.size());
    out.write(reinterpret_cast<const char*>(&num_indices), sizeof(uint32_t));

    if (!out.good()) {
        throw std::runtime_error("Failed to write header to file: " + std::string(filename));
    }

    out.write(reinterpret_cast<const char*>(indices.data()),
              num_indices * sizeof(uint32_t));

    if (!out.good()) {
        throw std::runtime_error("Failed to write indices to file: " + std::string(filename));
    }

    out.close();

    size_t file_size_kb = (sizeof(uint32_t) + num_indices * sizeof(uint32_t)) / 1024;
    std::cout << "[save_label_indices] Saved " << num_indices << " indices to "
              << filename << " (" << file_size_kb << " KB)" << std::endl;
}

std::vector<uint32_t> load_label_indices(const char* filename) {
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) {
        throw std::runtime_error("Cannot open .idx file: " + std::string(filename));
    }

    in.seekg(0, std::ios::end);
    size_t file_size = in.tellg();
    in.seekg(0, std::ios::beg);

    if (file_size < sizeof(uint32_t)) {
        throw std::runtime_error("File too small to contain header: " +
                               std::string(filename) + " (" +
                               std::to_string(file_size) + " bytes)");
    }

    uint32_t num_indices;
    in.read(reinterpret_cast<char*>(&num_indices), sizeof(uint32_t));

    if (!in.good()) {
        throw std::runtime_error("Failed to read header from file: " +
                               std::string(filename));
    }

    size_t expected_size = sizeof(uint32_t) + num_indices * sizeof(uint32_t);
    if (file_size != expected_size) {
        std::ostringstream oss;
        oss << "File size mismatch!\n"
            << "  File: " << filename << "\n"
            << "  Actual size: " << file_size << " bytes\n"
            << "  Expected size: " << expected_size << " bytes\n"
            << "  Num indices: " << num_indices << "\n"
            << "  File may be corrupted.";
        throw std::runtime_error(oss.str());
    }

    std::vector<uint32_t> indices(num_indices);
    in.read(reinterpret_cast<char*>(indices.data()), num_indices * sizeof(uint32_t));

    if (!in.good()) {
        throw std::runtime_error("Failed to read indices from file: " +
                               std::string(filename));
    }

    in.close();

    std::cout << "[load_label_indices] Loaded " << num_indices << " indices from "
              << filename << std::endl;

    return indices;
}