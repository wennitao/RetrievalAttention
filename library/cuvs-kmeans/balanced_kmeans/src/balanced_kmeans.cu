#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include <raft/core/handle.hpp>
#include <raft/core/operators.hpp>
#include <raft/random/make_blobs.cuh>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/util/cuda_utils.cuh>
#include <raft/core/resources.hpp>
#include <cuvs/cluster/kmeans.hpp>

#include <rmm/device_uvector.hpp>

#include <iostream>

using namespace cuvs::cluster;

class CuvsKmeans {
    public:
    CuvsKmeans(int64_t n_clusters, int64_t n_iters) 
        : stream(raft::resource::get_cuda_stream(handle)),
        n_clusters(n_clusters), n_iters(n_iters) {}

    void fit_predict(
        torch::Tensor key, // (n_samples, n_features)
        torch::Tensor value,  
        torch::Tensor labels, 
        torch::Tensor centroids 
    ) {
        int64_t n_samples = key.size(0), n_features = key.size(1);
        float inertia; 
        int n_iter;
        
        const float* key_ptr = key.data_ptr<float>();
        float* centroids_ptr = centroids.data_ptr<float>();
        int* labels_ptr = labels.data_ptr<int>();

        auto X_view = raft::make_device_matrix_view<const float, int>(key_ptr, n_samples, n_features);
        auto centroids_view = raft::make_device_matrix_view<float, int>(centroids_ptr, n_clusters, n_features);
        auto labels_view = raft::make_device_vector_view<int, int>(labels_ptr, n_samples);

        cuvs::cluster::kmeans::params params;
        params.n_clusters = n_clusters;
        params.max_iter = n_iters;

        cuvs::cluster::kmeans::fit_predict(handle, params, X_view, std::nullopt, centroids_view, labels_view, raft::make_host_scalar_view(&inertia), raft::make_host_scalar_view(&n_iter));
        raft::resource::sync_stream(handle, stream);
    }

    void balanced_fit_predict(
        torch::Tensor key, // (n_samples, n_features)
        torch::Tensor value, 
        torch::Tensor labels, 
        torch::Tensor centroids 
    ) {
        int64_t n_samples = key.size(0), n_features = key.size(1);
        int64_t n_clusters = centroids.size(0);

        const float* key_ptr = key.data_ptr<float>();
        float* centroids_ptr = centroids.data_ptr<float>();
        uint32_t* labels_ptr = labels.data_ptr<uint32_t>();

        auto X_view = raft::make_device_matrix_view<const float, int64_t>(key_ptr, n_samples, n_features);
        auto centroids_view = raft::make_device_matrix_view<float, int64_t>(centroids_ptr, n_clusters, n_features);
        auto labels_view = raft::make_device_vector_view<uint32_t, int64_t>(labels_ptr, n_samples);

        cuvs::cluster::kmeans::balanced_params params;
        // params.n_clusters = n_clusters;
        params.n_iters = 20;
        params.metric = cuvs::distance::DistanceType::InnerProduct;

        cuvs::cluster::kmeans::fit_predict(handle, params, X_view, centroids_view, labels_view);
        raft::resource::sync_stream(handle, stream);
    }

    private:
    raft::resources handle;
    cudaStream_t stream;
    int64_t n_clusters, n_iters;
};

void balanced_kmeans(int64_t seq_len) {
    raft::resources handle;
    cudaStream_t stream = raft::resource::get_cuda_stream(handle);

    int64_t n_features = 128, n_clusters = seq_len / 16;
    auto X = raft::make_device_matrix<float, int64_t>(handle, seq_len, n_features);
    auto labels = raft::make_device_vector<uint32_t, int64_t>(handle, seq_len);
    auto centroids = raft::make_device_matrix<float, int64_t>(handle, n_clusters, n_features);

    raft::random::make_blobs<float, uint32_t>(
        X.data_handle(),
        labels.data_handle(),
        seq_len, 
        n_features,
        n_clusters,
        stream, 
        true, 
        nullptr, 
        nullptr, 
        0.1, 
        true, 
        -1, 
        1, 
        1234
    );

    auto X_view = raft::make_device_matrix_view<const float, int64_t>(X.data_handle(), seq_len, n_features);
    auto centroids_view = raft::make_device_matrix_view<float, int64_t>(centroids.data_handle(), n_clusters, n_features);
    auto labels_view = raft::make_device_vector_view<uint32_t, int64_t>(labels.data_handle(), seq_len);

    cuvs::cluster::kmeans::balanced_params params;
    // params.n_clusters = n_clusters;
    params.n_iters = 100;
    params.metric = cuvs::distance::DistanceType::InnerProduct;
    
    cuvs::cluster::kmeans::fit_predict(handle, params, X_view, centroids_view, labels_view);
    raft::resource::sync_stream(handle, stream);

    // std::cout << raft::arr2Str(labels.data_handle(), seq_len, "labels", stream) << std::endl;

    uint32_t host_labels[seq_len];
    raft::copy(host_labels, labels.data_handle(), seq_len, stream);
    // assert that each cluster has roughly equal number of points
    int64_t counts[n_clusters];
    memset(counts, 0, sizeof(counts));
    for (int64_t i = 0; i < seq_len; i++) {
        counts[host_labels[i]]++;
    }
    int64_t expected_count = seq_len / n_clusters;
    for (int64_t i = 0; i < n_clusters; i++) {
        if (counts[i] != expected_count) {
            std::cout << "Cluster " << i << " has " << counts[i] << " points, expected " << expected_count << std::endl;
        }
    }
}

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::class_<CuvsKmeans>(m, "CuvsKmeans")
        .def(py::init<int64_t, int64_t>())
        .def("fit_predict", &CuvsKmeans::fit_predict)
        .def("balanced_fit_predict", &CuvsKmeans::balanced_fit_predict);

    m.def("balanced_kmeans", &balanced_kmeans, "balanced_kmeans");
}