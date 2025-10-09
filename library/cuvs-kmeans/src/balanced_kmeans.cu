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

void balanced_kmeans() {
    raft::resources handle;
    cudaStream_t stream = raft::resource::get_cuda_stream(handle);

    int64_t seq_len = 128 * 1000, n_features = 128, n_clusters = 8000;
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

int main() {
    balanced_kmeans();
    return 0;
}
