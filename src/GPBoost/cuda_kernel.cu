/*!
* This file is part of GPBoost a C++ library for combining
*	boosting with Gaussian process and mixed effects models
*
* Copyright (c) 2020 Fabio Sigrist. All rights reserved.
*
* Licensed under the Apache License Version 2.0. See LICENSE file in the project root for license information.
*/
#ifdef USE_CUDA_GP
#include <chrono>  // only for debugging
#include <thread> // only for debugging
#include <cstdio>
#include <math.h>
#include <GPBoost/GP_utils.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cublas_v2.h>
#include <cusparse.h>
#include <device_launch_parameters.h>
#include <cusolverDn.h>
#include <LightGBM/utils/log.h>
using LightGBM::Log;

// Define infinity
#ifndef CUDART_INF
#define CUDART_INF __longlong_as_double(0x7ff0000000000000ULL)
#endif

// Maximum neighbor size per data point
#define MAX_K 64

// Maximum number of GP parameters
#define MAX_NUM_PAR_GP 16


namespace GPBoost {

#define CUDA_CHECK(call)                                                     \
{                                                                            \
    cudaError_t err = call;                                                  \
    if (err != cudaSuccess) {                                                \
        fprintf(stderr, "CUDA error at %s:%d: %s\n",                         \
                __FILE__, __LINE__, cudaGetErrorString(err));fflush(stdout); \
        return false;                                                        \
    }                                                                        \
}

    __device__ double Matern_GPU(const double* __restrict__ pars,
        double d,
        const double shape,
        bool ard,
        double EPSILON_NUMBERS) {
        // Safety for zero distance
        double var = pars[0];
        double range;
        if (ard) {
            range = 1.;
        }
        else {
            range = pars[1];
        }
        if (d < EPSILON_NUMBERS) return var;
        double range_dist = range * d;
        if (shape == 0.5) {
            return var * exp(-range_dist);
        }
        else if (shape == 1.5) {
            return var * (1. + range_dist) * exp(-range_dist);
        }
        else if (shape == 2.5) {
            return var * (1. + range_dist + range_dist * range_dist / 3.) * exp(-range_dist);
        }
        else {
            return 0.0;
        }
    }

    __device__ double Matern_GPU_case(double var, double range_dist, int shape) {
        switch (shape) {
        case 5:  return var * exp(-range_dist);
        case 15: return var * (1. + range_dist) * exp(-range_dist);
        case 25: return var * (1. + range_dist + range_dist * range_dist / 3.) * exp(-range_dist);
        default: return 0.0;
        }
    }


    // Device function: compute distance
    __device__ double distances_funct_device(
        int coord_ind_i,            // index i
        int coords_ind_j,    // indices j
        int num_ip,                      // number of inducing points
        const double* coords,       // [num_data * dim_coords], row-major
        int dim_coords,             // coordinate dimension
        const double* corr_diag,    // [num_data]
        const double* chol_ip_cross_cov, // [num_ip * num_data] 
        int dist_funct,             // which distance is used
        const double var,
        const int shape,
        const double range,
        double EPSILON_NUMBERS
    ) {
        // Grab reference column for coord_ind_i
        // (assuming chol_ip_cross_cov is column-major: dim_coords x num_data)
        if (dist_funct == 1) {
            // Step 1: dot product
            double dot = 0.0;
            for (int d = 0; d < num_ip; d++) {
                double a = chol_ip_cross_cov[coords_ind_j * num_ip + d];// col j
                double b = chol_ip_cross_cov[coord_ind_i * num_ip + d]; // col i
                dot = fma(a, b, dot);
            }
            // Step 2: Euclidean distance if needed
            double sum = 0.0;
            for (int d = 0; d < dim_coords; d++) {
                double diff = coords[coords_ind_j * dim_coords + d] -
                    coords[coord_ind_i * dim_coords + d];
                sum = fma(diff, diff, sum);
            }
            double range_dist = range * sqrt(sum);
            double cov = Matern_GPU_case(var, range_dist, shape);
            //double cov = Matern_GPU(pars, dist_ij, shape, ard, EPSILON_NUMBERS);
            // Step 3: compute final residual distance
            double diag_i = corr_diag[coord_ind_i];
            double diag_j = corr_diag[coords_ind_j];
            //double val = (cov - dot) * rsqrt(diag_i * diag_j);
            //double tmp = 1.0 - fabs(val);
            double num = cov - dot;
            double tmp = diag_i * diag_j / (num * num);
            //return (tmp > 0.0) ? sqrt(tmp) : 0.0;
            return tmp;
        }
        return CUDART_INF;
    }

   
    // Brute-force kNN kernel -----------------
    __global__ void knn_bruteforce_kernel(
        int n, int d, int k,
        const double* coords,              // [n * d], row-major
        const double* corr_diag,           // [n]
        const double* chol_ip_cross_cov,   // [num_ip * n]
        int num_ip,
        const double var,
        const int shape,
        const double range,
        double EPSILON_NUMBERS,
        int dist_funct,
        int* knn_idx,   // [n * k], output
        int start_at
    ) {
        
        if (k > MAX_K) return;

        int i = blockIdx.x + start_at;   // one block per query point
        if (i >= n) return;
       
        int tid = threadIdx.x;

        extern __shared__ double shmem[];
        double* dist_buf = shmem;          // [blockDim.x]
        int* idx_buf = (int*)&dist_buf[blockDim.x * k];
        // local top-k buffers
        double local_dist[MAX_K];
        int local_idx[MAX_K];
        for (int kk = 0; kk < k; kk++) {
            local_dist[kk] = CUDART_INF;
            local_idx[kk] = -1;
        }
        // each thread checks candidates j < i
        for (int j = tid; j < i; j += blockDim.x) {
            // call your distance function with single j
            double dij = distances_funct_device(i,j,num_ip,coords,d,corr_diag,chol_ip_cross_cov,dist_funct,
                var, shape, range,EPSILON_NUMBERS);

            // insert into local top-k
            int worst = 0;
            for (int kk = 1; kk < k; kk++) {
                if (local_dist[kk] > local_dist[worst]) worst = kk;
            }
            if (dij < local_dist[worst]) {
                local_dist[worst] = dij;
                local_idx[worst] = j;
            }
        }
        // write local results to shared memory
        for (int kk = 0; kk < k; kk++) {
            dist_buf[tid * k + kk] = local_dist[kk];
            idx_buf[tid * k + kk] = local_idx[kk];
        }
        __syncthreads();

        // block reduction: keep only best k
        if (tid == 0) {
            double final_dist[MAX_K];
            int final_idx[MAX_K];
            for (int kk = 0; kk < k; kk++) {
                final_dist[kk] = CUDART_INF;
                final_idx[kk] = -1;
            }

            int total = blockDim.x * k;
            for (int t = 0; t < total; t++) {
                double dval = dist_buf[t];
                int jval = idx_buf[t];
                if (jval < 0) continue;

                int worst = 0;
                for (int kk = 1; kk < k; kk++) {
                    if (final_dist[kk] > final_dist[worst]) worst = kk;
                }
                if (dval < final_dist[worst]) {
                    final_dist[worst] = dval;
                    final_idx[worst] = jval;
                }
            }

            // insertion sort: sort results ascending (closest first)
            for (int a = 1; a < k; a++) {
                double key_dist = final_dist[a];
                int key_idx = final_idx[a];
                int b = a - 1;
                while (b >= 0 && final_dist[b] > key_dist) {
                    final_dist[b + 1] = final_dist[b];
                    final_idx[b + 1] = final_idx[b];
                    b--;
                }
                final_dist[b + 1] = key_dist;
                final_idx[b + 1] = key_idx;
            }

            // write out
            for (int kk = 0; kk < k; kk++) {
                knn_idx[(i - start_at) * k + kk] = final_idx[kk];
            }
        }
    }

    bool find_nearest_neighbors_bruteforce_GPU(
        const den_mat_t& coords,
        int num_data,
        int num_neighbors,      
        int start_at,
        int dim_coords,
        const vec_t& corr_diag,
        const den_mat_t& chol_ip_cross_cov,
        const double var,
        const int shape,
        const double range,
        double EPSILON_NUMBERS,
        int dist_funct,
        std::vector<std::vector<int>>& neighbors
    ) {

        // --- prepare sizes ---
        int total_threads = num_data - start_at;

        // --- allocate device memory ---
        double* d_coords = nullptr;
        double* d_corr_diag = nullptr;
        double* d_chol_ip_cross_cov = nullptr;
        int* d_neighbors = nullptr;

        den_mat_t chol_ip_cross_cov_T = chol_ip_cross_cov.transpose();
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> coords_row = coords;
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> chol_ip_cross_cov_row = chol_ip_cross_cov_T;

        CUDA_CHECK(cudaMalloc(&d_coords, coords_row.size() * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_corr_diag, corr_diag.size() * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_chol_ip_cross_cov, chol_ip_cross_cov_row.size() * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_neighbors, total_threads * num_neighbors * sizeof(int)));

        // --- copy data to device ---
        CUDA_CHECK(cudaMemcpy(d_coords, coords_row.data(), coords_row.size() * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_corr_diag, corr_diag.data(), corr_diag.size() * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_chol_ip_cross_cov, chol_ip_cross_cov.data(), chol_ip_cross_cov.size() * sizeof(double), cudaMemcpyHostToDevice));
        // --- launch kernel ---
        int threads = 128;
        int blocks = total_threads;   // one block per query point
        size_t shmem_size = threads * num_neighbors * (sizeof(double) + sizeof(int));
        knn_bruteforce_kernel << <blocks, threads, shmem_size >> > (
            num_data, dim_coords, num_neighbors,
            d_coords,
            d_corr_diag,
            d_chol_ip_cross_cov,
            (int)chol_ip_cross_cov.rows(), // num_ip
            var,
            shape,
            range,
            EPSILON_NUMBERS,
            dist_funct,
            d_neighbors,
            start_at
            );
        printf("kNN1\n"); fflush(stdout);
        cudaError_t launchErr = cudaGetLastError();
        if (launchErr != cudaSuccess) {
            fprintf(stderr, "kNN kernel launch failed: %s\n", cudaGetErrorString(launchErr)); fflush(stdout);
            return false;
        }
        printf("kNN2\n"); fflush(stdout);
        cudaError_t execErr = cudaDeviceSynchronize();
        if (execErr != cudaSuccess) {
            fprintf(stderr, "kNN kernel execution failed: %s\n", cudaGetErrorString(execErr)); fflush(stdout);
            return false;
        }
        printf("kNN21\n"); fflush(stdout);
        // --- copy back results ---
        std::vector<int> h_neighbors(total_threads * num_neighbors);

        CUDA_CHECK(cudaMemcpy(h_neighbors.data(), d_neighbors, h_neighbors.size() * sizeof(int), cudaMemcpyDeviceToHost));
        printf("kNN3\n"); fflush(stdout);
        // --- fill results ---
#pragma omp parallel for schedule(static)
        for (int i = start_at; i < num_data; i++) {
            for (int j = 0; j < num_neighbors; j++) {
                neighbors[i][j] = h_neighbors[(i - start_at) * num_neighbors + j];
            }
        }
        printf("kNN4\n"); fflush(stdout);
        // --- cleanup ---
        cudaFree(d_coords);
        cudaFree(d_corr_diag);
        cudaFree(d_chol_ip_cross_cov);
        cudaFree(d_neighbors);
        printf("kNN5\n"); fflush(stdout);
        return true;
    }

    __device__ void SortVectorsDecreasing_GPU(double* a, int* b, int n) {
        int j, k, l;
        double v;
        for (j = 1; j <= n - 1; j++) {
            k = j;
            while (k > 0 && a[k] < a[k - 1]) {  // decreasing order!
                v = a[k];
                l = b[k];
                a[k] = a[k - 1];
                b[k] = b[k - 1];
                a[k - 1] = v;
                b[k - 1] = l;
                k--;
            }
        }
    }

    __device__ void find_nearest_neighbors_fast_internal_GPU(
        const int i,
        const int num_data,
        const int num_neighbors,
        const int end_search_at,
        const int dim_coords,
        const double* coords,          // [num_data * dim_coords], row-major
        const int* sort_sum,           // [num_data]
        const int* sort_inv_sum,       // [num_data]
        const double* coords_sum,      // [num_data]
        int* neighbors_i,              // [num_neighbors], output
        double* nn_square_dist         // [num_neighbors], output
    ) {

        bool down = true;
        bool up = true;
        int up_i = sort_inv_sum[i];
        int down_i = sort_inv_sum[i];

        double smd, sed;
        while (up || down) {
            if (down_i == 0) { down = false; }
            if (up_i == (num_data - 1)) { up = false; }

            if (down) {
                down_i--;
                int cand = sort_sum[down_i];
                if (cand < i && cand <= end_search_at) {
                    smd = (coords_sum[cand] - coords_sum[i]) * (coords_sum[cand] - coords_sum[i]);
                    if (smd > dim_coords * nn_square_dist[num_neighbors - 1]) {
                        down = false;
                    }
                    else {
                        // squared Euclidean distance
                        sed = 0.0;
                        for (int d = 0; d < dim_coords; d++) {
                            double diff = coords[cand * dim_coords + d] - coords[i * dim_coords + d];
                            sed += diff * diff;
                        }
                        if (sed < nn_square_dist[num_neighbors - 1]) {
                            nn_square_dist[num_neighbors - 1] = sed;
                            neighbors_i[num_neighbors - 1] = cand;
                            SortVectorsDecreasing_GPU(nn_square_dist, neighbors_i, num_neighbors);
                        }
                    }
                }
            }
            if (up) {
                up_i++;
                int cand = sort_sum[up_i];
                if (cand < i && cand <= end_search_at) {
                    smd = (coords_sum[cand] - coords_sum[i]) * (coords_sum[cand] - coords_sum[i]);
                    if (smd > dim_coords * nn_square_dist[num_neighbors - 1]) {
                        up = false;
                    }
                    else {
                        // squared Euclidean distance
                        sed = 0.0;
                        for (int d = 0; d < dim_coords; d++) {
                            double diff = coords[cand * dim_coords + d] - coords[i * dim_coords + d];
                            sed += diff * diff;
                        }
                        if (sed < nn_square_dist[num_neighbors - 1]) {
                            nn_square_dist[num_neighbors - 1] = sed;
                            neighbors_i[num_neighbors - 1] = cand;
                            SortVectorsDecreasing_GPU(nn_square_dist, neighbors_i, num_neighbors);
                        }
                    }
                }
            }
        }
    }

    // Kernel
    __global__ void find_neighbors_kernel(
        int first_i,
        int num_data,
        int num_neighbors,
        int num_close_neighbors,
        int start_at,
        int end_search_at,
        int dim_coords,
        const double* coords,         // [num_data * dim_coords]
        const int* sort_sum,          // [num_data]
        const int* sort_inv_sum,      // [num_data]
        const double* coords_sum,     // [num_data]
        int* neighbors,               // [(num_data - first_i) * num_neighbors]
        double* dist_obs_neighbors,   // same shape (optional)
        bool save_distances,
        bool check_has_duplicates,
        int* has_duplicates_flag     // global flag (0 or 1)
    ) {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        int i = first_i + tid;
        if (i >= num_data) return;
        // output pointers for this thread
        int* neighbors_i = &neighbors[(i - first_i) * num_neighbors];
        double* dist_i = nullptr;
        if (save_distances) {
            dist_i = &dist_obs_neighbors[(i - first_i) * num_neighbors];
        }
        double nn_square_dist[MAX_K];

        // sanity checks
        if (num_neighbors > MAX_K) {
            // out-of-bounds risk, just bail
            return;
        }
        // initialize nearest
        for (int j = 0; j < num_neighbors; j++) {
            nn_square_dist[j] = CUDART_INF;
            neighbors_i[j] = -1;
        }
        find_nearest_neighbors_fast_internal_GPU(
            i, num_data, num_neighbors, end_search_at,
            dim_coords, coords, sort_sum, sort_inv_sum, coords_sum,
            neighbors_i, nn_square_dist
        );
        // --- distances & duplicates ---
        if (save_distances || (check_has_duplicates && (*has_duplicates_flag == 0))) {
            for (int j = 0; j < num_neighbors; j++) {
                double dij = sqrt(nn_square_dist[j]);
                if (save_distances) dist_i[j] = dij;
                if (check_has_duplicates && (*has_duplicates_flag == 0) && dij < 1e-12) {
                    atomicExch(has_duplicates_flag, 1);
                }
            }
        }
    }

    bool find_nearest_neighbors_Vecchia_fast_GPU(
        const den_mat_t& coords,
        int num_data,
        int num_neighbors,
        int num_close_neighbors,
        int start_at,
        int end_search_at,
        int dim_coords,
        const std::vector<int>& sort_sum,
        const std::vector<int>& sort_inv_sum,
        const std::vector<double>& coords_sum,
        std::vector<std::vector<int>>& neighbors,
        std::vector<den_mat_t>& dist_obs_neighbors,
        bool save_distances,
        bool& has_duplicates,
        bool check_has_duplicates
    ) {
        int first_i = (start_at <= num_neighbors) ? (num_neighbors + 1) : start_at;
        int total_threads = num_data - first_i;
        // --- allocate device memory ---
        double* d_coords = nullptr;
        int* d_sort_sum = nullptr;
        int* d_sort_inv_sum = nullptr;
        double* d_coords_sum = nullptr;
        int* d_neighbors = nullptr;
        double* d_dist_obs_neighbors = nullptr;
        int* d_has_duplicates = nullptr;
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> coords_row = coords;
        CUDA_CHECK(cudaMalloc(&d_coords, coords_row.size() * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_sort_sum, num_data * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_sort_inv_sum, num_data * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_coords_sum, num_data * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_neighbors, total_threads * num_neighbors * sizeof(int)));
        if (save_distances) {
            CUDA_CHECK(cudaMalloc(&d_dist_obs_neighbors, total_threads * num_neighbors * sizeof(double)));
        }
        CUDA_CHECK(cudaMalloc(&d_has_duplicates, sizeof(int)));
        // --- copy host data to device ---
        CUDA_CHECK(cudaMemcpy(d_coords, coords_row.data(), num_data * dim_coords * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_sort_sum, sort_sum.data(), num_data * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_sort_inv_sum, sort_inv_sum.data(), num_data * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_coords_sum, coords_sum.data(), num_data * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_has_duplicates, 0, sizeof(int)));
        int threads = 256;
        int blocks = (total_threads + threads - 1) / threads;
        printf("Launching kernel with %d blocks, %d threads (n=%d)\n",
            threads, blocks, total_threads);
        fflush(stdout);
        // --- run neighbor kernel ---
        find_neighbors_kernel << <blocks, threads >> > (
            first_i,
            num_data,
            num_neighbors,
            num_close_neighbors,
            start_at,
            end_search_at,
            dim_coords,
            d_coords,
            d_sort_sum,
            d_sort_inv_sum,
            d_coords_sum,
            d_neighbors,
            d_dist_obs_neighbors,
            save_distances,
            check_has_duplicates,
            d_has_duplicates
            );
        cudaError_t launchErr = cudaGetLastError();
        if (launchErr != cudaSuccess) {
            fprintf(stderr, "Neighbor kernel launch failed: %s\n", cudaGetErrorString(launchErr)); fflush(stdout);
            return false;
        }
        cudaError_t execErr = cudaDeviceSynchronize();
        if (execErr != cudaSuccess) {
            fprintf(stderr, "Neighbor kernel execution failed: %s\n", cudaGetErrorString(execErr)); fflush(stdout);
            return false;
        }

        // --- copy back results ---
        std::vector<int> h_neighbors(total_threads * num_neighbors);
        CUDA_CHECK(cudaMemcpy(h_neighbors.data(), d_neighbors, h_neighbors.size() * sizeof(int), cudaMemcpyDeviceToHost));

        std::vector<double> h_dist;
        if (save_distances) {
            h_dist.resize(total_threads * num_neighbors);
            CUDA_CHECK(cudaMemcpy(h_dist.data(), d_dist_obs_neighbors, h_dist.size() * sizeof(double), cudaMemcpyDeviceToHost));
        }
        int h_has_duplicates = 0;
        if (check_has_duplicates) {
            CUDA_CHECK(cudaMemcpy(&h_has_duplicates, d_has_duplicates, sizeof(int), cudaMemcpyDeviceToHost));
            has_duplicates = (h_has_duplicates == 1);
        }

        // --- fill into neighbors/dist_obs_neighbors ---
        for (int i = first_i; i < num_data; i++) {
            for (int j = 0; j < num_neighbors; j++) {
                neighbors[i - start_at][j] = h_neighbors[(i - first_i) * num_neighbors + j];
            }
            if (save_distances) {
                dist_obs_neighbors[i - start_at].resize(num_neighbors, 1);
                for (int j = 0; j < num_neighbors; j++) {
                    dist_obs_neighbors[i - start_at](j, 0) =
                        h_dist[(i - first_i) * num_neighbors + j];
                }
            }
        }
        // --- cleanup ---
        cudaFree(d_coords);
        cudaFree(d_sort_sum);
        cudaFree(d_sort_inv_sum);
        cudaFree(d_coords_sum);
        cudaFree(d_neighbors);
        if (save_distances) cudaFree(d_dist_obs_neighbors);
        cudaFree(d_has_duplicates);
        return true;
    }

    // compute squared Euclidean distance between two points in d dims
    __device__ double squared_distance(const double* __restrict__ a, const double* __restrict__ b, int d) {
        double s = 0.0;
        for (int k = 0; k < d; ++k) {
            double t = a[k] - b[k];
            s += t * t; 
        }
        return s;
    }

    __device__ double GradientRangeMatern_GPU(const double* __restrict__ a, 
        const double* __restrict__ b, 
        const double* __restrict__ pars,
        double d, 
        const double C, 
        const double shape, 
        const int par, 
        bool ard,
        double EPSILON_NUMBERS) {
        double range;
        if (ard) {
            range = 1.;
        }
        else {
            range = pars[1];
        }
        // Safety for zero distance
        if (d < EPSILON_NUMBERS) return 0.0;
        double range_dist = range * d;
        if (ard) {
            double dist_par = a[par - 1] - b[par - 1];
            dist_par = dist_par * dist_par;
            if (dist_par < EPSILON_NUMBERS) return 0.0;
            if (shape == 0.5) {
                return C * dist_par / d * exp(-range_dist);
            }
            else if (shape == 1.5) {
                return C * dist_par * exp(-d);
            }
            else if (shape == 2.5) {
                return C * dist_par * (1 + d) * exp(-d);
            }
            else {
                return 0.0;
            }
        }
        else {
            if (shape == 0.5) {
                return C * d * exp(-range_dist);
            }
            else if (shape == 1.5) {
                return C * d * d * exp(-range_dist);
            }
            else if (shape == 2.5) {
                return C / 3. * d * d * (1. + range_dist) * exp(-range_dist);
            }
            else {
                return 0.0;
            }
        }
    }

    __device__ void forward_solve(const double* __restrict__ L,
            const double* __restrict__ b,
            double* __restrict__ y, int k) {
        for (int i = 0; i < k; ++i) {
            double s = 0.0;
            for (int j = 0; j < i; ++j) s += L[i * k + j] * y[j];
            y[i] = (b[i] - s) / L[i * k + i];
        }
    }

    __device__ void back_solve_lt(const double* __restrict__ L,
            const double* __restrict__ y,
            double* __restrict__ x, int k) {
        for (int i = k - 1; i >= 0; --i) {
            double s = 0.0;
            for (int j = i + 1; j < k; ++j) s += L[j * k + i] * x[j];
            x[i] = (y[i] - s) / L[i * k + i];
        }
    }

    __device__ bool chol_small(double* __restrict__ L, const double* __restrict__ A,
            int k,
            const double EPSILON_NUMBERS) {
        // L lower; A symmetric
        // Copy A -> L
        for (int i = 0; i < k * k; ++i) L[i] = A[i];

        for (int r = 0; r < k; ++r) {
            double sum = 0.0;
            for (int t = 0; t < r; ++t) {
                double Lrt = L[r * k + t];
                sum += Lrt * Lrt;
            }
            double diag = L[r * k + r] - sum;
            if (diag <= EPSILON_NUMBERS) return false;
            double Lrr = sqrt(diag);
            L[r * k + r] = Lrr;

            for (int row = r + 1; row < k; ++row) {
                double ssum = 0.0;
                for (int t = 0; t < r; ++t) ssum += L[row * k + t] * L[r * k + t];
                double val = (L[row * k + r] - ssum) / Lrr;
                L[row * k + r] = val;
            }
            // zero out strictly upper (optional)
            for (int c = r + 1; c < k; ++c) L[r * k + c] = 0.0;
        }
        return true;
    }

    __global__ void CalcCovFactorGradientVecchia_GPU(
        const double shape,                 // smoothness param
        const double C,                     // range param
        const int n,                        // number of data points
        const int dim_coords,               // coordinate dimension
        const double* __restrict__ coords,  // n * dim_coords, row-major (coords[i*dim + d])
        const int* __restrict__ nn_ptr,     // length n+1  (nn_ptr[i+1]-nn_ptr[i] == k_i)
        const int* __restrict__ nn_idx,     // flattened neighbor indices
        const double jitter,                // e.g. 1e-8
        const double nugget_var,            // e.g. 1e-8
        double* __restrict__ B_data,        // flattened B rows: length == nn_ptr[n] (space preallocated)
        double* __restrict__ D_data,    // length n
        double* __restrict__ B_grad_data,   // length = num_params * total_nnz
        double* __restrict__ D_grad_data,   // length = num_params * n
        const double* __restrict__ C_vec,
        const double* __restrict__ pars,
        const int num_par,
        const int num_par_gp,
        bool gauss_likelihood,
        bool transf_scale,
        bool calc_cov_factor,
        bool calc_gradient,
        bool calc_gradient_nugget,
        bool exclude_marg_var_grad,
        bool ard,
        const double EPSILON_NUMBERS
    ) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= n) return;
        int start = nn_ptr[i];
        int end = nn_ptr[i + 1];
        int total_nnz = nn_ptr[n];
        int k = end - start;
        double Const = 1.;
        // --- Stack-allocated temporary arrays per thread ---
        double cov_mat_between_neighbors[MAX_K * MAX_K];
        double cov_grad_mats_between_neighbors[MAX_NUM_PAR_GP * MAX_K * MAX_K];
        double L[MAX_K * MAX_K];

        double cov_mat_obs_neighbors[MAX_K];
        double cov_grad_mats_obs_neighbors[MAX_NUM_PAR_GP * MAX_K];
        double y[MAX_K], z[MAX_K], A_i[MAX_K], A_i_grad_sigma2[MAX_K], A_i_grad[MAX_K];

        // pointers
        const double* xi = coords + ((size_t)i) * dim_coords;
        if (i > 0) {
            // compute cov_mat_obs_neighbors[j] = Sigma_{i, neighbor_j}
            for (int jj = 0; jj < k; ++jj) {
                int nj = nn_idx[start + jj];
                const double* xj = coords + ((size_t)nj) * dim_coords;
                double r = sqrt(squared_distance(xi, xj, dim_coords));
                cov_mat_obs_neighbors[jj] = Matern_GPU(pars, r, shape, ard, EPSILON_NUMBERS);
                if (calc_gradient) {
                    cov_grad_mats_obs_neighbors[0 * k + jj] = cov_mat_obs_neighbors[jj];
                    if (!transf_scale) {
                        cov_grad_mats_obs_neighbors[0 * k + jj] /= pars[0];
                    }
                    for (int ipar = 1; ipar < num_par; ++ipar) {
                        if (ard) {
                            Const = C_vec[ipar - 1];
                        }
                        else {
                            Const = C;
                        }
                        cov_grad_mats_obs_neighbors[ipar * k + jj] = GradientRangeMatern_GPU(xi, xj, pars, r, Const, shape, ipar, ard, EPSILON_NUMBERS);
                    }
                }
            }
            // compute Sigma_nn (symmetric)
            for (int p = 0; p < k; ++p) {
                for (int q = 0; q <= p; ++q) {
                    int idx_p = nn_idx[start + p];
                    int idx_q = nn_idx[start + q];
                    const double* xp = coords + ((size_t)idx_p) * dim_coords;
                    const double* xq = coords + ((size_t)idx_q) * dim_coords;
                    double r = sqrt(squared_distance(xp, xq, dim_coords));
                    double val = Matern_GPU(pars, r, shape, ard, EPSILON_NUMBERS);
                    cov_mat_between_neighbors[p * k + q] = val;
                    cov_mat_between_neighbors[q * k + p] = val;
                    if (calc_gradient) {
                        cov_grad_mats_between_neighbors[0 * k * k + p * k + q] = val;
                        cov_grad_mats_between_neighbors[0 * k * k + q * k + p] = val;
                        if (!transf_scale) {
                            cov_grad_mats_between_neighbors[0 * k * k + p * k + q] /= pars[0];
                            cov_grad_mats_between_neighbors[0 * k * k + q * k + p] /= pars[0];
                        }
                        for (int ipar = 1; ipar < num_par; ++ipar) {
                            if (ard) {
                                Const = C_vec[ipar - 1];
                            }
                            else {
                                Const = C;
                            }
                            cov_grad_mats_between_neighbors[ipar * k * k + p * k + q] = GradientRangeMatern_GPU(xp, xq, pars, r, Const, shape, ipar, ard, EPSILON_NUMBERS);
                            if (p != q) {
                                cov_grad_mats_between_neighbors[ipar * k * k + q * k + p] = cov_grad_mats_between_neighbors[ipar * k * k + p * k + q];
                            }
                        }
                    }
                }
            }
            if (gauss_likelihood) {
                if (transf_scale) {
                    for (int dd = 0; dd < k; ++dd) cov_mat_between_neighbors[dd * k + dd] += 1;
                }
                else {
                    for (int dd = 0; dd < k; ++dd) cov_mat_between_neighbors[dd * k + dd] += nugget_var;
                }
            }
            else {
                for (int dd = 0; dd < k; ++dd) cov_mat_between_neighbors[dd * k + dd] *= jitter;
            }
        }
        double Sigma_ii = pars[0];
        double Sigma_grad_ii;
        if (!transf_scale && gauss_likelihood) {
            Sigma_ii *= nugget_var;
        }
        if (calc_gradient) {
            if (transf_scale) {
                Sigma_grad_ii = Sigma_ii;
            }
            else {
                Sigma_grad_ii = 1.;
            }
            if (!exclude_marg_var_grad) {
                D_grad_data[i] = Sigma_grad_ii;
            }
            if (calc_gradient_nugget) {
                D_grad_data[num_par_gp - 1 + i] = 1.;
            }
        }
        if (calc_cov_factor) {
            D_data[i] = Sigma_ii;
        }
        if (i > 0) {
            // --- Cholesky: compute L such that Sigma = L * L^T
            // L stored in row-major: L[row*k + col], valid for col <= row
            if (!chol_small(L, cov_mat_between_neighbors, k, EPSILON_NUMBERS)) {
                return;
            }

            // --- Solve L * y = s^T  (forward substitution)
            forward_solve(L, cov_mat_obs_neighbors, y, k);

            // --- Solve L^T * A_i = y  (back substitution)
            back_solve_lt(L, y, A_i, k);
            if (calc_gradient) {
                if (calc_gradient_nugget) {
                    // --- Solve L * y = s^T  (forward substitution)
                    forward_solve(L, A_i, y, k);

                    // --- Solve L^T * A_i_grad_sigma2 = y  (back substitution)
                    back_solve_lt(L, y, A_i_grad_sigma2, k);
                }
                for (int ipar = 0; ipar < num_par; ++ipar) {
                    if (!exclude_marg_var_grad) {
                        // --- Solve L * y = s^T  (forward substitution)
                        const double* rhs = cov_grad_mats_obs_neighbors + ipar * k;
                        forward_solve(L, rhs, y, k);
                        // --- Solve L^T * A_i_grad = y  (back substitution)
                        back_solve_lt(L, y, A_i_grad, k);
                        for (int j = 0; j < k; ++j) {
                            double mult = 0.0;
                            for (int jj = 0; jj < k; ++jj) {
                                mult += A_i[jj] * cov_grad_mats_between_neighbors[ipar * k * k + j * k + jj];
                            }
                            z[j] = mult;
                        }
                        // --- Solve L * y = z^T  (forward substitution)
                        forward_solve(L, z, y, k);
                        // --- Solve L^T * z = y  (back substitution)
                        back_solve_lt(L, y, z, k);
                        for (int j = 0; j < k; ++j) {
                            A_i_grad[j] -= z[j];
                        }
                        for (int j = 0; j < k; ++j) {
                            B_grad_data[ipar * total_nnz + start + j] = -A_i_grad[j];
                        }
                        double dot_grad_1 = 0.0;
                        double dot_grad_2 = 0.0;
                        for (int j = 0; j < k; ++j) {
                            dot_grad_1 += cov_grad_mats_obs_neighbors[ipar * k + j] * A_i[j];
                            dot_grad_2 += cov_mat_obs_neighbors[j] * A_i_grad[j];
                        }
                        if (ipar == 0) {
                            D_grad_data[ipar * n + i] -= dot_grad_1 + dot_grad_2;
                        }
                        else {
                            D_grad_data[ipar * n + i] = -dot_grad_1 - dot_grad_2;
                        }
                    }
                }
                if (calc_gradient_nugget) {
                    for (int j = 0; j < k; ++j) {
                        B_grad_data[num_par_gp - 1 + start + j] = -A_i_grad_sigma2[j];
                    }
                    double dot_grad = 0.0;
                    for (int j = 0; j < k; ++j) dot_grad += cov_mat_obs_neighbors[j] * A_i_grad_sigma2[j];
                    D_grad_data[num_par_gp - 1 + i] -= dot_grad;
                }
            }
            // Now A_i = Sigma_nn^{-1} * s^T (k x 1)
            // B_i (1 x k) = (s * Sigma_nn^{-1}) = (A_i)^T  (because Sigma is symmetric)
            // store B at B_data[start + j] = -A_i[j]
            if (calc_cov_factor) {
                for (int j = 0; j < k; ++j) {
                    B_data[start + j] = -A_i[j];
                }
                double dot = 0.0;
                for (int j = 0; j < k; ++j) dot += cov_mat_obs_neighbors[j] * A_i[j];
                D_data[i] -= dot;
            }
        }
    }

    bool LaunchCalcCovFactorGradientVecchia_GPU(
        const double shape,                 // smoothness param
        const double C,                     // range param
        const int n,                        // number of data points
        const int dim_coords,               // coordinate dimension
        const double* __restrict__ coords,  // n * dim_coords, row-major (coords[i*dim + d])
        const int* __restrict__ nn_ptr,     // length n+1  (nn_ptr[i+1]-nn_ptr[i] == k_i)
        const int* __restrict__ nn_idx,     // flattened neighbor indices
        const double jitter,                // e.g. 1e-8
        const double nugget_var,            // e.g. 1e-8
        double* __restrict__ B_data,        // flattened B rows: length == nn_ptr[n] (space preallocated)
        double* __restrict__ D_data,    // length n
        double* __restrict__ B_grad_data,   // length = num_params * total_nnz
        double* __restrict__ D_grad_data,   // length = num_params * n
        const double* __restrict__ C_vec,
        const double* __restrict__ pars,
        const int num_par,
        const int num_par_gp,
        bool gauss_likelihood,
        bool transf_scale,
        bool calc_cov_factor,
        bool calc_gradient,
        bool calc_gradient_nugget,
        bool exclude_marg_var_grad,
        bool ard,
        const double EPSILON_NUMBERS) {


        int threadsPerBlock = 256;
        int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

        printf("Launching kernel with %d blocks, %d threads (n=%d)\n",
            blocksPerGrid, threadsPerBlock, n);
        fflush(stdout);

        CalcCovFactorGradientVecchia_GPU << <blocksPerGrid, threadsPerBlock >> > (
            shape, C, n, dim_coords,
            coords, nn_ptr, nn_idx,
            jitter, nugget_var,
            B_data, D_data, B_grad_data, D_grad_data, C_vec,
            pars, num_par, num_par_gp,
            gauss_likelihood, transf_scale,
            calc_cov_factor, calc_gradient,
            calc_gradient_nugget, exclude_marg_var_grad, ard, EPSILON_NUMBERS
            );

        cudaError_t launchErr = cudaGetLastError();
        if (launchErr != cudaSuccess) {
            fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(launchErr)); fflush(stdout);
            return false;
        }

        cudaError_t execErr = cudaDeviceSynchronize();
        if (execErr != cudaSuccess) {
            fprintf(stderr, "Kernel execution error: %s\n", cudaGetErrorString(execErr)); fflush(stdout);
            return false;
        }

        printf("Kernel completed successfully.\n"); fflush(stdout);
        // Record stop
         return true;
    }

    bool try_matmul_gpu(const den_mat_t& A, const den_mat_t& B, den_mat_t& C) {
        int M = A.rows(), K = A.cols(), N = B.cols();
        if (K != B.rows()) {
            Log::REInfo("[GPU] Dimension mismatch.");
            return false;
        }

        C.resize(M, N);

        const double* h_A = A.data();
        const double* h_B = B.data();
        double* h_C = C.data();

        double* d_A = nullptr, * d_B = nullptr, * d_C = nullptr;
        cudaError_t cuda_stat;
        cublasStatus_t stat;
        cublasHandle_t handle;

        size_t size_A = M * K * sizeof(double);
        size_t size_B = K * N * sizeof(double);
        size_t size_C = M * N * sizeof(double);

        cuda_stat = cudaMalloc((void**)&d_A, size_A);
        if (cuda_stat != cudaSuccess) return false;
        cuda_stat = cudaMalloc((void**)&d_B, size_B);
        if (cuda_stat != cudaSuccess) {
            cudaFree(d_A);
            return false;
        }

        cuda_stat = cudaMalloc((void**)&d_C, size_C);
        if (cuda_stat != cudaSuccess) {
            cudaFree(d_A); cudaFree(d_B);
            return false;
        }

        cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);

        stat = cublasCreate(&handle);
        if (stat != CUBLAS_STATUS_SUCCESS) {
            cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
            return false;
        }

        const double alpha = 1.0;
        const double beta = 0.0;

        // cuBLAS performs: C = alpha * op(A) * op(B) + beta * C
        // We want: C = A * B
        // A: MxK, B: KxN, C: MxN
        // So op(A) = A, op(B) = B
        stat = cublasDgemm(handle,
            CUBLAS_OP_N, CUBLAS_OP_N,  // No transpose
            M, N, K,                   // C is MxN, A is MxK, B is KxN
            &alpha,
            d_A, M,  // lda = leading dim of A = M (since column-major)
            d_B, K,  // ldb = leading dim of B = K
            &beta,
            d_C, M); // ldc = leading dim of C = M

        if (stat != CUBLAS_STATUS_SUCCESS) {
            cublasDestroy(handle);
            cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
            return false;
        }

        cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);

        cublasDestroy(handle);
        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);

        Log::REInfo("[GPU] Matrix multiplication completed with cuBLAS.");
        return true;
    }

    bool try_diag_times_dense_gpu(const vec_t& D, const den_mat_t& B, den_mat_t& C) {
        int M = B.rows();
        int N = B.cols();

        if (D.size() != M) {
            Log::REInfo("[GPU] Dimension mismatch between diagonal and matrix.");
            return false;
        }

        C.resize(M, N);

        // Host pointers
        const double* h_D = D.data();
        const double* h_B = B.data();
        double* h_C = C.data();

        // Device pointers
        double* d_D = nullptr;
        double* d_B = nullptr;
        double* d_C = nullptr;

        cudaMalloc((void**)&d_D, M * sizeof(double));
        cudaMalloc((void**)&d_B, M * N * sizeof(double));
        cudaMalloc((void**)&d_C, M * N * sizeof(double));

        cudaMemcpy(d_D, h_D, M * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B, M * N * sizeof(double), cudaMemcpyHostToDevice);
        // Create cuBLAS handle
        cublasHandle_t handle;
        cublasCreate(&handle);
        // Multiply: C = diag(D) * B (i.e., scale each row of B by D[i])
        // Use cuBLAS: d_C = diag(d_D) * d_B
        cublasStatus_t stat = cublasDdgmm(handle,
            CUBLAS_SIDE_LEFT, // Left = scale rows (use RIGHT to scale columns)
            M, N,
            d_B, M,
            d_D, 1, // stride = 1
            d_C, M);
        if (stat != CUBLAS_STATUS_SUCCESS) {
            Log::REInfo("[GPU] cuBLAS Ddgmm failed.");
            cudaFree(d_D); cudaFree(d_B); cudaFree(d_C);
            cublasDestroy(handle);
            return false;
        }

        cudaMemcpy(h_C, d_C, M * N * sizeof(double), cudaMemcpyDeviceToHost);

        // Clean up
        cudaFree(d_D);
        cudaFree(d_B);
        cudaFree(d_C);
        cublasDestroy(handle);

        Log::REInfo("[GPU] Diagonal x Dense matrix multiplication completed with cuBLAS.");
        return true;
    }

    bool try_spmatmul_gpu(const sp_mat_rm_t& A, const sp_mat_rm_t& B, sp_mat_rm_t& C) {
        if (A.cols() != B.rows()) return false;

        cusparseHandle_t handle = nullptr;
        cusparseSpMatDescr_t matA = nullptr, matB = nullptr, matC = nullptr;
        cusparseSpGEMMDescr_t spgemmDescr = nullptr;

        int m = A.rows(), k = A.cols(), n = B.cols();
        int A_nnz = A.nonZeros(), B_nnz = B.nonZeros();

        int* d_A_rowPtr = nullptr, * d_A_colInd = nullptr;
        double* d_A_values = nullptr;
        int* d_B_rowPtr = nullptr, * d_B_colInd = nullptr;
        double* d_B_values = nullptr;
        int* d_C_rowPtr = nullptr, * d_C_colInd = nullptr;
        double* d_C_values = nullptr;
        void* dBuffer1 = nullptr, * dBuffer2 = nullptr;

        // Allocate device memory for A
        cudaMalloc(&d_A_rowPtr, (m + 1) * sizeof(int));
        cudaMalloc(&d_A_colInd, A_nnz * sizeof(int));
        cudaMalloc(&d_A_values, A_nnz * sizeof(double));

        // Allocate device memory for B
        cudaMalloc(&d_B_rowPtr, (k + 1) * sizeof(int));
        cudaMalloc(&d_B_colInd, B_nnz * sizeof(int));
        cudaMalloc(&d_B_values, B_nnz * sizeof(double));

        // Copy A and B to device
        cudaMemcpy(d_A_rowPtr, A.outerIndexPtr(), (m + 1) * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_A_colInd, A.innerIndexPtr(), A_nnz * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_A_values, A.valuePtr(), A_nnz * sizeof(double), cudaMemcpyHostToDevice);

        cudaMemcpy(d_B_rowPtr, B.outerIndexPtr(), (k + 1) * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B_colInd, B.innerIndexPtr(), B_nnz * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B_values, B.valuePtr(), B_nnz * sizeof(double), cudaMemcpyHostToDevice);

        // cuSPARSE setup
        cusparseCreate(&handle);
        //cusparseCreateSpGEMMDescr(&spgemmDesc);
        cusparseSpGEMM_createDescr(&spgemmDescr);
        cusparseCreateCsr(&matA, m, k, A_nnz, d_A_rowPtr, d_A_colInd, d_A_values,
            CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);

        cusparseCreateCsr(&matB, k, n, B_nnz, d_B_rowPtr, d_B_colInd, d_B_values,
            CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);

        cusparseCreateCsr(&matC, m, n, 0, nullptr, nullptr, nullptr,
            CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);

        double alpha = 1.0, beta = 0.0;
        size_t bufferSize1 = 0, bufferSize2 = 0;

        // Phase 1: Work estimation
        cusparseSpGEMM_workEstimation(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, matA, matB, &beta, matC, CUDA_R_64F,
            CUSPARSE_SPGEMM_DEFAULT, spgemmDescr, &bufferSize1, nullptr);
        cudaMalloc(&dBuffer1, bufferSize1);
        cusparseSpGEMM_workEstimation(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, matA, matB, &beta, matC, CUDA_R_64F,
            CUSPARSE_SPGEMM_DEFAULT, spgemmDescr, &bufferSize1, dBuffer1);

        // Phase 2: Compute
        cusparseSpGEMM_compute(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, matA, matB, &beta, matC, CUDA_R_64F,
            CUSPARSE_SPGEMM_DEFAULT, spgemmDescr, &bufferSize2, nullptr);
        cudaMalloc(&dBuffer2, bufferSize2);
        cusparseSpGEMM_compute(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, matA, matB, &beta, matC, CUDA_R_64F,
            CUSPARSE_SPGEMM_DEFAULT, spgemmDescr, &bufferSize2, dBuffer2);

        // Phase 3: Copy to finalize matC
        int64_t C_num_rows, C_num_cols, C_nnz;
        cusparseSpMatGetSize(matC, &C_num_rows, &C_num_cols, &C_nnz);
        cudaMalloc(&d_C_rowPtr, (m + 1) * sizeof(int));
        cudaMalloc(&d_C_colInd, C_nnz * sizeof(int));
        cudaMalloc(&d_C_values, C_nnz * sizeof(double));

        cusparseCsrSetPointers(matC, d_C_rowPtr, d_C_colInd, d_C_values);
        cusparseSpGEMM_copy(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, matA, matB, &beta, matC, CUDA_R_64F,
            CUSPARSE_SPGEMM_DEFAULT, spgemmDescr);

        // Copy result to host
        std::vector<int> h_C_rowPtr(m + 1);
        std::vector<int> h_C_colInd(C_nnz);
        std::vector<double> h_C_values(C_nnz);

        cudaMemcpy(h_C_rowPtr.data(), d_C_rowPtr, (m + 1) * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_C_colInd.data(), d_C_colInd, C_nnz * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_C_values.data(), d_C_values, C_nnz * sizeof(double), cudaMemcpyDeviceToHost);

        // Build result Eigen matrix
        C.resize(m, n);
        C.makeCompressed();
        C.reserve(C_nnz);
        std::copy(h_C_rowPtr.begin(), h_C_rowPtr.end(), C.outerIndexPtr());
        std::copy(h_C_colInd.begin(), h_C_colInd.end(), C.innerIndexPtr());
        std::copy(h_C_values.begin(), h_C_values.end(), C.valuePtr());

        // Cleanup
        cudaFree(d_A_rowPtr); cudaFree(d_A_colInd); cudaFree(d_A_values);
        cudaFree(d_B_rowPtr); cudaFree(d_B_colInd); cudaFree(d_B_values);
        cudaFree(d_C_rowPtr); cudaFree(d_C_colInd); cudaFree(d_C_values);
        cudaFree(dBuffer1); cudaFree(dBuffer2);
        cusparseDestroySpMat(matA); cusparseDestroySpMat(matB); cusparseDestroySpMat(matC);
        //cusparseDestroySpGEMMDescr(spgemmDesc);
        cusparseSpGEMM_destroyDescr(spgemmDescr);
        cusparseDestroy(handle);

        return true;
    }

    bool try_sparse_dense_matmul_gpu(const sp_mat_rm_t& A, const den_mat_t& B, den_mat_t& C) {
        int M = A.rows(), K = A.cols(), N = B.cols();
        if (K != B.rows()) {
            Log::REInfo("[GPU] Dimension mismatch.");
            return false;
        }

        const int nnz = A.nonZeros();
        const int* h_csrOffsets = A.outerIndexPtr();  // Row pointers
        const int* h_columns = A.innerIndexPtr();     // Column indices
        const double* h_values = A.valuePtr();        // Non-zero values

        int* d_csrOffsets = nullptr;
        int* d_columns = nullptr;
        double* d_values = nullptr;
        double* d_B = nullptr;
        double* d_C = nullptr;

        cudaMalloc((void**)&d_csrOffsets, (M + 1) * sizeof(int));
        cudaMalloc((void**)&d_columns, nnz * sizeof(int));
        cudaMalloc((void**)&d_values, nnz * sizeof(double));
        cudaMalloc((void**)&d_B, K * N * sizeof(double));
        cudaMalloc((void**)&d_C, M * N * sizeof(double));

        cudaMemcpy(d_csrOffsets, h_csrOffsets, (M + 1) * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_columns, h_columns, nnz * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_values, h_values, nnz * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, B.data(), K * N * sizeof(double), cudaMemcpyHostToDevice);

        cusparseHandle_t handle;
        cusparseCreate(&handle);

        cusparseSpMatDescr_t matA;
        cusparseDnMatDescr_t matB, matC;

        cusparseCreateCsr(&matA, M, K, nnz,
            d_csrOffsets, d_columns, d_values,
            CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
            CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);

        cusparseCreateDnMat(&matB, K, N, K, d_B, CUDA_R_64F, CUSPARSE_ORDER_COL);
        cusparseCreateDnMat(&matC, M, N, M, d_C, CUDA_R_64F, CUSPARSE_ORDER_COL);

        const double alpha = 1.0;
        const double beta = 0.0;

        size_t bufferSize = 0;
        void* dBuffer = nullptr;
        cusparseSpMM_bufferSize(handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, matA, matB, &beta, matC,
            CUDA_R_64F, CUSPARSE_SPMM_CSR_ALG2,
            &bufferSize);

        cudaMalloc(&dBuffer, bufferSize);

        cusparseStatus_t stat = cusparseSpMM(handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, matA, matB, &beta, matC,
            CUDA_R_64F, CUSPARSE_SPMM_CSR_ALG2,
            dBuffer);

        if (stat != CUSPARSE_STATUS_SUCCESS) {
            Log::REInfo("[GPU] cuSPARSE SpMM failed.");
            cudaFree(dBuffer); cudaFree(d_csrOffsets); cudaFree(d_columns);
            cudaFree(d_values); cudaFree(d_B); cudaFree(d_C);
            cusparseDestroySpMat(matA); cusparseDestroyDnMat(matB);
            cusparseDestroyDnMat(matC); cusparseDestroy(handle);
            return false;
        }

        C.resize(M, N);  // Resize Eigen matrix before copying
        cudaMemcpy(C.data(), d_C, M * N * sizeof(double), cudaMemcpyDeviceToHost);

        // Clean up
        cudaFree(dBuffer); cudaFree(d_csrOffsets); cudaFree(d_columns);
        cudaFree(d_values); cudaFree(d_B); cudaFree(d_C);
        cusparseDestroySpMat(matA); cusparseDestroyDnMat(matB);
        cusparseDestroyDnMat(matC); cusparseDestroy(handle);

        return true;
    }

    bool try_solve_lower_triangular_gpu(const chol_den_mat_t& chol, const den_mat_t& R_host, den_mat_t& X_host) {
        den_mat_t L_host = chol.matrixL();
        int n = L_host.rows();
        int m = R_host.cols();
        if (L_host.cols() != n || R_host.rows() != n) {
            return false;
        }
        X_host.resize(n, m);
        // Allocate device memory
        double* d_L = nullptr;
        double* d_X = nullptr;

        cudaMalloc(&d_L, n * n * sizeof(double));
        cudaMalloc(&d_X, n * m * sizeof(double));

        cudaMemcpy(d_L, L_host.data(), n * n * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_X, R_host.data(), n * m * sizeof(double), cudaMemcpyHostToDevice);

        // Create cuBLAS handle
        cublasHandle_t handle;
        cublasStatus_t stat = cublasCreate(&handle);
        if (stat != CUBLAS_STATUS_SUCCESS) {
            cudaFree(d_L); cudaFree(d_X);
            return false;
        }
        const double alpha = 1.0;

        // Solve: L * X = R -> X = L^{-1} * R
        // L is lower-triangular, column-major
        // Left-side, lower-triangular, no transpose, non-unit diagonal
        stat = cublasDtrsm(
            handle,
            CUBLAS_SIDE_LEFT,      // Solve L * X = R
            CUBLAS_FILL_MODE_LOWER,
            CUBLAS_OP_N,           // No transpose
            CUBLAS_DIAG_NON_UNIT,  // Assume general diagonal
            n,                     // number of rows of L and X
            m,                     // number of columns of X
            &alpha,                // Scalar alpha
            d_L, n,                // L, leading dimension n
            d_X, n                 // R becomes X, leading dimension n
        );

        if (stat != CUBLAS_STATUS_SUCCESS) {
            cudaFree(d_L); cudaFree(d_X);
            cublasDestroy(handle);
            return false;
        }

        // Copy result back
        cudaMemcpy(X_host.data(), d_X, n * m * sizeof(double), cudaMemcpyDeviceToHost);

        // Cleanup
        cudaFree(d_L);
        cudaFree(d_X);
        cublasDestroy(handle);

        Log::REInfo("[GPU] Triangular solve with CUBLAS.");
        return true;
    }

    

    // CUDA kernel: Sigma(i,j) -= dot(M1.col(i), M2.col(j))
    __global__ void subtract_prod_from_mat_kernel(
        const double* __restrict__ M1,
        const double* __restrict__ M2,
        double* Sigma,
        int M1_rows, int M1_cols,
        int M2_rows, int M2_cols,
        bool only_triangular)
    {
        int i = blockIdx.y * blockDim.y + threadIdx.y;
        int j = blockIdx.x * blockDim.x + threadIdx.x;

        if (i >= M1_cols || j >= M2_cols) return;
        if (only_triangular && j < i) return;

        double dot = 0.0;
        for (int k = 0; k < M1_rows; ++k) {
            dot += M1[i * M1_rows + k] * M2[j * M2_rows + k];
        }

        // column-major access: Sigma(i, j) => j * rows + i
        atomicAdd(&Sigma[j * M1_cols + i], -dot);

        if (!only_triangular && j > i) {
            atomicAdd(&Sigma[i * M1_cols + j], -dot);  // symmetric fill
        }
    }
    __global__ void subtract_prod_from_sparse_mat_kernel(
    const int* __restrict__ row_ptr,
    const int* __restrict__ col_idx,
    double* __restrict__ values,
    const double* __restrict__ M1,  // Shape: (n_rows, K)
    const double* __restrict__ M2,  // Shape: (n_cols, K)
    int n_rows, int n_cols, int K)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_rows) return;

    int row_start = row_ptr[row];
    int row_end = row_ptr[row + 1];

    for (int idx = row_start; idx < row_end; ++idx) {
        int col = col_idx[idx];

        // Only compute upper triangle or diagonal
        if (row <= col) {
            double dot = 0.0;
            for (int k = 0; k < K; ++k) {
                dot += M1[row * K + k] * M2[col * K + k];
            }
            atomicAdd(&values[idx], -dot);
        }
            // Note: for full symmetry, the host must mirror Sigma(j,i) = Sigma(i,j) afterwards
    }
}

    void launch_subtract_sparse_kernel(
        const int* row_ptr, const int* col_idx, double* values,
        const double* M1, const double* M2,
        int n, int m, int K, bool only_triangular)
    {
        int blockSize = 256;
        int numBlocks = (n + blockSize - 1) / blockSize;
        subtract_prod_from_sparse_mat_kernel << <numBlocks, blockSize >> > (
            row_ptr, col_idx, values, M1, M2, n, m, K);
    }

    void launch_subtract_prod_from_mat_kernel(
        const double* M1, const double* M2, double* Sigma,
        int M1_rows, int M1_cols,
        int M2_rows, int M2_cols,
        bool only_triangular)
    {
        dim3 blockDim(16, 16);
        dim3 gridDim((M2_cols + blockDim.x - 1) / blockDim.x,
            (M1_cols + blockDim.y - 1) / blockDim.y);

        subtract_prod_from_mat_kernel << <gridDim, blockDim >> > (
            M1, M2, Sigma,
            M1_rows, M1_cols,
            M2_rows, M2_cols,
            only_triangular
            );
        cudaDeviceSynchronize();
    }

    
    bool cholesky_cusolver_to_eigen(chol_den_mat_t& llt, const den_mat_t& A_input) {
        int N = A_input.rows();
        if (A_input.cols() != N) {
            Log::REInfo("Input matrix is not square.");
            return false;
        }

        // Step 1: Create cuSolver handle
        cusolverDnHandle_t handle;
        cusolverStatus_t status = cusolverDnCreate(&handle);
        if (status != CUSOLVER_STATUS_SUCCESS) {
            Log::REInfo("cuSOLVER initialization failed.");
            return false;
        }

        // Step 2: Allocate GPU memory for matrix
        double* d_A = nullptr;
        cudaError_t cudaStat = cudaMalloc(&d_A, sizeof(double) * N * N);
        if (cudaStat != cudaSuccess) {
            Log::REInfo("cudaMalloc failed for d_A");
            cusolverDnDestroy(handle);
            return false;
        }

        cudaStat = cudaMemcpy(d_A, A_input.data(), sizeof(double) * N * N, cudaMemcpyHostToDevice);
        if (cudaStat != cudaSuccess) {
            Log::REInfo("cudaMemcpy failed");
            cudaFree(d_A);
            cusolverDnDestroy(handle);
            return false;
        }

        // Step 3: Query buffer size
        int work_size = 0;
        status = cusolverDnDpotrf_bufferSize(handle, CUBLAS_FILL_MODE_LOWER, N, d_A, N, &work_size);
        if (status != CUSOLVER_STATUS_SUCCESS) {
            Log::REInfo("cusolverDnDpotrf_bufferSize failed.");
            cudaFree(d_A);
            cusolverDnDestroy(handle);
            return false;
        }

        double* work = nullptr;
        cudaStat = cudaMalloc(&work, sizeof(double) * work_size);
        if (cudaStat != cudaSuccess) {
            Log::REInfo("cudaMalloc failed for workspace");
            cudaFree(d_A);
            cusolverDnDestroy(handle);
            return false;
        }

        int* dev_info = nullptr;
        cudaStat = cudaMalloc(&dev_info, sizeof(int));
        if (cudaStat != cudaSuccess) {
            Log::REInfo("cudaMalloc failed ");
            cudaFree(d_A);
            cudaFree(work);
            cusolverDnDestroy(handle);
            return false;
        }

        // Step 4: Compute Cholesky factorization
        status = cusolverDnDpotrf(handle, CUBLAS_FILL_MODE_LOWER, N, d_A, N, work, work_size, dev_info);
        if (status != CUSOLVER_STATUS_SUCCESS) {
            Log::REInfo("cusolverDnDpotrf failed.");
            cudaFree(d_A); cudaFree(work); cudaFree(dev_info);
            cusolverDnDestroy(handle);
            return false;
        }

        int dev_info_h = 0;
        cudaStat = cudaMemcpy(&dev_info_h, dev_info, sizeof(int), cudaMemcpyDeviceToHost);
        if (cudaStat != cudaSuccess) {
            Log::REInfo("cudaMemcpy failed");
            cudaFree(d_A); cudaFree(work); cudaFree(dev_info);
            cusolverDnDestroy(handle);
            return false;
        }

        if (dev_info_h != 0) {
            Log::REInfo("Cholesky factorization failed on GPU");
            cudaFree(d_A); cudaFree(work); cudaFree(dev_info);
            cusolverDnDestroy(handle);
            return false;
        }

        // Step 5: Copy result back (only lower triangle)
        den_mat_t L(N, N);
        cudaStat = cudaMemcpy(L.data(), d_A, sizeof(double) * N * N, cudaMemcpyDeviceToHost);
        if (cudaStat != cudaSuccess) {
            Log::REInfo("cudaMemcpy failed");
            cudaFree(d_A); cudaFree(work); cudaFree(dev_info);
            cusolverDnDestroy(handle);
            return false;
        }

        // Step 6: Feed to Eigen's LLT (only lower triangle will be used)
        llt.compute(L.selfadjointView<Eigen::Lower>());

        // Step 7: Cleanup
        cudaFree(d_A);
        cudaFree(work);
        cudaFree(dev_info);
        cusolverDnDestroy(handle);

        Log::REInfo("[GPU] Cholesky factorization with cuSOLVER completed successfully.");
        return true;
    }

}  // namespace GPBoost

#endif  // USE_CUDA_GP
