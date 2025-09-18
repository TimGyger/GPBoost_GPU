/*!
* This file is part of GPBoost a C++ library for combining
*	boosting with Gaussian process and mixed effects models
*
* Copyright (c) 2020 Fabio Sigrist. All rights reserved.
*
* Licensed under the Apache License Version 2.0. See LICENSE file in the project root for license information.
*/
#ifdef USE_CUDA_GP
#include <cstdio>
#include <math.h>
#include <GPBoost/GP_utils.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusparse.h>
#include <device_launch_parameters.h>
#include <cusolverDn.h>
#include <LightGBM/utils/log.h>
using LightGBM::Log;

// Maximum neighbor size per data point
#define MAX_K 32

// Maximum number of GP parameters
#define MAX_NUM_PAR_GP 16

namespace GPBoost {

    // compute squared Euclidean distance between two points in d dims
    __device__ double squared_distance(const double* __restrict__ a, const double* __restrict__ b, int d) {
        double s = 0.0;
        for (int k = 0; k < d; ++k) {
            double t = a[k] - b[k];
            s += t * t; 
        }
        return s;
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
            return var *  exp(-range_dist);
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
            if (dist_par < EPSILON_NUMBERS) return 0.0;
            dist_par = dist_par * dist_par;
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
        const double EPSILON_NUMBERS,
        double* __restrict__ d_cov_mat_between_neighbors,
        double* __restrict__ d_cov_grad_mats_between_neighbors,
        double* __restrict__ d_L
    ) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= n) return;
        if (i == 10) {
            printf("Thread1\n");
        }
        int start = nn_ptr[i];
        int end = nn_ptr[i + 1];
        int total_nnz = nn_ptr[n]; // length of flattened neighbor list 
        int k = end - start;

        int tid = threadIdx.x; // local thread inside block

        // Each thread uses its slice in global memory
        double* cov_mat_between_neighbors = d_cov_mat_between_neighbors + tid * MAX_K * MAX_K;
        double* cov_grad_mats_between_neighbors = d_cov_grad_mats_between_neighbors + tid * MAX_NUM_PAR_GP * MAX_K * MAX_K;
        double* L = d_L + tid * MAX_K * MAX_K;

        // small arrays remain on the stack
        double cov_mat_obs_neighbors[MAX_K];
        double cov_grad_mats_obs_neighbors[MAX_NUM_PAR_GP * MAX_K];
        double y[MAX_K], z[MAX_K], A_i[MAX_K], A_i_grad_sigma2[MAX_K], A_i_grad[MAX_K];
        if (i == 10) {
            printf("Thread2\n");
        }
        // pointers
        const double* xi = coords + ((size_t)i) * dim_coords;
        if (i > 0) {
            if (i == 10) {
                printf("Thread3\n");
            }
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
                        cov_grad_mats_obs_neighbors[ipar * k + jj] = GradientRangeMatern_GPU(xi, xj, pars, r, C, shape, ipar, ard, EPSILON_NUMBERS);
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
                    if (i == 10 && p == 0 && q == 0) {
                        printf("Thread4 %g %g %g %g\n", pars[0], shape, r, val);
                    }
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
                            cov_grad_mats_between_neighbors[ipar * k * k + p * k + q] = GradientRangeMatern_GPU(xp, xq, pars, r, C, shape, ipar, ard, EPSILON_NUMBERS);
                            if (p != q) {
                                cov_grad_mats_between_neighbors[ipar * k * k + q * k + p] = cov_grad_mats_between_neighbors[ipar * k * k + p * k + q];
                            }
                        }
                    }
                }
            }
            if (i == 10) {
                printf("Thread5 %g\n", cov_mat_between_neighbors[0]);
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
            if (i == 10) {
                printf("Thread6 %g %g %g\n", cov_mat_between_neighbors[0], nugget_var, jitter);
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
        if (i == 10) {
            printf("Thread7\n");
        }
        if (calc_cov_factor) {
            D_data[i] = Sigma_ii;
            if (i == 10) {
                printf("Thread8\n");
            }
        }

        if (i == 10) {
            printf("Thread9 %g %g %g\n", cov_mat_between_neighbors[0], cov_mat_between_neighbors[1], cov_mat_between_neighbors[2]);
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
            if (i == 10) {
                printf("Thread10\n");
            }
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
            if (i == 10) {
                printf("Thread11\n");
            }
            if (calc_cov_factor) {
                for (int j = 0; j < k; ++j) {
                    B_data[start + j] = -A_i[j];
                }
                if (i == 10) {
                    printf("Thread12\n");
                }
                double dot = 0.0;
                for (int j = 0; j < k; ++j) dot += cov_mat_obs_neighbors[j] * A_i[j];
                D_data[i] -= dot;
                if (i == 10 || i == 100 || i == 1000) {
                    printf("Thread %g and %g\n", D_data[i], dot);
                }
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

#define CUDA_CHECK(call)                                                     \
{                                                                            \
    cudaError_t err = call;                                                  \
    if (err != cudaSuccess) {                                                \
        fprintf(stderr, "CUDA error at %s:%d: %s\n",                         \
                __FILE__, __LINE__, cudaGetErrorString(err));fflush(stdout); \
        exit(EXIT_FAILURE);                                                  \
    }                                                                        \
}
        CUDA_CHECK(cudaFree(0));  // forces context creation
        printf("CUDA context initialized\n"); fflush(stdout);
        printf("Thread0 (before allocations)\n"); fflush(stdout);

        // berechne blocks/threads
        int threadsPerBlock = 128;
        int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

        // Total global threads
        int totalThreads = threadsPerBlock * blocksPerGrid;

        // Allocation sizes: one workspace per global thread
        size_t size_cov = (size_t)totalThreads * MAX_K * MAX_K * sizeof(double);
        size_t size_cov_grad = (size_t)totalThreads * MAX_NUM_PAR_GP * MAX_K * MAX_K * sizeof(double);
        size_t size_L = (size_t)totalThreads * MAX_K * MAX_K * sizeof(double);

        printf("Alloc sizes (bytes): cov=%zu, grad=%zu, L=%zu\n", size_cov, size_cov_grad, size_L); fflush(stdout);

        // Allocate device memory
        double* d_cov_mat_between_neighbors = nullptr;
        double* d_cov_grad_mats_between_neighbors = nullptr;
        double* d_L = nullptr;

        CUDA_CHECK(cudaMalloc(&d_cov_mat_between_neighbors, size_cov));
        CUDA_CHECK(cudaMalloc(&d_cov_grad_mats_between_neighbors, size_cov_grad));
        CUDA_CHECK(cudaMalloc(&d_L, size_L));

        printf("Device memory allocated\n"); fflush(stdout);

        CalcCovFactorGradientVecchia_GPU << <blocksPerGrid, threadsPerBlock >> > (
            shape, C, n, dim_coords,
            coords, nn_ptr, nn_idx,
            jitter, nugget_var,
            B_data, D_data, B_grad_data, D_grad_data,
            pars, num_par, num_par_gp,
            gauss_likelihood, transf_scale,
            calc_cov_factor, calc_gradient,
            calc_gradient_nugget, exclude_marg_var_grad, ard, EPSILON_NUMBERS,
            d_cov_mat_between_neighbors, d_cov_grad_mats_between_neighbors, d_L
            );

        cudaFree(d_cov_mat_between_neighbors);
        cudaFree(d_cov_grad_mats_between_neighbors);
        cudaFree(d_L);

        // Check for launch configuration/argument errors
        printf("Thread0\n"); fflush(stdout);
        cudaError_t launchErr = cudaGetLastError();
        if (launchErr != cudaSuccess) {
            printf("Kernel launch failed: %s\n", cudaGetErrorString(launchErr)); fflush(stdout);
            return false;
        }
        printf("Thread0\n"); fflush(stdout);
        // Now wait for kernel to finish
        cudaError_t execErr = cudaDeviceSynchronize();
        if (execErr != cudaSuccess) {
            printf("Kernel execution error: %s\n", cudaGetErrorString(execErr)); fflush(stdout);
            return false;
        }
        printf("Thread0\n"); fflush(stdout);
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
