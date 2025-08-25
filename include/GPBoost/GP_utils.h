/*!
* This file is part of GPBoost a C++ library for combining
*	boosting with Gaussian process and mixed effects models
*
* Copyright (c) 2020 - 2024 Fabio Sigrist and Tim Gyger. All rights reserved.
*
* Licensed under the Apache License Version 2.0. See LICENSE file in the project root for license information.
*/
#ifndef GPB_GP_UTIL_H_
#define GPB_GP_UTIL_H_
#include <memory>
#include <GPBoost/type_defs.h>
#include <GPBoost/utils.h>
#include <LightGBM/utils/log.h>
#include <chrono>  // only for debugging
#include <thread> // only for debugging
#ifdef USE_CUDA_GP
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusparse.h>
#include <cusolverDn.h>
#endif

using LightGBM::Log;

namespace GPBoost {

	/*!
	* \brief Determine unique locations and map duplicates in coordinates to first occurance of unique locations
	* \param coords Coordinates
	* \param num_data Number of data points
	* \param[out] uniques Index of unique coordinates / points
	* \param[out] unique_idx Index that indicates for every data point the corresponding unique point. Used for constructing incidence matrix Z_ if there are duplicates
	*/
	void DetermineUniqueDuplicateCoords(const den_mat_t& coords,
		data_size_t num_data,
		std::vector<int>& uniques,
		std::vector<int>& unique_idx);

	/*!
	* \brief Determine unique locations and map duplicates in coordinates to first occurance of unique locations
	* \param coords Coordinates
	* \param num_data Number of data points
	* \param[out] uniques Index of unique coordinates / points
	* \param[out] unique_idx Index that indicates for every data point the corresponding unique point. Used for constructing incidence matrix Z_ if there are duplicates
	*/
	void DetermineUniqueDuplicateCoordsFast(const den_mat_t& coords,
		data_size_t num_data,
		std::vector<int>& uniques,
		std::vector<int>& unique_idx);

	/*!
	* \brief Calculate distance matrix (dense matrix)
	* \param coords1 First set of points
	* \param coords2 Second set of points
	* \param only_one_set_of_coords If true, coords1 == coords2, and dist is a symmetric square matrix
	* \param[out] dist Matrix of dimension coords2.rows() x coords1.rows() with distances between all pairs of points in coords1 and coords2 (rows in coords1 and coords2). Often, coords1 == coords2
	*/
	template <class T_mat, typename std::enable_if <std::is_same<den_mat_t, T_mat>::value>::type* = nullptr >
	void CalculateDistances(const den_mat_t& coords1,
		const den_mat_t& coords2,
		bool only_one_set_of_coords,
		den_mat_t& dist) {
		dist = den_mat_t(coords2.rows(), coords1.rows());
		dist.setZero();
#pragma omp parallel for schedule(static)
		for (int i = 0; i < coords2.rows(); ++i) {
			int first_j = 0;
			if (only_one_set_of_coords) {
				dist(i, i) = 0.;
				first_j = i + 1;
			}
			for (int j = first_j; j < coords1.rows(); ++j) {
				dist(i, j) = (coords2.row(i) - coords1.row(j)).lpNorm<2>();
			}
		}
		if (only_one_set_of_coords) {
			dist.triangularView<Eigen::StrictlyLower>() = dist.triangularView<Eigen::StrictlyUpper>().transpose();
		}
	}//end CalculateDistances (dense)
	template <class T_mat, typename std::enable_if <std::is_same<sp_mat_t, T_mat>::value || std::is_same<sp_mat_rm_t, T_mat>::value>::type* = nullptr >
	void CalculateDistances(const den_mat_t& coords1,
		const den_mat_t& coords2,
		bool only_one_set_of_coords,
		T_mat& dist) {
		std::vector<Triplet_t> triplets;
		int n_max_entry;
		if (only_one_set_of_coords) {
			n_max_entry = (int)(coords1.rows() - 1) * (int)coords2.rows();
		}
		else {
			n_max_entry = (int)coords1.rows() * (int)coords2.rows();
		}
		triplets.reserve(n_max_entry);
#pragma omp parallel for schedule(static)
		for (int i = 0; i < coords2.rows(); ++i) {
			int first_j = 0;
			if (only_one_set_of_coords) {
#pragma omp critical
				{
					triplets.emplace_back(i, i, 0.);
				}
				first_j = i + 1;
			}
			for (int j = first_j; j < coords1.rows(); ++j) {
				double dist_i_j = (coords2.row(i) - coords1.row(j)).lpNorm<2>();
#pragma omp critical
				{
					triplets.emplace_back(i, j, dist_i_j);
					if (only_one_set_of_coords) {
						triplets.emplace_back(j, i, dist_i_j);
					}
				}
			}
		}
		dist = T_mat(coords2.rows(), coords1.rows());
		dist.setFromTriplets(triplets.begin(), triplets.end());
		dist.makeCompressed();
	}//end CalculateDistances (sparse)

	/*!
	* \brief Calculate distance matrix when compactly supported covariance functions are used
	* \param coords1 First set of points
	* \param coords2 Second set of points
	* \param only_one_set_of_coords If true, coords1 == coords2, and dist is a symmetric square matrix
	* \param taper_range Range parameter of Wendland covariance function / taper beyond which the covariance is zero, and distances are thus not needed
	* \param show_number_non_zeros If true, the percentage of non-zero values is shown
	* \param[out] dist Matrix of dimension coords2.rows() x coords1.rows() with distances between all pairs of points in coords1 and coords2 (rows in coords1 and coords2). Often, coords1 == coords2
	*/
	template <class T_mat, typename std::enable_if <std::is_same<den_mat_t, T_mat>::value>::type* = nullptr >
	void CalculateDistancesTapering(const den_mat_t& coords1, //(this is a placeholder which is not used, only here for template compatibility)
		const den_mat_t& coords2,
		bool only_one_set_of_coords,
		double,
		bool,
		den_mat_t& dist) {
		CalculateDistances<T_mat>(coords1, coords2, only_one_set_of_coords, dist);
	}//end CalculateDistancesTapering (dense)
	template <class T_mat, typename std::enable_if <std::is_same<sp_mat_t, T_mat>::value || std::is_same<sp_mat_rm_t, T_mat>::value>::type* = nullptr >
	void CalculateDistancesTapering(const den_mat_t& coords1,
		const den_mat_t& coords2,
		bool only_one_set_of_coords,
		double taper_range,
		bool show_number_non_zeros,
		T_mat& dist) {
		std::vector<Triplet_t> triplets;
		int n_max_entry;
		if (only_one_set_of_coords) {
			n_max_entry = 30 * (int)coords1.rows();
		}
		else {
			n_max_entry = 10 * (int)coords1.rows() + 10 * (int)coords2.rows();
		}
		triplets.reserve(n_max_entry);
		//Sort along the sum of the coordinates
		int num_data;
		int dim_coords = (int)coords1.cols();
		double taper_range_square = taper_range * taper_range;
		if (only_one_set_of_coords) {
			num_data = (int)coords1.rows();
		}
		else {
			num_data = (int)(coords1.rows() + coords2.rows());
		}
		std::vector<double> coords_sum(num_data);
		std::vector<int> sort_sum(num_data);
		if (only_one_set_of_coords) {
#pragma omp parallel for schedule(static)
			for (int i = 0; i < num_data; ++i) {
				coords_sum[i] = coords1(i, Eigen::all).sum();
			}
		}
		else {
			den_mat_t coords_all(num_data, dim_coords);
			coords_all << coords2, coords1;
#pragma omp parallel for schedule(static)
			for (int i = 0; i < num_data; ++i) {
				coords_sum[i] = coords_all(i, Eigen::all).sum();
			}
		}
		SortIndeces<double>(coords_sum, sort_sum);
		std::vector<int> sort_inv_sum(num_data);
#pragma omp parallel for schedule(static)
		for (int i = 0; i < num_data; ++i) {
			sort_inv_sum[sort_sum[i]] = i;
		}
		// Search for and calculate distances that are smaller than taper_range
		//  using a fast approach based on results of Ra and Kim (1993)
#pragma omp parallel for schedule(static)
		for (int i = 0; i < coords2.rows(); ++i) {
			if (only_one_set_of_coords) {
#pragma omp critical
				{
					triplets.emplace_back(i, i, 0.);
				}
			}
			bool down = true;
			bool up = true;
			int up_i = sort_inv_sum[i];
			int down_i = sort_inv_sum[i];
			double smd, sed;
			while (up || down) {
				if (down_i == 0) {
					down = false;
				}
				if (up_i == (num_data - 1)) {
					up = false;
				}
				if (down) {
					down_i--;
					if ((only_one_set_of_coords && sort_sum[down_i] > i) ||
						(!only_one_set_of_coords && sort_sum[down_i] >= coords2.rows())) {
						smd = std::pow(coords_sum[sort_sum[down_i]] - coords_sum[i], 2);
						if (smd > dim_coords * taper_range_square) {
							down = false;
						}
						else {
							if (only_one_set_of_coords) {
								sed = (coords1(sort_sum[down_i], Eigen::all) - coords1(i, Eigen::all)).squaredNorm();
							}
							else {
								sed = (coords1(sort_sum[down_i] - coords2.rows(), Eigen::all) - coords2(i, Eigen::all)).squaredNorm();
							}
							if (sed < taper_range_square) {
								double dist_i_j = std::sqrt(sed);
#pragma omp critical
								{
									if (only_one_set_of_coords) {
										triplets.emplace_back(i, sort_sum[down_i], dist_i_j);
										triplets.emplace_back(sort_sum[down_i], i, dist_i_j);
									}
									else {
										triplets.emplace_back(i, sort_sum[down_i] - coords2.rows(), dist_i_j);
									}
								}
							}//end sed < taper_range_square
						}//end smd <= dim_coords * taper_range_square
					}
				}//end down
				if (up) {
					up_i++;
					if ((only_one_set_of_coords && sort_sum[up_i] > i) ||
						(!only_one_set_of_coords && sort_sum[up_i] >= coords2.rows())) {
						smd = std::pow(coords_sum[sort_sum[up_i]] - coords_sum[i], 2);
						if (smd > dim_coords * taper_range_square) {
							up = false;
						}
						else {
							if (only_one_set_of_coords) {
								sed = (coords1(sort_sum[up_i], Eigen::all) - coords1(i, Eigen::all)).squaredNorm();
							}
							else {
								sed = (coords1(sort_sum[up_i] - coords2.rows(), Eigen::all) - coords2(i, Eigen::all)).squaredNorm();
							}
							if (sed < taper_range_square) {
								double dist_i_j = std::sqrt(sed);
#pragma omp critical
								{
									if (only_one_set_of_coords) {
										triplets.emplace_back(i, sort_sum[up_i], dist_i_j);
										triplets.emplace_back(sort_sum[up_i], i, dist_i_j);
									}
									else {
										triplets.emplace_back(i, sort_sum[up_i] - coords2.rows(), dist_i_j);
									}
								}
							}//end sed < taper_range_square
						}//end smd <= dim_coords * taper_range_square
					}
				}//end up
			}//end while (up || down)
		}//end loop over data i

// Old, slow version
//#pragma omp parallel for schedule(static)
//		for (int i = 0; i < coords2.rows(); ++i) {
//			int first_j = 0;
//			if (only_one_set_of_coords) {
//#pragma omp critical
//				{
//					triplets.emplace_back(i, i, 0.);
//				}
//				first_j = i + 1;
//			}
//			for (int j = first_j; j < coords1.rows(); ++j) {
//				double dist_i_j = (coords2.row(i) - coords1.row(j)).lpNorm<2>();
//				if (dist_i_j < taper_range) {
//#pragma omp critical
//					{
//						triplets.emplace_back(i, j, dist_i_j);
//						if (only_one_set_of_coords) {
//							triplets.emplace_back(j, i, dist_i_j);
//						}
//					}
//				}
//			}
//		}
		
		dist = T_mat(coords2.rows(), coords1.rows());
		dist.setFromTriplets(triplets.begin(), triplets.end());
		dist.makeCompressed();
		if (show_number_non_zeros) {
			double prct_non_zero;
			int non_zeros = (int)dist.nonZeros();
			if (only_one_set_of_coords) {
				prct_non_zero = ((double)non_zeros) / coords1.rows() / coords1.rows() * 100.;
				int num_non_zero_row = non_zeros / (int)coords1.rows();
				Log::REInfo("Average number of non-zero entries per row in covariance matrix: %d (%g %%)", num_non_zero_row, prct_non_zero);
			}
			else {
				prct_non_zero = non_zeros / coords1.rows() / coords2.rows() * 100.;
				Log::REInfo("Number of non-zero entries in covariance matrix: %d (%g %%)", non_zeros, prct_non_zero);
			}
		}
	}//end CalculateDistancesTapering (sparse)

	/*!
	* \brief Subtract the inner product M^TM from a matrix Sigma
	* \param[out] Sigma Matrix from which M^TM is subtracted
	* \param M Matrix M
	* \param only_triangular true/false only compute triangular matrix
	*/
	template <class T_mat, typename std::enable_if <std::is_same<den_mat_t, T_mat>::value>::type* = nullptr >
	void SubtractInnerProdFromMat(T_mat& Sigma,
		const den_mat_t& M,
		bool only_triangular) {
		CHECK(Sigma.rows() == M.cols());
		CHECK(Sigma.cols() == M.cols());
#pragma omp parallel for schedule(static)
		for (int i = 0; i < Sigma.rows(); ++i) {
			for (int j = i; j < Sigma.cols(); ++j) {
				Sigma(i, j) -= M.col(i).dot(M.col(j));
				if (!only_triangular) {
					if (j > i) {
						Sigma(j, i) = Sigma(i, j);
					}
				}
			}
		}
	}//end SubtractInnerProdFromMat (dense)
	template <class T_mat, typename std::enable_if <std::is_same<sp_mat_t, T_mat>::value || std::is_same<sp_mat_rm_t, T_mat>::value>::type* = nullptr >
	void SubtractInnerProdFromMat(T_mat & Sigma,
		const den_mat_t & M,
		bool only_triangular) {
		CHECK(Sigma.rows() == M.cols());
		CHECK(Sigma.cols() == M.cols());
#pragma omp parallel for schedule(static)
		for (int k = 0; k < Sigma.outerSize(); ++k) {
			for (typename T_mat::InnerIterator it(Sigma, k); it; ++it) {
				int i = (int)it.row();
				int j = (int)it.col();
				if (i <= j) {
					it.valueRef() -= M.col(i).dot(M.col(j));
					if (!only_triangular) {
						if (i < j) {
							Sigma.coeffRef(j, i) = Sigma.coeff(i, j);
						}
					}
				}
			}
		}
	}//end SubtractInnerProdFromMat (sparse)

	/*!
	* \brief Subtract the product M1^T * M2 from a matrix Sigma
	* \param[out] Sigma Matrix from which M1^T * M2 is subtracted
	* \param M1 Matrix M1
	* \param M2 Matrix M2
	* \param only_triangular true/false only compute triangular matrix
	*/
	template <class T_mat, typename std::enable_if <std::is_same<den_mat_t, T_mat>::value>::type* = nullptr >
	void SubtractProdFromMat(T_mat& Sigma,
		const den_mat_t& M1,
		const den_mat_t& M2,
		bool only_triangular) {
		CHECK(Sigma.rows() == M1.cols());
		CHECK(Sigma.cols() == M2.cols());
#pragma omp parallel for schedule(static)
		for (int i = 0; i < Sigma.rows(); ++i) {
			for (int j = i; j < Sigma.cols(); ++j) {
				Sigma(i, j) -= M1.col(i).dot(M2.col(j));
				if (!only_triangular) {
					if (j > i) {
						Sigma(j, i) = Sigma(i, j);
					}
				}
			}
		}
	}//end SubtractProdFromMat (dense)
	template <class T_mat, typename std::enable_if <std::is_same<sp_mat_t, T_mat>::value || std::is_same<sp_mat_rm_t, T_mat>::value>::type* = nullptr >
	void SubtractProdFromMat(T_mat & Sigma,
		const den_mat_t & M1,
		const den_mat_t & M2,
		bool only_triangular) {
		CHECK(Sigma.rows() == M1.cols());
		CHECK(Sigma.cols() == M2.cols());
#pragma omp parallel for schedule(static)
		for (int k = 0; k < Sigma.outerSize(); ++k) {
			for (typename T_mat::InnerIterator it(Sigma, k); it; ++it) {
				int i = (int)it.row();
				int j = (int)it.col();
				if (i <= j) {
					it.valueRef() -= M1.col(i).dot(M2.col(j));
					if (!only_triangular) {
						if (i < j) {
							Sigma.coeffRef(j, i) = Sigma.coeff(i, j);
						}
					}
				}
			}
		}
	}//end SubtractProdFromMat (sparse)

	/*!
	* \brief Subtract the product M1^T * M2 from a matrix non square Sigma (prediction)
	* \param[out] Sigma Matrix from which M1^T * M2 is subtracted
	* \param M1 Matrix M1
	* \param M2 Matrix M2
	*/
	template <class T_mat, typename std::enable_if <std::is_same<den_mat_t, T_mat>::value>::type* = nullptr >
	void SubtractProdFromNonSqMat(T_mat& Sigma,
		const den_mat_t& M1,
		const den_mat_t& M2) {
		CHECK(Sigma.rows() == M1.cols());
		CHECK(Sigma.cols() == M2.cols());
#pragma omp parallel for schedule(static)
		for (int i = 0; i < Sigma.rows(); ++i) {
			for (int j = 0; j < Sigma.cols(); ++j) {
				Sigma(i, j) -= M1.col(i).dot(M2.col(j));
			}
		}
	}//end SubtractProdFromNonSqMat (dense)
	template <class T_mat, typename std::enable_if <std::is_same<sp_mat_t, T_mat>::value || std::is_same<sp_mat_rm_t, T_mat>::value>::type* = nullptr >
	void SubtractProdFromNonSqMat(T_mat & Sigma,
		const den_mat_t & M1,
		const den_mat_t & M2) {
		CHECK(Sigma.rows() == M1.cols());
		CHECK(Sigma.cols() == M2.cols());
#pragma omp parallel for schedule(static)
		for (int k = 0; k < Sigma.outerSize(); ++k) {
			for (typename T_mat::InnerIterator it(Sigma, k); it; ++it) {
				int i = (int)it.row();
				int j = (int)it.col();
				it.valueRef() -= M1.col(i).dot(M2.col(j));
			}
		}
	}//end SubtractProdFromNonSqMat (sparse)

	/*!
	* \brief Subtract the matrix from a matrix Sigma
	* \param[out] Sigma Matrix from which M is subtracted
	* \param M Matrix
	*/
	template <class T_mat, typename std::enable_if <std::is_same<den_mat_t, T_mat>::value>::type* = nullptr >
	void SubtractMatFromMat(T_mat& Sigma,
		const den_mat_t& M) {
#pragma omp parallel for schedule(static)
		for (int i = 0; i < Sigma.rows(); ++i) {
			for (int j = i; j < Sigma.cols(); ++j) {
				Sigma(i, j) -= M(i, j);
				if (j > i) {
					Sigma(j, i) = Sigma(i, j);
				}
			}
		}
	}//end SubtractMatFromMat (dense)
	template <class T_mat, typename std::enable_if <std::is_same<sp_mat_t, T_mat>::value || std::is_same<sp_mat_rm_t, T_mat>::value>::type* = nullptr >
	void SubtractMatFromMat(T_mat & Sigma,
		const den_mat_t & M) {
#pragma omp parallel for schedule(static)
		for (int k = 0; k < Sigma.outerSize(); ++k) {
			for (typename T_mat::InnerIterator it(Sigma, k); it; ++it) {
				int i = (int)it.row();
				int j = (int)it.col();
				if (i <= j) {
					it.valueRef() -= M(i, j);
					if (i < j) {
						Sigma.coeffRef(j, i) = Sigma.coeff(i, j);
					}
				}
			}
		}
	}//end SubtractMatFromMat (sparse)

	/*
	Calculate the smallest distance between each of the data points and any of the input means.
	* \param means data cluster means that determine the inducing points
	* \param data data coordinates
	* \param[out] distances smallest distance between each of the data points and any of the input means
	*/
	void closest_distance(const den_mat_t& means,
		const den_mat_t& data,
		vec_t& distances);

	/*
	This is an alternate initialization method based on the [kmeans++](https://en.wikipedia.org/wiki/K-means%2B%2B)
	initialization algorithm.
	* \param data data coordinates
	* \param k Size of inducing points
	* \param gen RNG
	* \param[out] means data cluster means that determine the inducing points
	*/
	void random_plusplus(const den_mat_t& data,
		int k,
		RNG_t& gen,
		den_mat_t& means);

	/*
	Calculate means based on data points and their cluster assignments.
	* \param data data coordinates
	* \param  clusters index of the mean each data point is closest to
	* \param[out] means data cluster means that determine the inducing points
	* \param[out] indices indices of closest data points to means
	*/
	void calculate_means(const den_mat_t& data,
		vec_t& clusters,
		den_mat_t& means,
		vec_t& indices);

	/*
	This implementation of k-means uses [Lloyd's Algorithm](https://en.wikipedia.org/wiki/Lloyd%27s_algorithm)
	with the [kmeans++](https://en.wikipedia.org/wiki/K-means%2B%2B) used for initializing the means.
	* \param data Coordinates / input features
	* \param k Number of cluster centers (usually = inducing points)
	* \param gen RNG
	* \param[out] means data cluster means that determine the inducing points
	* \param[out] max_int maximal number of iterations
	*/
	void kmeans_plusplus(const den_mat_t& data,
		int k,
		RNG_t& gen,
		den_mat_t& means,
		int max_it);

	/*
	Determines indices of data which is inside a ball with given radius around given point
	* \param data data coordinates
	* \param indices_start indices of data considered
	* \param radius radius of ball
	* \param mid centre of ball
	* \param[out] indices indices of data points inside ball
	*/
	void data_in_ball(const den_mat_t& data,
		const std::vector<int>& indices_start,
		double radius,
		const vec_t& mid,
		std::vector<int>& indices);

	/*
	CoverTree Algorithmus
	* \param data data coordinates
	* \param eps size of cover part
	* \param gen RNG
	* \param[out] means data cluster means that determine the inducing points
	*/
	void CoverTree(const den_mat_t& data,
		double eps,
		RNG_t& gen,
		den_mat_t& means);

	/*!
	* \brief Matrix-multiplication A * B = C
	* \param A First Matrix
	* \param B Second Matrix
	* \param[out] C = A * B
	* \param GPU_use if false Use CPU 
	*/
	void matmul(const den_mat_t& A, const den_mat_t& B, den_mat_t& C, bool GPU_use);

	void spmatmul(const sp_mat_rm_t& A, const sp_mat_rm_t& B, sp_mat_rm_t& C, bool GPU_use);
	/*!
	* \brief Matrix-multiplication D * B = C
	* \param D Diagonal Matrix as Vector
	* \param B Matrix
	* \param[out] C = A * B
	* \param GPU_use if false Use CPU
	*/
	void diag_dense_matmul(const vec_t& D, const den_mat_t& B, den_mat_t& C, bool GPU_use);

	/*!
	* \brief Sparse-Dense-Matrix-multiplication A * B = C
	* \param A First Matrix
	* \param B Second Matrix
	* \param[out] C = A * B
	* \param GPU_use if false Use CPU
	*/
	void sparse_dense_matmul(const sp_mat_rm_t& A, const den_mat_t& B, den_mat_t& C, bool GPU_use);

	/*!
	* \brief Linear solve L^{-1} * R = X for given Cholesky factor L 
	* \param chol Cholesky factor L
	* \param R_host Right-hand side
	* \param[out] X = L^{-1} * R
	* \param GPU_use if false Use CPU
	*/
	void solve_lower_triangular(const chol_den_mat_t& chol, const den_mat_t& R_host, den_mat_t& X_host, bool GPU_use);

	/*!
	* \brief Solves Sigma^{-1} * R = X for given Cholesky factor L of Sigma
	* \param chol Cholesky factor L
	* \param R_host Right-hand side
	* \param[out] X = Sigma^{-1} * R
	* \param GPU_use if false Use CPU
	*/
	//void solve_linear_sys(const chol_den_mat_t& chol, const den_mat_t& R_host, den_mat_t& X_host, bool GPU_use);

#ifdef USE_CUDA_GP

	void launch_subtract_prod_from_mat_kernel(
		const double* M1, const double* M2, double* Sigma,
		int M1_rows, int M1_cols,
		int M2_rows, int M2_cols,
		bool only_triangular);

	void launch_subtract_sparse_kernel(
		const int* row_ptr, const int* col_idx, double* values,
		const double* M1, const double* M2,
		int n, int m, int K, bool only_triangular);


	// Host function
	template <class T_mat, typename std::enable_if <std::is_same<den_mat_t, T_mat>::value>::type* = nullptr >
	bool try_SubtractProdFromMat_CUDA(T_mat& Sigma,
		const den_mat_t& M1,
		const den_mat_t& M2,
		bool only_triangular)
	{

		const int n = Sigma.rows();
		const int m = Sigma.cols();
		const int k = M1.rows();  // Inner dimension
		if (n != M1.cols() || m != M2.cols()) {
			return false;
		}
		size_t size_M1 = sizeof(double) * k * n;
		size_t size_M2 = sizeof(double) * k * m;
		size_t size_Sigma = sizeof(double) * n * m;

		double* d_M1;
		double* d_M2;
		double* d_Sigma;

		cudaMalloc(&d_M1, size_M1);
		cudaMalloc(&d_M2, size_M2);
		cudaMalloc(&d_Sigma, size_Sigma);

		cudaMemcpy(d_M1, M1.data(), size_M1, cudaMemcpyHostToDevice);
		cudaMemcpy(d_M2, M2.data(), size_M2, cudaMemcpyHostToDevice);
		cudaMemcpy(d_Sigma, Sigma.data(), size_Sigma, cudaMemcpyHostToDevice);

		launch_subtract_prod_from_mat_kernel (
			d_M1, d_M2, d_Sigma,
			k, n, k, m,
			only_triangular
			);

		cudaMemcpy(Sigma.data(), d_Sigma, size_Sigma, cudaMemcpyDeviceToHost);

		cudaFree(d_M1);
		cudaFree(d_M2);
		cudaFree(d_Sigma);

		Log::REInfo("[GPU] Subtract product with cuBLAS.");
		return true;
	}
	template <class T_mat, typename std::enable_if <std::is_same<sp_mat_t, T_mat>::value || std::is_same<sp_mat_rm_t, T_mat>::value>::type* = nullptr>
	bool try_SubtractProdFromMat_CUDA(T_mat & Sigma,
			const den_mat_t & M1,
			const den_mat_t & M2,
			bool only_triangular)
	{
		
		const int n = Sigma.rows();
		const int m = Sigma.cols();
		const int K = M1.rows();

		if (n != M1.cols() || m != M2.cols()) {
			return false;
		}

		//Sigma.makeCompressed();
		const int nnz = Sigma.nonZeros();
		const int* h_row_ptr = Sigma.outerIndexPtr();
		const int* h_col_idx = Sigma.innerIndexPtr();
		const double* h_values = Sigma.valuePtr();

		// Device memory
		int* d_row_ptr = nullptr, * d_col_idx = nullptr;
		double* d_values = nullptr, * d_M1 = nullptr, * d_M2 = nullptr;

		cudaMalloc(&d_row_ptr, (n + 1) * sizeof(int));
		cudaMalloc(&d_col_idx, nnz * sizeof(int));
		cudaMalloc(&d_values, nnz * sizeof(double));
		cudaMalloc(&d_M1, K * n * sizeof(double));
		cudaMalloc(&d_M2, K * m * sizeof(double));

		// Copy data to device
		cudaMemcpy(d_row_ptr, h_row_ptr, (n + 1) * sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(d_col_idx, h_col_idx, nnz * sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(d_values, h_values, nnz * sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(d_M1, M1.data(), K * m * sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(d_M2, M2.data(), K * m * sizeof(double), cudaMemcpyHostToDevice);
		
		// Kernel launch
		launch_subtract_sparse_kernel(
			d_row_ptr, d_col_idx, d_values,
			d_M1, d_M2, n, m, K, only_triangular
			);
		//cudaDeviceSynchronize();
		// Copy result back
		cudaMemcpy((void*)h_values, d_values, nnz * sizeof(double), cudaMemcpyDeviceToHost);
		
		// Free device memory
		cudaFree(d_row_ptr);
		cudaFree(d_col_idx);
		cudaFree(d_values);
		cudaFree(d_M1);
		cudaFree(d_M2);

		// Mirror for full matrix if needed
		if (!only_triangular) {
#pragma omp parallel for schedule(static)
			for (int k = 0; k < Sigma.outerSize(); ++k) {
				for (typename T_mat::InnerIterator it(Sigma, k); it; ++it) {
					int i = it.row();
					int j = it.col();
					if (i < j) {
						Sigma.coeffRef(j, i) = Sigma.coeff(i, j);
					}
				}
			}
		}
		
		Log::REInfo("[GPU] Subtracted M1^T * M2 from sparse Sigma.");
		return true;
	}

	template <class T_mat>
	void SubtractProdFromMatrix(T_mat& Sigma, const den_mat_t& M1, const den_mat_t& M2, bool only_triangular, bool GPU_use) {
		if (!GPU_use) {
			Log::REInfo("[Fallback] Forced Eigen matrix-multiplication.");
			SubtractProdFromMat<T_mat>(Sigma, M1, M2, only_triangular);
			return;
		}
		
		int device_count = 0;
		cudaError_t err = cudaGetDeviceCount(&device_count);
		if (err != cudaSuccess || device_count == 0) {
			Log::REInfo("[Fallback] No CUDA devices found. Using Eigen for subtract Matrix product.");
			SubtractProdFromMat<T_mat>(Sigma, M1, M2, only_triangular);
			GPU_use = false;
			return;
		}
		if (!try_SubtractProdFromMat_CUDA(Sigma,M1,M2,only_triangular)) {
			Log::REInfo("[Fallback] Error in computation on GPU. Using Eigen for subtract Matrix product.");
			SubtractProdFromMat<T_mat>(Sigma, M1, M2, only_triangular);
		}
	}

	template <class T_chol, typename std::enable_if <std::is_same<chol_den_mat_t, T_chol>::value>::type* = nullptr >
	bool try_solve_cholesky_gpu(const T_chol& chol, const den_mat_t& R_host, den_mat_t& X_host) {
		den_mat_t L_host = chol.matrixL();  // L from LL^T
		int n = L_host.rows();
		int m = R_host.cols();

		if (L_host.cols() != n || R_host.rows() != n) {
			return false;
		}
		X_host.resize(n, m);
		// Allocate memory
		double* d_L = nullptr;
		double* d_Y = nullptr;
		double* d_X = nullptr;

		cudaMalloc(&d_L, n * n * sizeof(double));
		cudaMalloc(&d_Y, n * m * sizeof(double));
		cudaMalloc(&d_X, n * m * sizeof(double));

		cudaMemcpy(d_L, L_host.data(), n * n * sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(d_Y, R_host.data(), n * m * sizeof(double), cudaMemcpyHostToDevice);  // Start Y = R

		// Create cuBLAS handle
		cublasHandle_t handle;
		cublasCreate(&handle);

		const double alpha = 1.0;

		// Step 1: Solve L * Y = R
		cublasStatus_t stat1 = cublasDtrsm(
			handle,
			CUBLAS_SIDE_LEFT,
			CUBLAS_FILL_MODE_LOWER,
			CUBLAS_OP_N,
			CUBLAS_DIAG_NON_UNIT,
			n, m,
			&alpha,
			d_L, n,
			d_Y, n  // In-place
		);

		if (stat1 != CUBLAS_STATUS_SUCCESS) {
			cudaFree(d_L); cudaFree(d_Y); cudaFree(d_X);
			cublasDestroy(handle);
			return false;
		}

		// Step 2: Solve L^T * X = Y
		cudaMemcpy(d_X, d_Y, n * m * sizeof(double), cudaMemcpyDeviceToDevice);  // Copy Y into X

		cublasStatus_t stat2 = cublasDtrsm(
			handle,
			CUBLAS_SIDE_LEFT,
			CUBLAS_FILL_MODE_LOWER,
			CUBLAS_OP_T,  // Transpose
			CUBLAS_DIAG_NON_UNIT,
			n, m,
			&alpha,
			d_L, n,
			d_X, n  // In-place
		);

		if (stat2 != CUBLAS_STATUS_SUCCESS) {
			cudaFree(d_L); cudaFree(d_Y); cudaFree(d_X);
			cublasDestroy(handle);
			return false;
		}

		// Copy result back
		X_host.resize(n, m);
		cudaMemcpy(X_host.data(), d_X, n * m * sizeof(double), cudaMemcpyDeviceToHost);

		// Cleanup
		cudaFree(d_L);
		cudaFree(d_Y);
		cudaFree(d_X);
		cublasDestroy(handle);

		Log::REInfo("[GPU] Full Cholesky solve (Sigma^-1 * R) with cuBLAS.");
		return true;
	}
	template <class T_chol, typename std::enable_if <std::is_same<chol_sp_mat_t, T_chol>::value || std::is_same<chol_sp_mat_rm_t, T_chol>::value>::type* = nullptr >
	bool try_solve_cholesky_gpu(const T_chol & chol, const den_mat_t & R_host, den_mat_t & X_host) {
#pragma omp parallel for schedule(static)   
		for (int i = 0; i < R_host.cols(); ++i) {
			X_host.col(i) = chol.solve(R_host.col(i));
		}
		return true;
	}


	template <class T_chol>
	void solve_linear_sys(const T_chol& chol, const den_mat_t& R_host, den_mat_t& X_host, bool GPU_use) {
		if (!GPU_use) {
			Log::REInfo("[Fallback] Forced Eigen matrix-multiplication.");
			X_host = chol.solve(R_host);
			return;
		}
		int device_count = 0;
		cudaError_t err = cudaGetDeviceCount(&device_count);
		if (err != cudaSuccess || device_count == 0) {
			Log::REInfo("[Fallback] No CUDA devices found. Using Eigen for matrix-multiplication.");
			X_host = chol.solve(R_host);
			GPU_use = false;
			return;
		}

		if (!try_solve_cholesky_gpu(chol, R_host, X_host)) {
			Log::REInfo("[Fallback] Error in computation on GPU. Using Eigen for matrix-multiplication.");
			X_host = chol.solve(R_host);
		}
	}

#else

template <class T_chol>
void solve_linear_sys(const T_chol& chol, const den_mat_t& R_host, den_mat_t& X_host, bool GPU_use) {
	if (GPU_use) {
		Log::REInfo("[Fallback] Not able to compile CUDA Code. Continuing with CPU support.");
		GPU_use = false;
	}
	X_host = chol.solve(R_host);
}

template <class T_mat>
void SubtractProdFromMatrix(T_mat& Sigma, const den_mat_t& M1, const den_mat_t& M2, bool only_triangular, bool GPU_use) {
	if (GPU_use) {
		Log::REInfo("[Fallback] Not able to compile CUDA Code. Continuing with CPU support.");
		GPU_use = false;
	}
	SubtractProdFromMat<T_mat>(Sigma, M1, M2, only_triangular);
}

#endif  // USE_CUDA_GP



	/*!
	* \brief Cholesky factor of A_input = LL^T
	* \param[out] llt Cholesky factor L
	* \param A_input Matrix
	* \param GPU_use if false Use CPU
	*/
	void cholesky_solver(chol_den_mat_t& llt, const den_mat_t& A_input, bool GPU_use);

}  // namespace GPBoost

#endif   // GPB_GP_UTIL_H_
