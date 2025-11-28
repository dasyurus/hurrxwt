#include "csv.h"
#include <unistd.h>
#include <math.h>
#include <stdlib.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <cstdio>
#include <hip/hip_runtime.h>
#include <tuple>

#include <iomanip>
#include <sstream>
#include <fstream>
#include <sys/stat.h> //for mkdir

#define NL 256

hipError_t err;

// Function to multiply two complex numbers, a * conj(b)
__device__ float c_mult_conj_abs(float2 a, float2 b) {
	float2 out;
	out.x = a.x * b.x + a.y * b.y;
	out.y = -a.x * b.y + a.y * b.x;
	return sqrt(out.x*out.x + out.y*out.y);
}

// Function to multiply two complex numbers
__device__ float2 c_mult(float2 a, float2 b) {
	float2 out;
	out.x = a.x * b.x - a.y * b.y;
	out.y = a.x * b.y + a.y * b.x;
	return out;
}

// Scale kernel
__global__ void scale(float2* data, long long MN, int N) {
	long long idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < MN) {
	    data[idx].x /= N;
	    data[idx].y /= N;
	}
}

// Conjugate kernel
__global__ void conj(float2* data, long long MN) {
	long long idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < MN) {
	    data[idx].y *= -1;
	}
}

// FFT kernel
__global__ void fft_kernel(float2* data, int N, int logN, int M, const float2* omega, const int* i_to_j, const int* kj, const int* kjm, int loopSize) {
	int idx_M = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx_M < M) {
	    float2 temp[NL]; // Adjust the size based on N (here assumed to be at most 256; change if N changes)
	    
	    for (int i = 0; i < N; i++) { 
	        int idx = idx_M * N + i;
	        temp[i_to_j[i]] = data[idx];
	    }

	    for (int s = 0; s < logN; s++) { 
	        for (int w = 0; w < loopSize; w++) { 
	            int idx = s * loopSize + w;

	            float2 t = c_mult(omega[idx], temp[kjm[idx]]);
	            float2 u = temp[kj[idx]];

	            temp[kj[idx]] = { u.x + t.x, u.y + t.y };
	            temp[kjm[idx]] = { u.x - t.x, u.y - t.y };
	        }
	    }

	    for (int i = 0; i < N; i++) {
	        int idx = idx_M * N + i;
	        data[idx] = temp[i];
	    }
	}
}
// FFT kernel
__global__ void fft_kernel2(float* data, int N, int padN, int log_padN, int M, const float2* omega, const int* i_to_j, const int* kj, const int* kjm, int loopSize, float2 * out) {
	int idx_M = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx_M < M) {
	    float2 temp[NL]; // these have to be constants
	    
	float2 scratch[NL];
	    for (int i = 0; i < padN; i++) { 
	        int idx = idx_M * N + i;
	    scratch[i] = {0.0, 0.0};
	    if (i<N) scratch[i].x = data[idx];
	}
	for (int i = 0; i < padN; i++) {
	        temp[i_to_j[i]] = scratch[i];
	    }

	    for (int s = 0; s < log_padN; s++) { 
	        for (int w = 0; w < loopSize; w++) { 
	            int idx = s * loopSize + w;

	            float2 t = c_mult(omega[idx], temp[kjm[idx]]);
	            float2 u = temp[kj[idx]];

	            temp[kj[idx]] = { u.x + t.x, u.y + t.y };
	            temp[kjm[idx]] = { u.x - t.x, u.y - t.y };
	        }
	    }
	    for (int i = 0; i < padN; i++) {
	        int idx = idx_M * padN + i;
	        out[idx] = temp[i];
	    }
	
	}
}

// also calculate fft, but across two dimensions
__global__ void fft_kernel2d(float2* data, int N, int padN, int log_padN, int M, const float2* omega, const int* i_to_j, const int* kj, const int* kjm, int loopSize) {
	int idx_M = blockIdx.x * blockDim.x + threadIdx.x;
	int idx_f = blockIdx.y * blockDim.y + threadIdx.y;

	if (idx_M < M) {
	    float2 temp[NL]; // these have to be constants
	    
	for (int i = 0; i < padN; i++) {
	        int idx = idx_M * N*padN + idx_f * padN + i;
	        temp[i_to_j[i]] = data[idx];
	    }

	    for (int s = 0; s < log_padN; s++) { 
	        for (int w = 0; w < loopSize; w++) { 
	            int idx = s * loopSize + w;

	            float2 t = c_mult(omega[idx], temp[kjm[idx]]);
	            float2 u = temp[kj[idx]];

	            temp[kj[idx]] = { u.x + t.x, u.y + t.y };
	            temp[kjm[idx]] = { u.x - t.x, u.y - t.y };
	        }
	    }
	    for (int i = 0; i < padN; i++) {
	        int idx = idx_M * N*padN + idx_f * padN + i;
	        data[idx] = temp[i];
	    }
	
	}
}

__device__ __host__ float rad(float in) {
	return 3.14159265 * in / 180;
}

__device__ __host__ float deg(float in) {
	return 180*in/ 3.14159265 ;
}

__device__ __host__ void fit_line_first_principles(const float *x_points, const float *y_points, size_t n, float *m, float *b) {
	// calculate means
	float sum_x = 0.0;
	float sum_y = 0.0;
	for (size_t i = 0; i < n; ++i) {
	    sum_x += x_points[i];
	    sum_y += y_points[i];
	}
	float mean_x = sum_x / n;
	float mean_y = sum_y / n;

	// now get slope and y-int
	float numerator = 0.0;
	float denominator = 0.0;
	for (size_t i = 0; i < n; ++i) {
	    float x_diff = x_points[i] - mean_x;
	    float y_diff = y_points[i] - mean_y;
	    numerator += x_diff * y_diff;
	    denominator += x_diff * x_diff;
	}

	*m = numerator / denominator;
	*b = mean_y - (*m) * mean_x;

}

__device__ __host__ float standard_deviation(const float *data, size_t n) {

	// mean
	float sum = 0.0;
	for (size_t i = 0; i < n; ++i) {
	    sum += data[i];
	}
	float mean = sum / n;

	// squared differences from mean
	float squared_diffs_sum = 0.0;
	for (size_t i = 0; i < n; ++i) {
	    float diff = data[i] - mean;
	    squared_diffs_sum += diff * diff;
	}

	// rest
	float variance = squared_diffs_sum / (n - 1);
	float std_dev = sqrt(variance);

	return std_dev;
}

// scale each wind speed measurement by distance from a "home" coordinates 
__global__ void scale_wind2(float * lat, float * lon, int aa, int * wind, float * wind_out, const int N, const float d_max, const float lat_increment, const float lon_increment, const int lat_range,  const int lon_range_batched, const float lat_start, const float lon_start ){
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int d_lat = threadIdx.y + blockIdx.y * blockDim.y;
	int d_lon = threadIdx.z + blockIdx.z * blockDim.z;
	long long idx = (long long)d_lat * N* lon_range_batched + (long long)d_lon*N + (long long)i;
	if (i <N && d_lat < lat_range && d_lon < lon_range_batched) {
						
		d_lon += aa * lon_range_batched;
		float home_lat = rad((float)d_lat*lat_increment+lat_start);
		float home_lon = -rad((float)d_lon*lon_increment+lon_start); // note negative sign, we are in western hemisphere
		float dlon = home_lon - lon[i];
		float dlat = home_lat - lat[i];
		float a = pow(sin(dlat/2),2) + cos(lat[i]) * cos(home_lat) * pow(sin(dlon/2),2);
		float r =  2 * asin(sqrt(a)) * 6371.0f;

		if (r>d_max) {
			wind_out[idx] = 0.0f;
		}
		else {
			wind_out[idx] = (float)wind[i] * (1.0f- r / d_max);
		}

	}


}

// reduce into a time series for a given home coordinates
__global__ void make_ts2(const int aa, const int *yearidx, const int Ny, const int N, float * ts_map, const float *wind, int * ts_map_d, const int lat_range, const int lon_range, const int lon_range_batched)
{
	int i = threadIdx.z + blockIdx.z * blockDim.z;
	int d_lat = threadIdx.x + blockIdx.x * blockDim.x;
	int d_lon = threadIdx.y + blockIdx.y * blockDim.y;
	int min = yearidx[i];
	int max = yearidx[i+1];
	if (i <Ny && d_lat < lat_range && d_lon < lon_range_batched) {
		int d_lon2 = d_lon+ aa * lon_range_batched;
		int idx = d_lat * Ny * lon_range + d_lon2 * Ny +i;
		long long idx_w = (long long)d_lat * N* lon_range_batched + (long long)d_lon*N;
		float s  = 0.0;
		for (int j = min; j<max; j++) {
			s += wind[idx_w+(long long)j];
			// note how many hits we have per year
			// this is reduced in reduce_ts_map_d() for # of hits for each position 
			if (wind[idx_w+(long long)j] > 0.0) {
				ts_map_d[idx]++;
			}

		}
		ts_map[idx] = s;
	}
}

// calculate how many hits there are at each position in total
// we previously produces hits per year per position
__global__ void reduce_ts_map_d(const int * ts_map_d, int * ts_map_d3, const int N, const int Nll) {
	int i = threadIdx.x + blockDim.x*blockIdx.x;
	if (i<Nll) {
		int idx = N * i;
		for (int j = 0; j < N; j++) {
			ts_map_d3[i] += ts_map_d[idx+j];
		}

	}
}

// reduce to sort of a sparse storage of the time series array
// treat ts_map as 1-d array
__global__ void recast_ts_map(float * ts_map_1d, const int ts_map_1d_size, const int N, const int * idx_conversion, const float * ts_map) {

	int i = threadIdx.x + blockDim.x*blockIdx.x;
	int j = threadIdx.y + blockDim.y*blockIdx.y;
	if (i<ts_map_1d_size && j < N) {
		int idx_1d = i * N + j;
		int idx = idx_conversion[i]*N + j;
		ts_map_1d[idx_1d] = ts_map[idx];
		
	}
}

// detrend (many) timeseries 
__global__ void detrend(float * ts_map_1d, const float * t, const int ts_map_1d_size, const int N) {
	int i = threadIdx.x + blockDim.x*blockIdx.x;

	if (i < ts_map_1d_size) {

		float x[NL];
		int idx_r =  i * N;
		for (int ii = 0 ; ii < N; ii++)
			x[ii] = ts_map_1d[idx_r+ii];
		
		float m, b;
		fit_line_first_principles(t,x,N,&m,&b);
		for (int ii =0 ; ii<N; ii++) {
			x[ii] = x[ii] - (m*t[ii]+b);
		}
		float x_std = standard_deviation(x,N);
		for (int ii =0 ; ii<N; ii++) {
			x[ii] = x[ii] / x_std;
		}
		for (int ii =0 ; ii<N; ii++) {
			ts_map_1d[idx_r+ii] = x[ii];
		}
		
		// useful to pick out specific time series
		//if (i==0 && ts_map_1d_size==1) {
		//if (i==15176) {
		//	for (int ii=0; ii<N; ii++) printf("%.7f\n", ts_map_1d[i*N+ii]);
		//}
	}
}

// calculate array W for the convolution operations within a CWT
__global__ void make_W(float2 * W, float2 * ts_map_1d_f, float * psi_ft_bar, int ts_map_1d_size, int N, int padN) {

	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	int k = threadIdx.z + blockIdx.z * blockDim.z;

	if (i < ts_map_1d_size && j < N && k < padN) {
		int idx_W = i * N * padN + j * padN + k;
		int idx_ts = i * padN + k;
		int idx_psi = j * padN + k;

		W[idx_W].x = ts_map_1d_f[idx_ts].x * psi_ft_bar[idx_psi];
		W[idx_W].y = ts_map_1d_f[idx_ts].y * psi_ft_bar[idx_psi];
			

	}
}

//  Allen and Smith autoregressive lag-1 autocorrelation coefficient
//  There is no safety on the discriminant - if D<0, it will throws nans
//  original function subtracs mean, but we're always feeding in detrended series,
//  so mean is already 0.
__global__ void make_alpha(float * alpha, float * ts_map_1d, int ts_map_1d_size, int N) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i < ts_map_1d_size) {
		int idx = i * N;
		float c0 = 0.0, c1 = 0.0;
		for (int j = 0; j<N; j++) {
			c0 += ts_map_1d[idx+j] * ts_map_1d[idx+j];
		}
		c0 /= N;
		for (int j =0; j<N-1; j++) {
			c1 += ts_map_1d[idx+j] * ts_map_1d[idx+j+1];
		}		
		c1 /= (N-1);
		float B = -c1 * N - c0 * N*N - 2 * c0 + 2 * c1 - c1 * N*N + c0 * N;
			float A = c0 * N* N;
			float C = N * (c0 + c1 * N - c1);
			float D = B*B - 4 * A * C;

  		alpha[i] = (-B - sqrt(D)) / (2 * A);

	}
}

// lag-1 autoregressive theoretical power spectrum
__global__ void make_Pk(float * Pk, float *freqs, float *alpha, int ts_map_1d_size, int N, float dt) {
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	if (i < ts_map_1d_size && j < N) {
		int idx = i * N + j;
		float a = alpha[i];
		float f = freqs[j]*dt;
		Pk[idx] = (1-a*a) / (a*a - 2 * a * cos(2*M_PI*f) +1);
	}
}

// calculate xwt coefficients, and divide by significance level
// a coordinate is significant when >=1.
__global__ void make_xsig95(float * xsig95, float2 * W, float2 * Wx, int N, int padN, int ts_map_1d_size, int driver_idx, float * Pk1, float * Pk2, float PPF_over_dof) { 
	// assume 1 driver at a time (memory constraint more than anything)
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	int k = threadIdx.z + blockIdx.z * blockDim.z;

	if (i < ts_map_1d_size && j < N && k < N) { // k == padN dimension which we'll trim to N
		int idx_W = i * N * padN + j * padN + k;
		int idx_Wx = driver_idx * N * padN + j * padN + k;
		int idx_xsig95 = i * N * N + j * N + k;
		int idx_Pk1 = i * N + j;
		int idx_Pk2 = driver_idx * N + j;

		float xsignif = sqrt((Pk1[idx_Pk1] * Pk2[idx_Pk2])) * PPF_over_dof;

		xsig95[idx_xsig95] = c_mult_conj_abs(W[idx_W], Wx[idx_Wx]) / xsignif;
	}

}

// calculate amount of signficant period-time coordinates
__global__ void reduce_xsig95(float * xwtred, float * xsig95, int ts_map_1d_size, int N) {
	int i = threadIdx.x + blockDim.x*blockIdx.x;
	if (i<ts_map_1d_size) {
		xwtred[i]=0.0;
		for (int j=0; j<N; j++) {
			for (int k=0; k<N; k++) {
				int idx = i*N*N + j*N + k;
				if (xsig95[idx]>=1.0) xwtred[i] += 1.0;
			}
		}

	}
}

// initialize to 0
__global__ void fillBuffer(float* d_buffer, float value, int size) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < size) {
	    d_buffer[idx] = value;
	}
}

// initialize to 0
__global__ void fillBuffer_d(int* d_buffer, int value, int size) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < size) {
	    d_buffer[idx] = value;
	}
}

void fillBufferV(float* d_buffer, float value, int size) {
	int blockSize = 256; // Define block size
	int numBlocks = (size + blockSize - 1) / blockSize; // Calculate number of blocks
	fillBuffer<<<numBlocks, blockSize>>>(d_buffer, value, size);

}

// initialize to a given value
__global__ void fillBuffer(int* d_buffer, int value, int size) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < size) {
	    d_buffer[idx] = value;
	}
}

void fillBufferV(int* d_buffer, int value, int size) {
	int blockSize = 256; // Define block size
	int numBlocks = (size + blockSize - 1) / blockSize; // Calculate number of blocks
	fillBuffer_d<<<numBlocks, blockSize>>>(d_buffer, value, size);

}

int sum(std::vector<int> in) {
	int s = 0;
	for (unsigned int i = 0; i < in.size(); i++) 
		s += in[i];
	return s;
}

float sum(std::vector<float> in) {
	float s = 0;
	for (unsigned int i = 0; i < in.size(); i++) 
		s += in[i];
	return s;
}

int get_min(std::vector<int> in) {
	int out = in[0];
	
	for (unsigned int i = 1; i<in.size(); i++) { // we had in[0] earlier
		if (in[i] < out) {
			out = in[i];
		}
	}
	return out;
}

int get_max(std::vector<int> in) {
	int out = in[0];
	
	for (unsigned int i = 1; i<in.size(); i++) {
		if (in[i] > out) {
			out = in[i];
		}
	}
	
	return out;
}

// calculate at what indices in the main hurricanes dataset
// the cutoffs for each year are
std::vector<int> get_yearidx(std::vector<int> years) {
	// we assume that this is already sorted, which the
	// the dataset is.
	int min = get_min(years);
	int max = get_max(years);
	std::vector<int> out;
	out.resize(max - min + 2);
	
	out[0] = 0;
	int current_year = min;
	int j = 0;
	
	for (unsigned int i = 1; i < years.size(); i++) {
		if (years[i] != current_year) {
			current_year = years[i];
			j++;
			out[j] = i;
		}
	}
	out[out.size()-1] = years.size();

	return out;
		
}

float get_mean(std::vector<float> in)
{
	return sum(in)/in.size();
}

int get_work_size(int global, int local) {
	return (global + local -1)/local * local; // last * local not needed in cuda/hip!
}
long long get_work_size_lld(long long global, int local) {
	return (global + local -1)/local * local; // last * local not needed in cuda/hip!
}

// geometrically spaced array, to determine periods for CWTs
std::vector<float> geomspace(float start, float end, size_t num) {
	std::vector<float> result;
	result.reserve(num);
	if (num == 1) {
	    result.push_back(start);
	    return result;
	}
	float log_start = std::log(start);
	float log_end = std::log(end);
	for (size_t i = 0; i < num; ++i) {
	    float ratio = static_cast<float>(i) / (num - 1);
	    result.push_back(std::exp(log_start + ratio * (log_end - log_start)));
	}
	return result;
}

// calculate cwt frequencies
std::vector<float> make_ftfreqs(int N, float dt) {
	// Generate kplus: 1 to N/2
	size_t N_half = N / 2;
	std::vector<float> kplus;
	kplus.reserve(N_half);
	for (size_t i = 1; i <= N_half; ++i) {
			kplus.push_back(i * 2.0 * M_PI / (N * dt));
	}

	// Generate kminus: 1 to (N-1)/2, then sort -kminus
	size_t N_minus = (N - 1) / 2;
	std::vector<float> kminus;
	kminus.reserve(N_minus);
	for (size_t i = 1; i <= N_minus; ++i) {
		kminus.push_back(-static_cast<float>(i) * 2.0 * M_PI / (N * dt));
	}

	// Sort kminus
	std::sort(kminus.begin(), kminus.end());

	// Concatenate 0, kplus, kminus into ftfreqs
	std::vector<float> ftfreqs;
	ftfreqs.reserve(1 + kplus.size() + kminus.size());
	ftfreqs.push_back(0.0);
	ftfreqs.insert(ftfreqs.end(), kplus.begin(), kplus.end());
	ftfreqs.insert(ftfreqs.end(), kminus.begin(), kminus.end());

	return ftfreqs;
}

float psi_ft(float in, float f0) {
	return (pow(M_PI, -0.25)) * exp(-0.5 * (in - f0) * (in -f0));
}

double2 h_c_mult(double2 a, double2 b) {
	double2 out;
	out.x = a.x * b.x - a.y * b.y;
	out.y = a.x * b.y + a.y * b.x;
	return out;
}
// FFT functions for bit-twiddle algorithm
// Function to get indices for bit-reversal
std::vector<int> get_i_to_j(int N) {
	std::vector<int> out(N);
	int logN = static_cast<int>(std::log2(N));
	for (int i = 0; i < N; ++i) {
	    int j = 0;
	    for (int k = 0; k < logN; ++k) {
	        if (i & (1 << k)) {
	            j |= (1 << ((logN - 1) - k));
	        }
	    }
	    out[i] = j;
	}
	return out;
}

// Function to get butterfly indices and omega values
std::tuple<std::vector<int>, std::vector<int>, std::vector<float2>> get_bf_idxs(int N) {
	int logN = static_cast<int>(std::log2(N));
	std::vector<int> kj, kjm;
	std::vector<double2> omega;
	for (int s = 1; s <= logN; s++) {
	    int m = 1 << s;
	    double omega_real = cos(2 * M_PI / m);
	    double omega_imag = sin(2 * M_PI / m);
	    double2 omega_ = { static_cast<double>(omega_real), static_cast<double>(-omega_imag) };
	    for (int k = 0; k < N; k += m) {
	        double2 init_omega = { 1.0f, 0.0f };
	        omega.push_back(init_omega);

	        for (int j = 0; j < m / 2; j++) {
	            kj.push_back(k+j);
	            kjm.push_back(k+j+m/2);
	            
	            if (j < ((m/2)-1)) {
	                omega.push_back(h_c_mult(omega[omega.size() - 1], omega_));
	            }
	        }
	    }
	}
	std::vector<float2> out(omega.size());
	for (unsigned int i = 0; i < out.size(); i++) {
		out[i].x = (float)omega[i].x;
		out[i].y = (float)omega[i].y;
	}
	return std::make_tuple(kj, kjm, out);
}

int main(int argc, char* argv[]) {

	float d_max = 250.0f;
	if (argc < 2) {
		std::cout << "Output file not specified!\n";
		return -1;
	}
	//if (argc > 2)
	//	d_max = ::atof(argv[3]);
		
	
	CSV a = CSV("../data/hurricanes_full_1870.csv", 0);
	//CSV b = CSV("../data/AMO_annual_1870.csv", 1);
	int ndrivers = 31;
	CSV g = CSV("../data/convolved_drivers_31.csv", ndrivers);

	//std::vector<float> amo = b.driver;
	// geographical details
	
	/*
	float lat_start = 30.0;
	float lon_start = 55.0;
	float lat_end = 60.0;
	float lon_end = 75.0;
	float lat_increment = 0.1;
	float lon_increment = 0.1;
	*/
	
	float lat_start = 0.0;
	float lon_start = 0.0;
	float lat_end = 72.0;
	float lon_end = 120.0;
	float lat_increment = 0.3;
	float lon_increment = 0.3;

	int nhits = 50; // minimum number of hits at each point to consider
	
	std::vector<int> a_wind = a.wind;
	std::vector<float> a_lat = a.lat;
	std::vector<float> a_lon = a.lon;
	std::vector<int> a_year = a.year;
	
	std::string out_root = argv[1];
	std::string out_ppm = out_root;// + ".ppm";
	
	int N = a_wind.size();
	
	std::vector<int> year_idxs = get_yearidx(a_year);

	// calculate time coorindates
	float dt = 1; // assume it's 1 for the purposes of this job
	std::vector<float> t;
	int min_year = get_min(a_year);
	int max_year = get_max(a_year);
	t.resize(max_year-min_year+1);
	std::cout << "t.size(): " << t.size() << std::endl;
	for (int i = min_year; i<=max_year; i+=((int)dt)) {
		t[i-min_year] = (float)i;
	}

	// cwt parameters
	// this follows the cwt function from
	// https://github.com/dasyurus/tsaw/blob/master/tsaw/pycwt_fork/wavelet.py
	float f0 = 6.;
	float wv_flambda = (4 * M_PI) / (6 + sqrt(2 + f0 *f0));
	float wv_coi = 1. / sqrt(2);
	float wv_dof = 2.;
	float significance_level = 0.9;
	significance_level = 0.8646 * significance_level/.95; // per Grinsted paper
	//float PPF = chi2.ppf(significance_level, dof); // per py code; from scipy.stats import chi2
	// here we just stick in the result per above numbers
	float PPF = 3.4195635854190196;
	float PPF_over_dof = PPF/wv_dof;
	float T_start = 2 * dt; //nyquist
	float T_end =  dt* wv_flambda*wv_coi * t.size()/2.0;
	float next2 = pow(2, ceil(log(t.size()) / log(2)));
	T_end = next2 / floor(next2 / T_end);
	std::vector<float> freqs_in = geomspace(T_start, T_end, t.size());
	for (int i = 0; i < freqs_in.size(); i++)
		freqs_in[i] = 1 / (wv_flambda * freqs_in[i]);
	std::vector<float> sj;
	sj.resize(freqs_in.size());
	for (int i = 0; i< sj.size(); i++)
		sj[i] = 1 / (wv_flambda * freqs_in[i]);

	// this should be equal to next2, but we recalculate per original fxn
	float padN = pow(2, floor(log(t.size()) / log(2) + 0.4999)+1);
	int log_padN = static_cast<int>(std::log2(padN));
	std::vector<float> ftfreqs = make_ftfreqs(padN,dt);
	std::vector<float> psi_ft_bar(sj.size() * ftfreqs.size());

	// Fill the flattened vector
	for (size_t i = 0; i < sj.size(); ++i) {
		float sj_left = sqrt(sj[i] * ftfreqs[1] * padN); 
			for (size_t j = 0; j < ftfreqs.size(); ++j) {
	    		psi_ft_bar[i * ftfreqs.size() + j] =  sj_left*psi_ft(sj[i] * ftfreqs[j],f0);
			}
	}

	// transfer psi_ft_bar and freqs_in to GPU
	float *d_psi_ft_bar;
	err = hipMalloc(&d_psi_ft_bar, sizeof(float)*psi_ft_bar.size());
		err = hipMemcpy(d_psi_ft_bar,psi_ft_bar.data(), sizeof(float)*psi_ft_bar.size(), hipMemcpyHostToDevice);
	float *d_freqs;
	err = hipMalloc(&d_freqs, sizeof(float)*freqs_in.size());
	err = hipMemcpy(d_freqs, freqs_in.data(), sizeof(float)*freqs_in.size(), hipMemcpyHostToDevice);

	// generate coefficients for FFTs
	auto bf_idxs = get_bf_idxs(padN);
	std::vector<int> kj = std::get<0>(bf_idxs);
	std::vector<int> kjm = std::get<1>(bf_idxs);
	std::vector<float2> omega2 = std::get<2>(bf_idxs);
	int loopSize = static_cast<int>(kj.size()/ log_padN);

	std::vector<int> i_to_j = get_i_to_j(padN);
	// stick fft stuff onto gpu 
	
	float2 *d_omega;
   	int *d_i_to_j, *d_kj, *d_kjm;

	err = hipMalloc(&d_omega, sizeof(float2) * omega2.size());
	err = hipMalloc(&d_i_to_j, sizeof(int) * i_to_j.size());
	err = hipMalloc(&d_kj, sizeof(int) * kj.size());
	err = hipMalloc(&d_kjm, sizeof(int) * kjm.size());

	err = hipMemcpy(d_omega, omega2.data(), sizeof(float2) * omega2.size(), hipMemcpyHostToDevice);
	err = hipMemcpy(d_i_to_j, i_to_j.data(), sizeof(int) * i_to_j.size(), hipMemcpyHostToDevice);
	err = hipMemcpy(d_kj, kj.data(), sizeof(int) * kj.size(), hipMemcpyHostToDevice);
	err = hipMemcpy(d_kjm, kjm.data(), sizeof(int) * kjm.size(), hipMemcpyHostToDevice);

	// transfer time vector to GPU, needed for detrending
	float *d_t;
	    err = hipMalloc(&d_t, t.size()*sizeof(float));
	    err = hipMemcpy(d_t, t.data(), t.size() * sizeof(float), hipMemcpyHostToDevice);

	for (int i = 0; i < N; i++) {
		a_lat[i] = rad(a_lat[i]);
		a_lon[i] = rad(a_lon[i]);
	}

	// calculate geographical details
	
	float lat_size = (lat_end-lat_start) / lat_increment;
	float lon_size = (lon_end-lon_start) / lon_increment;

	int lat_range = round(lat_size);
	int lon_range = round(lon_size);

	std::cout << "lat_size: " << lat_size << std::endl;
	std::cout << "lon_size: " << lon_size << std::endl;

	int lon_batch = 50; // for managing memory use on GPU
	int lon_range_batched = lon_range / lon_batch;

	float *lat, *lon, *ts_map;
	int *yearidx, *wind, *ts_map_d, *ts_map_d3;
	float *wind_out2;
	float *ts_map_1d;
	float2 *ts_map_1d_f;

	// allocate more stuff on GPU
	int gridN = lat_range * lon_range;
	long long wo2 = (long long)gridN*(long long)a_wind.size() * sizeof(float)/lon_batch;
	err = hipMalloc(&lat, a_wind.size()*sizeof(float));
	err = hipMalloc(&lon, a_wind.size()*sizeof(float));
	err = hipMalloc(&wind, a_wind.size()*sizeof(int));
	err = hipMalloc(&wind_out2, wo2);
	err = hipMalloc(&yearidx, (year_idxs.size()) * sizeof(int));
	err = hipMalloc(&ts_map, gridN * t.size() * sizeof(float));
	err = hipMalloc(&ts_map_d, gridN * t.size() * sizeof(int));
	err = hipMalloc(&ts_map_d3, gridN * sizeof(int));
	
	err = hipMemcpy(lat, a_lat.data(), a_lat.size() * sizeof(float), hipMemcpyHostToDevice);
	err = hipMemcpy(lon, a_lon.data(), a_lat.size() * sizeof(float), hipMemcpyHostToDevice);
	err = hipMemcpy(wind, a_wind.data(), a_lat.size() * sizeof(int), hipMemcpyHostToDevice);
	err = hipMemcpy(yearidx, year_idxs.data(), year_idxs.size() * sizeof(int), hipMemcpyHostToDevice);
	
	
	fillBufferV(ts_map, 0, gridN * t.size());
	fillBufferV(ts_map_d, 0, gridN * t.size());
	fillBufferV(ts_map_d3, 0, gridN);
  
	//output vector
	std::vector<float> vhitmap;
	vhitmap.resize(gridN);
		
	// generate array of time series for each coordinate
	dim3 blockSize2(32,1,1);
	dim3 numBlocks2(get_work_size(a_wind.size(), blockSize2.x), get_work_size(lat_range,blockSize2.y), lon_range_batched);
	dim3 blockSize3(32,1,1);
	dim3 numBlocks3(get_work_size(lat_range,blockSize3.x), get_work_size(lon_range_batched,blockSize3.y), get_work_size(t.size(), blockSize3.z));

	for (int aa = 0; aa<lon_batch; aa++) {
		scale_wind2<<<numBlocks2,blockSize2>>>(lat, lon, aa, wind, wind_out2, N, d_max, lat_increment, lon_increment, lat_range, lon_range_batched, lat_start, lon_start);
		make_ts2<<<numBlocks3,blockSize3>>>(aa,yearidx, t.size(),N,ts_map,wind_out2,ts_map_d,lat_range,lon_range,lon_range_batched);
	}
	
	// reduce hits for each coordinate
	dim3 numBlocks33(get_work_size(gridN,blockSize3.x), 1, 1);
	reduce_ts_map_d<<<numBlocks33,blockSize3>>>(ts_map_d,ts_map_d3,t.size(),gridN);

	// free no longer needed buffers
	err = hipFree(&wind_out2); 
	err = hipFree(&lat);			       
	err = hipFree(&lon);			       
	err = hipFree(&wind);			       

	// recast to a 1-d array, and remove coordinates with hits < nhits
	std::vector<int> v_perlatlon;
	v_perlatlon.resize(gridN);
	err = hipMemcpy(v_perlatlon.data(), ts_map_d3, lat_range* lon_range * sizeof(int), hipMemcpyDeviceToHost);

	int ts_map_1d_size = 0;
	std::vector<int> v_idx_conversion;
	for (unsigned int i = 0; i < v_perlatlon.size(); i++) {
		if (v_perlatlon[i]>=nhits) {
			v_idx_conversion.push_back(i); // crude but shouldn't be too slow
			ts_map_1d_size++;
		}
	}

	err = hipFree(ts_map_d);
	err = hipFree(ts_map_d3);

	// guess eventual maximum size needed to stored at once in GPU memory
	// based on current parameters
	float rough_total_size = (ts_map_1d_size*padN*t.size()*sizeof(float2)+(ts_map_1d_size)*t.size()*t.size()*sizeof(float))/(1024*1024*1024);
	rough_total_size *= 1.15; // adjust for stack weirdness on device
	std::cout<< "Rough total size: " << rough_total_size << " GB\n";

	std::cout << "Initial # of coordinates: " << v_perlatlon.size() << std::endl;
	std::cout << "Sparse # of coordinates: " << ts_map_1d_size << std::endl;

	// allocate for 1d sparse arrays, now that we know their size
	err = hipMalloc(&ts_map_1d, t.size()*ts_map_1d_size*sizeof(float));
	err = hipMalloc(&ts_map_1d_f, padN*ts_map_1d_size*sizeof(float2));
	int * idx_conversion;
	err = hipMalloc(&idx_conversion, v_idx_conversion.size()*sizeof(int));
	err = hipMemcpy(idx_conversion, v_idx_conversion.data(), v_idx_conversion.size() * sizeof(int), hipMemcpyHostToDevice);
	long long size_W = ts_map_1d_size*padN*t.size();
	float2 *d_W;
	err  = hipMalloc(&d_W,size_W*sizeof(float2));

	// recast into 1-d sparse array
	dim3 numBlocks4(get_work_size(ts_map_1d_size,blockSize3.x), get_work_size(t.size(),blockSize3.y), 1);
	recast_ts_map<<<numBlocks4,blockSize3>>>(ts_map_1d,ts_map_1d_size,t.size(),idx_conversion,ts_map);
	
	// start cwt procedure
	dim3 numBlocks5(get_work_size(ts_map_1d_size,blockSize3.x), 1, 1);
	detrend<<<numBlocks5,blockSize3>>>(ts_map_1d,d_t,ts_map_1d_size,t.size());

	// take fft of time series at each coordinate
	fft_kernel2<<<numBlocks5, blockSize3>>>(ts_map_1d, t.size(),  padN, log_padN,ts_map_1d_size, d_omega, d_i_to_j, d_kj, d_kjm, loopSize, ts_map_1d_f);

	//std::cout << "Size of ts_maps: " << ts_map_1d_size*(padN+t.size())*sizeof(float2)/(1024*1024*1024) << " GB\n";
	//std::cout << "Size of W: " << ts_map_1d_size*padN*t.size()*sizeof(float2)/(1024*1024*1024) << " GB\n";
	dim3 blockSize6(32,1,1);
	dim3 numBlocks6(get_work_size(ts_map_1d_size,blockSize6.x),get_work_size(t.size(),blockSize6.y),get_work_size(padN,blockSize6.z));
	make_W<<<numBlocks6,blockSize6>>>(d_W,ts_map_1d_f,d_psi_ft_bar,ts_map_1d_size,t.size(),padN);

	// perform ifft of W
	// conj -> fft -> conj -> scale
	dim3 blockSize7(32,1,1);
	dim3 numBlocks7(get_work_size_lld(size_W,blockSize7.x),1,1);
	dim3 blockSize8(32,1,1);
	dim3 numBlocks8(get_work_size(ts_map_1d_size,blockSize8.x),get_work_size(t.size(),blockSize8.y),1);

	conj<<<numBlocks7,blockSize7>>>(d_W,size_W);
	fft_kernel2d<<<numBlocks8, blockSize8>>>(d_W, t.size(),  padN, log_padN,ts_map_1d_size, d_omega, d_i_to_j, d_kj, d_kjm, loopSize);
	conj<<<numBlocks7,blockSize7>>>(d_W,size_W);
	scale<<<numBlocks7,blockSize7>>>(d_W,size_W,padN);

	// calculate some autoregression stuff
	float * d_alpha;
	err = hipMalloc(&d_alpha,ts_map_1d_size*sizeof(float));
	float * d_Pk1;
	err = hipMalloc(&d_Pk1,ts_map_1d_size*t.size()*sizeof(float));
	dim3 blockSize9(32,1,1);
	dim3 numBlocks9(get_work_size(ts_map_1d_size,blockSize9.x),1,1);
	make_alpha<<<numBlocks9,blockSize9>>>(d_alpha,ts_map_1d,ts_map_1d_size,t.size());
	dim3 numBlocks91(get_work_size(ts_map_1d_size,blockSize9.x),get_work_size(t.size(),blockSize9.y),1);
	make_Pk<<<numBlocks91,blockSize9>>>(d_Pk1,d_freqs,d_alpha,ts_map_1d_size,t.size(),dt);
		   
	// end cwt procedure
	// start driver and xwt procedures
	float * driver;
	float2 * driver_f;
	std::vector<float> v_drivers = g.driver;

	err = hipMalloc(&driver, v_drivers.size()*sizeof(float));
	err = hipMemcpy(driver, v_drivers.data(), v_drivers.size() * sizeof(float), hipMemcpyHostToDevice);
	err = hipMalloc(&driver_f, v_drivers.size()*sizeof(float2));

	float2 *d_Wx;
	long long size_Wx = ndrivers*padN*t.size();
	err  = hipMalloc(&d_Wx,size_Wx*sizeof(float2));
	//std::cout << "Size of Wx: " << ndrivers*padN*t.size()*sizeof(float2)/(1024*1024*1024) << " GB\n";
	float * d_alphax;
	err = hipMalloc(&d_alphax,ndrivers*sizeof(float));
	float *d_xsig95; // holds abs(xcoefs), and then processed to xsig95
	err  = hipMalloc(&d_xsig95,ts_map_1d_size*t.size()*t.size()*sizeof(float)); // trim padding here
	float * d_Pk2;
	err = hipMalloc(&d_Pk2,ndrivers*t.size()*sizeof(float));
	//std::cout << "Size of xsig95: " << (ts_map_1d_size)*t.size()*t.size()*sizeof(float)/(1024*1024*1024) << " GB\n";
	float *d_xwtred;
	err = hipMalloc(&d_xwtred,ts_map_1d_size*sizeof(float));

	// calculate cwt of drivers, as part of the xwt procedure
	dim3 numBlocks10(get_work_size(ndrivers,blockSize3.x), 1, 1);
	detrend<<<numBlocks10,blockSize3>>>(driver,d_t,ndrivers,t.size());
	
	fft_kernel2<<<numBlocks10, blockSize3>>>(driver, t.size(),  padN, log_padN,ndrivers, d_omega, d_i_to_j, d_kj, d_kjm, loopSize, driver_f);

	dim3 blockSize11(32,1,1);
	dim3 numBlocks11(get_work_size(ndrivers,blockSize6.x),get_work_size(t.size(),blockSize6.y),get_work_size(padN,blockSize6.z));
	make_W<<<numBlocks11,blockSize11>>>(d_Wx,driver_f,d_psi_ft_bar,ndrivers,t.size(),padN);

	dim3 blockSize12(32,1,1);
	dim3 numBlocks12(get_work_size_lld(size_Wx,blockSize7.x),1,1);
	dim3 blockSize13(32,1,1);
	dim3 numBlocks13(get_work_size(ndrivers,blockSize8.x),get_work_size(t.size(),blockSize8.y),1);
	conj<<<numBlocks12,blockSize12>>>(d_Wx,size_Wx);
	fft_kernel2d<<<numBlocks13, blockSize13>>>(d_Wx, t.size(),  padN, log_padN,ndrivers, d_omega, d_i_to_j, d_kj, d_kjm, loopSize);
	conj<<<numBlocks12,blockSize12>>>(d_Wx,size_Wx);
	scale<<<numBlocks12,blockSize12>>>(d_Wx,size_Wx,padN);

	dim3 blockSize14(32,1,1);
	dim3 numBlocks14(get_work_size(ndrivers,blockSize14.x),1,1);
	make_alpha<<<numBlocks14,blockSize14>>>(d_alphax,driver,ndrivers,t.size());
	dim3 numBlocks141(get_work_size(ndrivers,blockSize14.x),get_work_size(t.size(),blockSize14.y),1);
	make_Pk<<<numBlocks141,blockSize14>>>(d_Pk2,d_freqs,d_alphax,ndrivers,t.size(),dt);

	// calculate xwt, and reduce to # of siginficant (time,period) points
	dim3 blockSize15(32,1,1);
	dim3 numBlocks15(get_work_size(ts_map_1d_size,blockSize15.x),get_work_size(t.size(),blockSize15.y),get_work_size(t.size(),blockSize15.z));
	dim3 blockSize16(32,1,1);
	dim3 numBlocks16(get_work_size(ts_map_1d_size,blockSize16.x),1);

	std::vector<float> _vhitmap;
	_vhitmap.resize(ts_map_1d_size);
	// work one driver at a time (better on memory)
	for (int dd=0; dd<ndrivers; dd++) {
		// do xwt calculation and reduction
		make_xsig95<<<numBlocks15,blockSize15>>>(d_xsig95, d_W, d_Wx, t.size(), padN, ts_map_1d_size, dd, d_Pk1, d_Pk2, PPF_over_dof);
		reduce_xsig95<<<numBlocks16,blockSize16>>>(d_xwtred, d_xsig95, ts_map_1d_size, t.size());
	
		// retrieve results and save
		err = hipMemcpy(_vhitmap.data(), d_xwtred, ts_map_1d_size * sizeof(float), hipMemcpyDeviceToHost);

		int _i =0;
		float sum_sig = 0.0;
		float max_sig = 0.0;
		int max_lat, max_lon, max_idx, max_idx_1d;
		for (unsigned int i = 0; i < v_perlatlon.size(); i++) {
			if (v_perlatlon[i]>=nhits) {
				vhitmap[i] = _vhitmap[_i] / (t.size()*t.size());
				sum_sig += _vhitmap[_i];
				if (_vhitmap[_i]>max_sig) {
					max_sig = _vhitmap[_i];
					max_idx = i;
					max_idx_1d = _i;
					max_lat = i / lon_range;
					max_lon = i - max_lat * lon_range;

				}

				_i++;

			}

		}
		int ddd = dd;
		std::cout << "Driver " << ddd << ": " << sum_sig/1e6 << std::endl;
		std::cout << "Driver " << ddd << ": " << max_sig << std::endl;
		std::cout << "\t" << "idx: " << max_idx << ", idx_1d: " << max_idx_1d << std::endl;
		std::cout << "\t" << "lat: " << max_lat * lat_increment + lat_start
			            << ", lon: " << max_lon * lon_increment + lon_start << std::endl;

		int idx;
		std::ofstream myfile2;
		out_ppm = out_root+ "_" +std::to_string(dd) + ".ppm";
		myfile2.open(out_ppm.c_str());
		//header for PPM
		
		myfile2 << "P3" << std::endl;
		myfile2 << lon_range << " " << lat_range << std::endl;
		myfile2 << "255" << std::endl;
		
		// fill in stuff
		for (unsigned aa  = 0; aa < lat_range; aa++) {
			for (unsigned bb = 0; bb< lon_range; bb++) {
				int aa_ = (lat_range - 1) - aa;
				int bb_ = (lon_range - 1) - bb;
				idx = lon_range * aa_ + bb_;
				float q = vhitmap[idx]; // write to easy-to-copy var

				
				if (q < 0.02)
					myfile2 << "11 20 37" << std::endl;
				else if (q >= 0.02 && q<0.04)
					myfile2 << "19 56 89" << std::endl;
				else if (q >= 0.04 && q<0.06)
					myfile2 << "72 88 122" << std::endl;
				else if (q >= 0.06 && q<0.08)
					myfile2 << "108 94 116" << std::endl;
				else if (q >= 0.08 && q<0.10)
					myfile2 << "144 96 108" << std::endl;
				else if (q >= 0.10 && q< 0.12)
					myfile2 << "188 100 97" << std::endl;
				else if (q >=  0.12 && q< 0.14)
					myfile2 << "230 121 96" << std::endl;
				else if (q >=  0.14 && q< 0.16)
					myfile2 << "233 162 120" << std::endl;
				else if (q >=  0.16 && q< 0.18)
					myfile2 << "232 199 158" << std::endl;
				else if (q >=  0.18)
					myfile2 << "254 245 219" << std::endl;
					/*
				if (q < 0.03)
					myfile2 << "11 20 37" << std::endl;
				else if (q >= 0.03 && q<0.06)
					myfile2 << "19 56 89" << std::endl;
				else if (q >= 0.06 && q<0.09)
					myfile2 << "72 88 122" << std::endl;
				else if (q >= 0.09 && q<0.12)
					myfile2 << "108 94 116" << std::endl;
				else if (q >= 0.12 && q<0.15)
					myfile2 << "144 96 108" << std::endl;
				else if (q >= 0.15 && q< 0.18)
					myfile2 << "188 100 97" << std::endl;
				else if (q >=  0.18 && q< 0.21)
					myfile2 << "230 121 96" << std::endl;
				else if (q >=  0.21 && q< 0.24)
					myfile2 << "233 162 120" << std::endl;
				else if (q >=  0.24 && q< 0.27)
					myfile2 << "232 199 158" << std::endl;
				else if (q >=  0.27)
					myfile2 << "254 245 219" << std::endl;
					 */
				//else if (q >= 0.2)
				//	myfile2 << "165 0 26" << std::endl;
			}
		}
		myfile2.close();
	}
	return 0;

}
