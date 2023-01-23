#include "Func.h"

/////////////////////////////////////////////////////////////////////////
// 1. 함수는 Colab 환경에서 동작해야 합니다.
// 2. 자유롭게 구현하셔도 되지만 모든 함수에서 GPU를 활용해야 합니다.
// 3. CPU_Func.cu에 있는 Image_Check함수에서 True가 Return되어야 하며, CPU코드에 비해 속도가 빨라야 합니다.
/////////////////////////////////////////////////////////////////////////

__constant__ float filter[25] =
{
	0.0029165,	0.0130709,	0.0215502,	0.0130709,	0.0029165,
	0.0130709,	0.0585795,	0.0965813,	0.0585795,	0.0130709,
	0.0215502,	0.0965813,	0.159236,	0.0965813,	0.0215502,
	0.0130709,	0.0585795,	0.0965813,	0.0585795,	0.0130709,
	0.0029165,	0.0130709,	0.0215502,	0.0130709,	0.0029165
};
//filter is in constant memory

__constant__ int filter_x[9] = { -1,0,1,-2,0,2,-1,0,1 };
__constant__ int filter_y[9] = { 1,2,1,0,0,0,-1,-2,-1 };
//filter_x and filter_y in constant memory

__host__ void GPU_Grayscale(uint8_t* buf, uint8_t* gray, uint8_t start_add, int len);
__host__ void GPU_Noise_Reduction(int width, int height, uint8_t* gray, uint8_t* gaussian);
__host__ void GPU_Intensity_Gradient(int width, int height, uint8_t* gaussian, uint8_t* sobel, uint8_t* angle);
__host__ void GPU_Non_maximum_Suppression(int width, int height, uint8_t* angle, uint8_t* sobel, uint8_t* suppression_pixel, uint8_t& min, uint8_t& max);
__host__ void GPU_Hysteresis_Thresholding(int width, int height, uint8_t* suppression_pixel, uint8_t* hysteresis, uint8_t min, uint8_t max);

__global__ void buf_to_gray_global(const uint8_t* buf, uint8_t* gray) //buf in global memory, gray in global memory 
{
	const unsigned int i = blockDim.x * blockIdx.x + threadIdx.x; //get index (index is 0 ~ 2400981)
	int tmp;
	//i, tmp, tmp2 in register

	if (i > 2400053) //remain index is 0, 1 ... 2400052, 2400053
		return;

	else if (i >= 0 && i < 54) //exception for index which is lower than start_add (part of header)
	{
		gray[i] = buf[i];

		return;
	}

	else //else index : not header (body) (remain index is 54, 55 ... 2400052, 2400053
	{
		if (i % 3 == 0) //i is 54, 57, ... 2400051
		{
			tmp = (buf[i] * 0.114 + buf[i + 1] * 0.587 + buf[i + 2] * 0.299); //3 global memory read
			gray[i] = tmp; //1 global memory write

			return;

		}

		else if (i % 3 == 1) //i is 55, 58, ... 2400052
		{
			tmp = (buf[i - 1] * 0.114 + buf[i] * 0.587 + buf[i + 1] * 0.299); //3 global memory read
			gray[i] = tmp; //1 global memory write

			return;
		}

		else //(i % 3 == 2) i is 56, 59, ... 2400053
		{
			tmp = (buf[i - 2] * 0.114 + buf[i - 1] * 0.587 + buf[i] * 0.299); //3 global memory read
			gray[i] = tmp; //1 global memory write

			return;
		}
	}
}
//(4 global memory access) per thread
//for GPU_Grayscale

__global__ void buf_to_gray_shared(const uint8_t* buf, uint8_t* gray) //buf in global memory, gray in global memory 
{
	const unsigned int i = blockDim.x * blockIdx.x + threadIdx.x; //get index (index is 0 ~ 2400981)
	const unsigned int tx = threadIdx.x; //get thread index (unique inside block) (thread index is 0 ~ 1023)
	int tmp;
	int tmp2;
	//i, tx, tmp, tmp2 in register

	__shared__ uint8_t shared_buf[1023];
	//there are 1023 threads in 1 block (length of shared_buf is 1023)
	//and there are 2347 blocks in 1 grid (there are 2347 shared_buf in 1 grid)
	//shared_buf in shared memory

	if (i > 2400053) //remain index is 0, 1 ... 2400052, 2400053
		return;

	tmp2 = shared_buf[tx] = buf[i]; //copy single element global memory -> shared memory (each thread)
	//1 global memory read, 1 shared memory write,
	__syncthreads();
	
	if (i >= 0 && i < 54) //another exception for index which is part of header
	{
		gray[i] = tmp2;

		return;
	}

	else //else index : not header (body) (remain index is 54, 55 ... 2400052, 2400053)
	{
		if (i % 3 == 0) //i is 54, 57, ... 2400051
		{
			tmp = (tmp2 * 0.114 + shared_buf[tx + 1] * 0.587 + shared_buf[tx + 2] * 0.299); //2 shared memory read
			gray[i] = tmp; //1 global memory write

			return;
		}
		 
		else if (i % 3 == 1) //i is 55, 58, ... 2400052
		{
			tmp = (shared_buf[tx - 1] * 0.114 + tmp2 * 0.587 + shared_buf[tx + 1] * 0.299); //2 shared memory read
			gray[i] = tmp; //1 global memory write

			return;
		}

		else //(i % 3 == 2) i is 56, 59, ... 2400053
		{
			tmp = (shared_buf[tx - 2] * 0.114 + shared_buf[tx - 1] * 0.587 + tmp2 * 0.299); //2 shared memory read
			gray[i] = tmp; //1 global memory write

			return;
		}
	}
}
//(2(1 + 1) global memory access + 3(1 + 2) shared memory access + 1 syncthreads) per thread
//for GPU_Grayscale

__host__ void GPU_Grayscale(uint8_t* buf, uint8_t* gray, uint8_t start_add, int len)
{
	//cout << _msize(buf) << endl;
	//cout << _msize(gray) << endl;
	//cout << (int)start_add << endl;
	//cout << len << endl;

	dim3 GridDim(2347, 1, 1); //2344 blocks in 1 grid
	dim3 BlockDim(1023, 1, 1); //1024 threads in 1 block
	//24000981 threads in 1 grid

	uint8_t* GPU_buf = NULL;
	uint8_t* GPU_gray = NULL;
	cudaMalloc((void**)&GPU_buf, 2400054 * sizeof(uint8_t)); //GPU_buf = buf for device
	cudaMalloc((void**)&GPU_gray, 2400054 * sizeof(uint8_t)); //GPU_gray = gray for device
	cudaMemcpy(GPU_buf, buf, 2400054 * sizeof(uint8_t), cudaMemcpyHostToDevice);
	//memory set(allocate + copy) for device

	//buf_to_gray_global << <GridDim, BlockDim >> > (GPU_buf, GPU_gray);
	buf_to_gray_shared << <GridDim, BlockDim >> > (GPU_buf, GPU_gray);
	//!!!

	cudaMemcpy(gray, GPU_gray, 2400054 * sizeof(uint8_t), cudaMemcpyDeviceToHost);
	cudaFree(GPU_buf);
	cudaFree(GPU_gray);
	//memory clean(copy + free) for device

	return;
}

__global__ void gray_to_gaussian_global(const uint8_t* gray, uint8_t* gaussian) //buf in global memory, gray in global memory 
{
	const unsigned int i = blockDim.x * blockIdx.x + threadIdx.x; //get index (index is 0 ~ 2399999)
	const int width = 1000, height = 800;
	int row, col;
	int wanted_row, wanted_col;
	float v = 0;
	int k, l;
	//i, sigma, width, height, row, col, v, k, l in register

	if (i > 2399999) //terminate dummy thread (remain index is 0 ~ 2399999)
		return;

	else //index is 0 ~ 2399999
	{	
		//there is no zero padding section code

		row = (i / 3) / width; //row is 0, 1 ... 798, 799
		col = (i / 3) % width; //col is 0, 1 ... 998, 999
		for (k = -2; k <= 2; k++)
		{
			for (l = -2; l <= 2; l++)
			{
				wanted_row = row + k; //wanted_row is -2, -1 ... 800, 801
				wanted_col = col + l; //wanted_col is -2, -1 ... 1000, 1001

				if (wanted_row < 0 || wanted_row > height - 1)
					continue; //v += 0 (zero padding)

				else if (wanted_col < 0 || wanted_col > width - 1)
					continue; //v += 0 (zero padding)

				else //remain index_row is 0, 1 ... 798, 799 and remain index_col is 0, 1 ... 998, 999
					v += gray[(wanted_row * width + wanted_col) * 3] * filter[(k + 2) * 5 + (l + 2)]; //25 global memory read, 25 constant memory read
			}
		}
		//algorithm of conv2d_5x5
		//Gaussian blur section

		gaussian[i] = v; //current component(R or G or B) is v
		//1 global memory write

		return;
	}
}
//(26 global memory access) per thread
//for GPU_Noise_Reduction

__global__ void gray_to_gaussian_shared(const uint8_t* gray, uint8_t* gaussian) //buf in global memory, gray in global memory 
{
	const unsigned int i = blockDim.x * blockIdx.x + threadIdx.x; //get index (i is 0 ~ 799999)
	const unsigned int tx = threadIdx.x; //get thread index (unique inside block) (thread index is 0 ~ 999)
	const unsigned int index = i * 3; //index for accessing gray or gaussian (index is 0 ~ 2399997)

	const int width = 1000, height = 800;
	int row, col;
	int wanted_row, wanted_col;
	float v = 0;
	int k, l;
	//i, sigma, width, height, row, col, v, k, l in register

	__shared__ uint8_t shared_gray[5000];
	//there are 1000 threads in 1 block (length of shared_gray is 5000)
	//and there are 800 blocks in 1 grid (there are 800 shared_gray in 1 grid)
	//shared_gray in shared memory

	if (index > 2399999) //terminate dummy thread (remain index is 0 ~ 2399999)
		return;

	if (index >= 6000) //when row is 2, 3 ... 7998, 7999
		shared_gray[tx] = gray[index - 6000]; //copy single element global memory -> shared memory (each thread)
	if (index >= 3000) //when row is 1, 2 ... 7998, 7999
		shared_gray[tx + 1000] = gray[index - 3000]; //copy single element global memory -> shared memory (each thread)
	shared_gray[tx + 2000] = gray[index]; //copy single element global memory -> shared memory (each thread)
	if (index < 2397000) //when row is 0, 1 ... 7996, 7997
		shared_gray[tx + 3000] = gray[index + 3000]; //copy single element global memory -> shared memory (each thread)
	if (index < 2394000) //when row is 0, 1 ... 7997, 7998
		shared_gray[tx + 4000] = gray[index + 6000]; //copy single element global memory -> shared memory (each thread)
	//5 global memory read, 5 shared memory write
	__syncthreads();
	
	if(1) //index is 0 ~ 2399999
	{
		//there is no zero padding section code

		row = (index / 3) / width; //row is 0, 1 ... 798, 799
		col = (index / 3) % width; //col is 0, 1 ... 998, 999
		for (k = -2; k <= 2; k++)
		{
			for (l = -2; l <= 2; l++)
			{
				wanted_row = row + k; //wanted_row is -2, -1 ... 800, 801
				wanted_col = col + l; //wanted_col is -2, -1 ... 1000, 1001

				if (wanted_row < 0 || wanted_row > height - 1)
					continue; //v += 0 (zero padding)

				else if (wanted_col < 0 || wanted_col > width - 1)
					continue; //v += 0 (zero padding)

				wanted_row = wanted_row - row; //wanted_row is -2, -1, 0, 1, 2 (delta_row)
				wanted_col = wanted_col - col; //wanted_col is -2, -1, 0, 1, 2 (delta_col)
				v += shared_gray[tx + 2000 + wanted_row * 1000 + wanted_col] * filter[(k + 2) * 5 + (l + 2)]; //25 shared memory read, 25 constant memory read
			}
		}
		//algorithm of conv2d_5x5
		//Gaussian blur section

		gaussian[index] = v;
		gaussian[index + 1] = v;
		gaussian[index + 2] = v;
		//current component(R or G or B) is v
		//3 global memory write

		return;
	}
}
//(8(5 + 3) global memory access + 30(5 + 25) shared memory access + 1 syncthreads) per thread
//for GPU_Noise_Reduction

__host__ void GPU_Noise_Reduction(int width, int height, uint8_t* gray, uint8_t* gaussian)
{
	//cout << width << endl;
	//cout << height << endl;
	//cout << _msize(gray - 54) << endl;
	//cout << _msize(gaussian - 54) << endl;

	dim3 GridDim(800, 1, 1); //800 blocks in 1 grid
	dim3 BlockDim(1000, 1, 1); //1000 threads in 1 block
	//800000 threads in 1 grid

	uint8_t* GPU_gray = NULL;
	uint8_t* GPU_gaussian = NULL;
	cudaMalloc((void**)&GPU_gray, 2400000 * sizeof(uint8_t)); //GPU_gray = gray for device
	cudaMalloc((void**)&GPU_gaussian, 2400000 * sizeof(uint8_t)); //GPU_gaussian = gaussian for device
	cudaMemcpy(GPU_gray, gray, 2400000 * sizeof(uint8_t), cudaMemcpyHostToDevice);
	//memory set(allocate + copy) for device

	//gray_to_gaussian_global << <GridDim, BlockDim >> > (GPU_gray, GPU_gaussian);
	gray_to_gaussian_shared << <GridDim, BlockDim >> > (GPU_gray, GPU_gaussian);
	//!!!

	cudaMemcpy(gaussian, GPU_gaussian, 2400000 * sizeof(uint8_t), cudaMemcpyDeviceToHost);

	cudaFree(GPU_gray);
	cudaFree(GPU_gaussian);
	//memory clean(copy + free) for device

	return;
}

__global__ void gaussian_to_sobel_and_angle_global(const uint8_t* gaussian, uint8_t* sobel, uint8_t* angle) //gaussian in global memory, sobel in global memory, angle in global memory
{
	const unsigned int i = blockDim.x * blockIdx.x + threadIdx.x; //get index (index is 0 ~ 2399999)
	const int width = 1000, height = 800;
	int row, col;
	int wanted_row, wanted_col;
	int gy = 0, gx = 0;
	int k, l;
	int t;
	uint8_t v;
	float t_angle;
	int tmp;
	//i, sigma, width, height, row, col, gy, gx, k, l, t, v, t_angle, temp in register

	if (i > 2399999) //terminate dummy thread (remain index is 0 ~ 2399999)
		return;

	else //index is 0 ~ 2399999
	{
		//there is no zero padding section code

		row = (i / 3) / width; //row is 0, 1 ... 798, 799
		col = (i / 3) % width; //col is 0, 1 ... 998, 999
		for (k = -1; k <= 1; k++)
		{
			for (l = -1; l <= 1; l++)
			{
				wanted_row = row + k; //wanted_row is -1, 0 ... 799, 800
				wanted_col = col + l; //wanted_col is -1, 0 ... 999, 1000

				if (wanted_row < 0 || wanted_row > height - 1)
					continue; //gy += 0, gx += 0 (zero padding)

				else if (wanted_col < 0 || wanted_col > width - 1)
					continue; //gy += 0, gx += 0 (zero padding)

				else //remain index_row is 2, 3 ... 798, 799 and remain index_col is 2, 3 ... 998, 999
				{
					tmp = gaussian[(wanted_row * width + wanted_col) * 3];
					gy += tmp * filter_y[(k + 1) * 3 + l + 1];
					gx += tmp * filter_x[(k + 1) * 3 + l + 1];
				}
				//9 global memory read, 9 constant memory read
			}
		}
		//algorithm of conv2d_3x3

		t = sqrt((double)(gx * gx + gy * gy));
		//t = 128;
		if (t > 255)
			v = 255;
		else
			v = t;
		sobel[i] = v; //1 globla memory write
		//now sobel is written

		if (gy != 0 || gx != 0)
			t_angle = (float)atan2((double)gy, (double)gx) * 180.0 / 3.14;
		//t_angle = 0;
		if ((t_angle > -22.5 && t_angle <= 22.5) || (t_angle > 157.5 || t_angle <= -157.5))
			angle[i / 3] = 0;
		else if ((t_angle > 22.5 && t_angle <= 67.5) || (t_angle > -157.5 && t_angle <= -112.5))
			angle[i / 3] = 45;
		else if ((t_angle > 67.5 && t_angle <= 112.5) || (t_angle > -112.5 && t_angle <= -67.5))
			angle[i / 3] = 90;
		else if ((t_angle > 112.5 && t_angle <= 157.5) || (t_angle > -67.5 && t_angle <= -22.5))
			angle[i / 3] = 135;
		//1 global memory write
		//now angle is written

		return;
	}
}
//(11 global memory access) per thread
//for GPU_Noise_Reduction

__global__ void gaussian_to_sobel_and_angle_shared(const uint8_t* gaussian, uint8_t* sobel, uint8_t* angle) //gaussian in global memory, sobel in global memory, angle in global memory
{
	const unsigned int i = blockDim.x * blockIdx.x + threadIdx.x; //get index (i is 0 ~ 799999)
	const unsigned int tx = threadIdx.x; //get thread index (unique inside block) (thread index is 0 ~ 999)
	const unsigned int index = i * 3; //index for accessing gray or gaussian (index is 0 ~ 2399997)

	const int width = 1000, height = 800;
	int row, col;
	int wanted_row, wanted_col;
	int gy = 0, gx = 0;
	int k, l;
	int t;
	uint8_t v;
	float t_angle;
	int tmp;
	//i, sigma, width, height, row, col, gy, gx, k, l, t, v, t_angle, temp in register

	__shared__ uint8_t shared_gaussian[3000];
	//there are 1000 threads in 1 block (length of shared_gaussian is 3000)
	//and there are 800 blocks in 1 grid (there are 800 shared_gaussian in 1 grid)
	//shared_gaussian in shared memory

	if (index > 2399999) //terminate dummy thread (remain index is 0 ~ 2399999)
		return;

	if (index >= 3000) //when row is 1, 2 ... 7998, 7999
		shared_gaussian[tx] = gaussian[index - 3000]; //copy single element global memory -> shared memory (each thread)
	shared_gaussian[tx + 1000] = gaussian[index]; //copy single element global memory -> shared memory (each thread)
	if (index < 2397000) //when row is 0, 1 ... 7997, 7998
		shared_gaussian[tx + 2000] = gaussian[index + 3000]; //copy single element global memory -> shared memory (each thread)
	//3 global memory read, 3 shared memory write
	__syncthreads();

	if(1) //index is 0 ~ 2399999
	{
		//there is no zero padding section code

		row = (index / 3) / width; //row is 0, 1 ... 798, 799
		col = (index / 3) % width; //col is 0, 1 ... 998, 999
		for (k = -1; k <= 1; k++)
		{
			for (l = -1; l <= 1; l++)
			{
				wanted_row = row + k; //wanted_row is -1, 0 ... 799, 800
				wanted_col = col + l; //wanted_col is -1, 0 ... 999, 1000

				if (wanted_row < 0 || wanted_row > height - 1)
					continue; //gy += 0, gx += 0 (zero padding)

				else if (wanted_col < 0 || wanted_col > width - 1)
					continue; //gy += 0, gx += 0 (zero padding)

				else //remain index_row is 2, 3 ... 798, 799 and remain index_col is 2, 3 ... 998, 999
				{
					wanted_row = wanted_row - row; //wanted_row is -1, 0, 1 (delta_row)
					wanted_col = wanted_col - col; //wanted_col is -1, 0, 1 (delta_col)
					tmp = shared_gaussian[tx + 1000 + wanted_row * 1000 + wanted_col];
					gy += tmp * filter_y[(k + 1) * 3 + l + 1];
					gx += tmp * filter_x[(k + 1) * 3 + l + 1];
					//9 shared memory read, 9 constant memory read
				}
				//9 global memory read, 9 constant memory read
			}
		}
		//algorithm of conv2d_3x3

		t = sqrt((double)(gx * gx + gy * gy));
		//t = 128;
		if (t > 255)
			v = 255;
		else
			v = t;
		sobel[index] = v;
		sobel[index + 1] = v;
		sobel[index + 2] = v;
		//3 globla memory write
		//now sobel is written

		if (gy != 0 || gx != 0)
			t_angle = (float)atan2((double)gy, (double)gx) * 180.0 / 3.14;
		//t_angle = 0;
		if ((t_angle > -22.5 && t_angle <= 22.5) || (t_angle > 157.5 || t_angle <= -157.5))
			angle[index / 3] = 0;
		else if ((t_angle > 22.5 && t_angle <= 67.5) || (t_angle > -157.5 && t_angle <= -112.5))
			angle[index / 3] = 45;
		else if ((t_angle > 67.5 && t_angle <= 112.5) || (t_angle > -112.5 && t_angle <= -67.5))
			angle[index / 3] = 90;
		else if ((t_angle > 112.5 && t_angle <= 157.5) || (t_angle > -67.5 && t_angle <= -22.5))
			angle[index / 3] = 135;
		//1 global memory write
		//now angle is written

		return;
	}
}
//(7(3 + 3 + 1) global memory access + 12(3 + 9) shared memory access + 1 syncthreads) per thread
//for GPU_Noise_Reduction

__host__ void GPU_Intensity_Gradient(int width, int height, uint8_t* gaussian, uint8_t* sobel, uint8_t* angle)
{
	//cout << width << endl;
	//cout << height << endl;
	//cout << _msize(gaussian - 54) << endl;
	//cout << _msize(sobel - 54) << endl;
	//cout << _msize(angle) << endl;
	
	dim3 GridDim(800, 1, 1); //800 blocks in 1 grid
	dim3 BlockDim(1000, 1, 1); //1000 threads in 1 block
	//800000 threads in 1 grid

	uint8_t* GPU_gaussian = NULL;
	uint8_t* GPU_sobel = NULL;
	uint8_t* GPU_angle = NULL;
	cudaMalloc((void**)&GPU_gaussian, 2400000 * sizeof(uint8_t)); //GPU_gaussian = gaussian for device
	cudaMalloc((void**)&GPU_sobel, 2400000 * sizeof(uint8_t)); //GPU_sobel = gaussian for device
	cudaMalloc((void**)&GPU_angle, 800000 * sizeof(uint8_t)); //GPU_angle = gaussian for device
	cudaMemcpy(GPU_gaussian, gaussian, 2400000 * sizeof(uint8_t), cudaMemcpyHostToDevice);
	//memory set(allocate + copy) for device

	//gaussian_to_sobel_and_angle_global << <GridDim, BlockDim >> > (GPU_gaussian, GPU_sobel, GPU_angle);
	gaussian_to_sobel_and_angle_shared << <GridDim, BlockDim >> > (GPU_gaussian, GPU_sobel, GPU_angle);
	//!!!

	cudaMemcpy(sobel, GPU_sobel, 2400000 * sizeof(uint8_t), cudaMemcpyDeviceToHost);
	cudaMemcpy(angle, GPU_angle, 800000 * sizeof(uint8_t), cudaMemcpyDeviceToHost);
	cudaFree(GPU_gaussian);
	cudaFree(GPU_sobel);
	cudaFree(GPU_angle);
	//memory clean(copy + free) for device
	
	return;
}

__global__ void angle_and_sobel_to_suppression_pixel_global(const uint8_t* angle, const uint8_t* sobel, uint8_t* suppression_pixel, uint8_t* p_min, uint8_t* p_max) //angle, sobel, suppression_pixel, min, max in global memory
{
	const unsigned int i = blockDim.x * blockIdx.x + threadIdx.x; //get index (index is 0 ~ 2399999)
	const int width = 1000, height = 800;
	uint8_t p1, p2;
	uint8_t v;
	int row, col;
	uint8_t tmp;
	//i, width, height, p1, p2, v, row, col, tmp in register

	if (i > 2399999) //terminate dummy thread (remain index is 0 ~ 2399999)
		return;

	else //index is 0 ~ 2399999
	{
		row = (i / 3) / width; //row is 0, 1 ... 798, 799
		col = (i / 3) % width; //col is 0, 1 ... 998, 999

		if (row == 0 || row == height - 1)
			return;

		else if (col == 0 || col == width - 1)
			return;

		else
		{
			//remain row is 1, 2 ... 797, 798
			//remain col is 1, 2 ... 997, 998

			tmp = angle[row * width + col]; //1 global memory read

			if (tmp == 0)
			{
				p1 = sobel[((row + 1) * width + col) * 3];
				p2 = sobel[((row - 1) * width + col) * 3];
			} //2 global memory read

			else if (tmp == 45)
			{
				p1 = sobel[((row + 1) * width + col - 1) * 3];
				p2 = sobel[((row - 1) * width + col + 1) * 3];
			} //2 global memory read
			 
			else if (tmp == 90)
			{
				p1 = sobel[((row)*width + col + 1) * 3];
				p2 = sobel[((row)*width + col - 1) * 3];
			} //2 global memory read

			else
			{
				p1 = sobel[((row + 1) * width + col + 1) * 3];
				p2 = sobel[((row - 1) * width + col - 1) * 3];
			} //2 global memory read

			v = sobel[(row * width + col) * 3]; //1 global memory read

			if (*(p_min) > v) //1 global memory read
				*(p_min) = v; //1 global memory write (optional)

			if (*(p_max) < v) //1 global memory read
				*(p_max) = v; //1 global memory write (optional)

			if ((v >= p1) && (v >= p2))
				suppression_pixel[i] = v;
			else
				suppression_pixel[i] = 0;
			//1 global memory write

			return;
		}
	}
}
//(9 global memory access) per thread
//for GPU_Noise_Reduction

__global__ void angle_and_sobel_to_suppression_pixel_shared(const uint8_t* angle, const uint8_t* sobel, uint8_t* suppression_pixel, uint8_t* p_min, uint8_t* p_max) //angle, sobel, suppression_pixel, min, max in global memory
{
	const unsigned int i = blockDim.x * blockIdx.x + threadIdx.x; //get index (i is 0 ~ 799999)
	const unsigned int tx = threadIdx.x; //get thread index (unique inside block) (thread index is 0 ~ 999)
	const unsigned int index = i * 3; //index for accessing gray or gaussian (index is 0 ~ 2399997)

	const int width = 1000, height = 800;
	uint8_t p1, p2;
	uint8_t v;
	int row, col;
	int tmp;
	//i, tx, index width, height, p1, p2, v, row, col, tmp in register

	__shared__ uint8_t shared_sobel[3000];
	//there are 1000 threads in 1 block (length of shared_sobel is 3000)
	//and there are 800 blocks in 1 grid (there are 800 shared_sobel in 1 grid)
	//shared_sobel in shared memory

	if (index > 2399999) //terminate dummy thread (remain index is 0 ~ 2399999)
		return;

	if (index >= 3000) //when row is 1, 2 ... 7998, 7999
		shared_sobel[tx] = sobel[index - 3000]; //copy single element global memory -> shared memory (each thread)
	shared_sobel[tx + 1000] = sobel[index]; //copy single element global memory -> shared memory (each thread)
	if (index < 2397000) //when row is 0, 1 ... 7997, 7998
		shared_sobel[tx + 2000] = sobel[index + 3000]; //copy single element global memory -> shared memory (each thread)
	//3 global memory read, 3 shared memory write
	__syncthreads();

	if(1) //index is 0 ~ 2399999
	{
		row = (index / 3) / width; //row is 0, 1 ... 798, 799
		col = (index / 3) % width; //col is 0, 1 ... 998, 999

		if (row == 0 || row == height - 1)
			return;

		else if (col == 0 || col == width - 1)
			return;

		else
		{
			//remain row is 1, 2 ... 797, 798
			//remain col is 1, 2 ... 997, 998

			tmp = angle[row * width + col]; //1 global memory read

			if (tmp == 0)
			{
				p1 = shared_sobel[tx + 2000];
				p2 = shared_sobel[tx];
			}
			else if (tmp == 45)
			{
				p1 = shared_sobel[tx + 2000 - 1];
				p2 = shared_sobel[tx + 1];
			}
			else if (tmp == 90)
			{
				p1 = shared_sobel[tx + 1000 + 1];
				p2 = shared_sobel[tx + 1000 - 1];
			}
			else
			{
				p1 = shared_sobel[tx + 2000 + 1];
				p2 = shared_sobel[tx - 1];
			}
			//2 shared memory read

			v = shared_sobel[tx + 1000]; //1 shared memory read

			if (*(p_min) > v) //1 global memory read
				*(p_min) = v; //1 global memory write (optional)
			if (*(p_max) < v) //1 global memory read
				*(p_max) = v; //1 global memory write (optional)

			if ((v >= p1) && (v >= p2))
			{
				suppression_pixel[index] = v;
				suppression_pixel[index + 1] = v;
				suppression_pixel[index + 2] = v;
			}
			else
			{
				suppression_pixel[index] = 0;
				suppression_pixel[index + 1] = 0;
				suppression_pixel[index + 2] = 0;
			}
			//3 global memory write

			return;
		}
	}
}
//(11(3 + 1 + 2 + 3) global memory access + 6(3 + 3) shared memory access + 1 syncthreads) per thread
//for GPU_Noise_Reduction

__host__ void GPU_Non_maximum_Suppression(int width, int height, uint8_t* angle, uint8_t* sobel, uint8_t* suppression_pixel, uint8_t& min, uint8_t& max)
{
	//cout << width << endl;
	//cout << height << endl;
	//cout << _msize(angle) << endl;
	//cout << _msize(sobel - 54) << endl;
	//cout << _msize(suppression_pixel - 54) << endl;
	//cout << (int)min << endl;
	//cout << (int)max << endl;

	dim3 GridDim(2400, 1, 1); //800 blocks in 1 grid
	dim3 BlockDim(1000, 1, 1); //1000 threads in 1 block
	//800000 threads in 1 grid

	uint8_t* GPU_angle = NULL;
	uint8_t* GPU_sobel = NULL;
	uint8_t* GPU_suppression_pixel = NULL;
	uint8_t* GPU_min = NULL;
	uint8_t* GPU_max = NULL;
	cudaMalloc((void**)&GPU_angle, 800000 * sizeof(uint8_t)); //GPU_angle = angle for device
	cudaMalloc((void**)&GPU_sobel, 2400000 * sizeof(uint8_t)); //GPU_sobel = sobel for device
	cudaMalloc((void**)&GPU_suppression_pixel, 2400000 * sizeof(uint8_t)); //GPU_suppression_pixel = suppression_pixel for device
	cudaMalloc((void**)&GPU_min, sizeof(uint8_t)); //GPU_min = min for device
	cudaMalloc((void**)&GPU_max, sizeof(uint8_t)); //GPU_min = min for device
	cudaMemcpy(GPU_angle, angle, 800000 * sizeof(uint8_t), cudaMemcpyHostToDevice);
	cudaMemcpy(GPU_sobel, sobel, 2400000 * sizeof(uint8_t), cudaMemcpyHostToDevice);
	cudaMemcpy(GPU_min, &min, sizeof(uint8_t), cudaMemcpyHostToDevice);
	cudaMemcpy(GPU_max, &max, sizeof(uint8_t), cudaMemcpyHostToDevice);
	//memory set(allocate + copy) for device

	angle_and_sobel_to_suppression_pixel_global << <GridDim, BlockDim >> > (GPU_angle, GPU_sobel, GPU_suppression_pixel, GPU_min, GPU_max);
	//angle_and_sobel_to_suppression_pixel_shared << <GridDim, BlockDim >> > (GPU_angle, GPU_sobel, GPU_suppression_pixel, GPU_min, GPU_max);
	//!!!

	cudaMemcpy(&min, GPU_min, sizeof(uint8_t), cudaMemcpyDeviceToHost);
	cudaMemcpy(&max, GPU_max, sizeof(uint8_t), cudaMemcpyDeviceToHost);
	//cout << (int)min << (int)max << endl;

	cudaMemcpy(suppression_pixel, GPU_suppression_pixel, 2400000 * sizeof(uint8_t), cudaMemcpyDeviceToHost);
	cudaFree(GPU_angle);
	cudaFree(GPU_sobel);
	cudaFree(GPU_suppression_pixel);
	cudaFree(GPU_min);
	cudaFree(GPU_max);
	//memory clean(copy + free) for device
}

__global__ void suppression_pixel_to_hysteresis_global(const uint8_t* suppression_pixel, uint8_t* hysteresis) //suppression_pixel, hysteresis in global memory
{
	const unsigned int i = blockDim.x * blockIdx.x + threadIdx.x; //get index (index is 0 ~ 2399999)
	const int width = 1000, height = 800;
	const uint8_t low_t = 2;
	const uint8_t high_t = 51;
	uint8_t v;
	int row, col;
	int k, l;
	uint8_t done = 0;
	uint8_t tmp;
	//i, width, height, low_t, high_t, row, col, k, l, done, tmp in register

	if (i > 2399999) //terminate dummy thread (remain index is 0 ~ 2399999)
		return;

	else //index is 0 ~ 2399999
	{
		v = suppression_pixel[i]; //1 global memory read

		if (v < low_t)
			v = 0;
		else if (v < high_t)
			v = 123;
		else
			v = 255;

		if (v == 123)
		{
			row = (i / 3) / width; //row is 0, 1 ... 798, 799
			col = (i / 3) % width; //col is 0, 1 ... 998, 999

			for (k = row - 1; k <= row + 1; k++) //k is -1, 0 ... 799, 800
			{
				for (l = col - 1; l <= col + 1; l++) //l is -1, 0 ... 999, 1000
				{
					if ((k < height && l < width) && (k >= 0 && l >= 0))
					{
						//remamin k is 0, 1 ... 798, 799
						//remamin l is 0, 1 ... 998, 999

						tmp = suppression_pixel[(k * width + l) * 3]; //9 global memory read

						if (tmp < low_t)
							tmp = 0;
						else if (tmp < high_t)
							tmp = 123;
						else
							tmp = 255;

						if (tmp == 255)
						{
							v = 254;
							done = 1;
							break;
						}
					}
				}

				if (done == 1)
					break;
			}
		}

		if (v != 255 && v != 254)
			v = 0;

		hysteresis[i] = v; //1 global memory write

		return;
	}
}
//(11 global memory access) per thread
//for GPU_Noise_Reduction

__global__ void suppression_pixel_to_hysteresis_shared(const uint8_t* suppression_pixel, uint8_t* hysteresis) //suppression_pixel, hysteresis in global memory
{
	const unsigned int i = blockDim.x * blockIdx.x + threadIdx.x; //get index (i is 0 ~ 799999)
	const unsigned int tx = threadIdx.x; //get thread index (unique inside block) (thread index is 0 ~ 999)
	const unsigned int index = i * 3; //index for accessing gray or gaussian (index is 0 ~ 2399997)

	const int width = 1000, height = 800;
	const uint8_t low_t = 2;
	const uint8_t high_t = 51;
	uint8_t v;
	int row, col;
	int k, l;
	uint8_t done = 0;
	uint8_t tmp;
	//i, width, height, low_t, high_t, row, col, k, l, done, tmp in register

	__shared__ uint8_t shared_hysteresis[3000];
	//there are 1000 threads in 1 block (length of shared_hysteresis is 3000)
	//and there are 800 blocks in 1 grid (there are 800 shared_hysteresis in 1 grid)
	//shared_hysteresis in shared memory

	if (index > 2399999) //terminate dummy thread (remain index is 0 ~ 2399999)
		return;

	if(1) //index is 0 ~ 2399999
	{
		if (index >= 3000) //when row is 1, 2 ... 7998, 7999
		{
			tmp = suppression_pixel[index - 3000]; //1 global memory read

			if (tmp < low_t)
				shared_hysteresis[tx] = 0;
			else if (tmp < high_t)
				shared_hysteresis[tx] = 123;
			else
				shared_hysteresis[tx] = 255;
			//1 shared memory write
		}

		v = suppression_pixel[index]; //1 global memory read

		if (v < low_t)
			v = shared_hysteresis[tx + 1000] = 0;
		else if (v < high_t)
			v = shared_hysteresis[tx + 1000] = 123;
		else
			v = shared_hysteresis[tx + 1000] = 255;
		//1 shared memory write

		if (index < 2397000) //when row is 0, 1 ... 7997, 7998
		{
			tmp = suppression_pixel[index + 3000];

			if (tmp < low_t)
				shared_hysteresis[tx + 2000] = 0;
			else if (tmp < high_t)
				shared_hysteresis[tx + 2000] = 123;
			else
				shared_hysteresis[tx + 2000] = 255;
			//1 shared memory write
		}

		__syncthreads();
		
		if (v == 123)
		{
			row = (index / 3) / width; //row is 0, 1 ... 798, 799
			col = (index / 3) % width; //col is 0, 1 ... 998, 999

			for (k = row - 1; k <= row + 1; k++) //k is -1, 0 ... 799, 800
			{
				for (l = col - 1; l <= col + 1; l++) //l is -1, 0 ... 999, 1000
				{
					if ((k < height && l < width) && (k >= 0 && l >= 0))
					{
						//remamin k is 0, 1 ... 798, 799
						//remamin l is 0, 1 ... 998, 999

						if (shared_hysteresis[tx + 1000 + (k - row) * 1000 + (l - col)] == 255) //9 shared memory read
						{
							v = 254;

							done = 1;
							break;
						}
					}
				}

				if (done == 1)
					break;
			}
		}
	
		if (v != 255 && v != 254)
			v = 0;

		hysteresis[index] = v;
		hysteresis[index + 1] = v;
		hysteresis[index + 2] = v;
		//3 global memory write

		return;
	}
}
//(6 global memory access + 12(3 + 9) shared memory access + 1 syncthreads) per thread
//for GPU_Noise_Reduction

__host__ void GPU_Hysteresis_Thresholding(int width, int height, uint8_t* suppression_pixel, uint8_t* hysteresis, uint8_t min, uint8_t max)
{
	//cout << width << endl;
	//cout << height << endl;
	//cout << _msize(suppression_pixel - 54) << endl;
	//cout << _msize(hysteresis - 54) << endl;
	//cout << (int)min << endl;
	//cout << (int)max << endl;

	dim3 GridDim(800, 1, 1); //800 blocks in 1 grid
	dim3 BlockDim(1000, 1, 1); //1000 threads in 1 block
	//800000 threads in 1 grid

	uint8_t* GPU_suppression_pixel = NULL;
	uint8_t* GPU_hysteresis = NULL;
	cudaMalloc((void**)&GPU_suppression_pixel, 2400000 * sizeof(uint8_t)); //GPU_s = gaussian for device
	cudaMalloc((void**)&GPU_hysteresis, 2400000 * sizeof(uint8_t)); //GPU_gaussian = gaussian for device
	cudaMemcpy(GPU_suppression_pixel, suppression_pixel, 2400000 * sizeof(uint8_t), cudaMemcpyHostToDevice);
	//memory set(allocate + copy) for device

	//suppression_pixel_to_hysteresis_global << <GridDim, BlockDim >> > (GPU_suppression_pixel, GPU_hysteresis);
	suppression_pixel_to_hysteresis_shared << <GridDim, BlockDim >> > (GPU_suppression_pixel, GPU_hysteresis);
	//!!!

	cudaMemcpy(hysteresis, GPU_hysteresis, 2400000 * sizeof(uint8_t), cudaMemcpyDeviceToHost);
	cudaFree(GPU_suppression_pixel);
	cudaFree(GPU_hysteresis);
	//memory clean(copy + free) for device

	return;
}