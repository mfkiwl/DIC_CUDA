#include"Header.h"

#include <cufft.h>

#include <cufftw.h>
#include<device_functions.h>
#include <cstdlib>
#include <cstdio>
#include <cstring>

__global__ void conjugate(cuFloatComplex *comp)
{
	int tid = threadIdx.x + threadIdx.y*blockDim.x;
	int bid = blockIdx.x;
	int dim = blockDim.x*blockDim.y;
	comp[bid*dim+tid].y = -comp[bid*dim + tid].y;

}



__global__ void multiplication(cuFloatComplex *comp1, cuFloatComplex *comp2)
{
	int tid = threadIdx.x + threadIdx.y*blockDim.x;
	int bid = blockIdx.x;
	int dim = blockDim.x*blockDim.y;
	float t;
	t = comp1[bid*dim + tid].x;
	comp1[bid*dim + tid].x = comp1[bid*dim + tid].x * comp2[bid*dim + tid].x - comp1[bid*dim + tid].y * comp2[bid*dim + tid].y;
	comp1[bid*dim + tid].y = t * comp2[bid*dim + tid].y + comp1[bid*dim + tid].y * comp2[bid*dim + tid].x;
}

__global__ void latched_position(double *Mats, double *displacement_x, double *displacement_y,int subsetsize)
{
	int x = threadIdx.x;
	int y = threadIdx.y;
	int bid = blockIdx.x;
	int dim = blockDim.x*blockDim.y;
	__shared__ double max[64 * 64];
	__shared__ int a[64 * 64], b[64 * 64];
	if (x + dim*y == 0)
	{
		for (int j = 0;j < 64 * 64;j++)
		{
			max[j] = 0;
			a[j] = 0;
			b[j] = 0;
		}
	}
	max[x + blockDim.x*y] = Mats[bid*dim+x + blockDim.x*y]/subsetsize;
	int k = 64*64/ 2;
	a[x + blockDim.x*y] = x;
	b[x + blockDim.x*y] = y;
	while (k != 0)
	{
		if (max[x + blockDim.x*y] < max[x + blockDim.x*y + k])
		{
			max[x + blockDim.x*y] = max[x + blockDim.x*y + k];
			a[x + blockDim.x*y] = a[x + blockDim.x*y + k];
			b[x + blockDim.x*y] = b[x + blockDim.x*y + k];
		}
		__syncthreads();
		k = k / 2;

	}
	if (x + blockDim.x*y == 0)
	{
		if (max[0] < 0.03)
		{
			displacement_x[bid] = 0;
			displacement_y[bid] = 0;
		}
		else
		{

			displacement_x[bid] = a[0];
			displacement_y[bid] = b[0];
		}
	}
}



void CU_FFTCC(int iSubsetH, int iSubsetW, int iNumberX, int iNumberY, float *subset_aveR, float *subset_aveT, int d_iU, int d_iV)
{
	cuFloatComplex *CU_FT_R, *CU_FT_T;
	cufftReal * CuFFT;
	int iNumbersize = iNumberX*iNumberY;
	int subsetsize = iSubsetH*iSubsetH;
	checkCudaErrors(cudaMalloc((void **)&CU_FT_R, iNumbersize* iSubsetH *(iSubsetW / 2 + 1) * sizeof(cuFloatComplex)));
	checkCudaErrors(cudaMalloc((void **)&CU_FT_T, iNumbersize* iSubsetH *(iSubsetW / 2 + 1) * sizeof(cuFloatComplex)));
	checkCudaErrors(cudaMalloc((void **)&CuFFT, iNumbersize*iSubsetH *iSubsetW * sizeof(cufftReal)));

	cufftHandle plan1, plan2;
	checkCudaErrors(cufftPlan2d(&plan1, iSubsetH, iSubsetW, CUFFT_R2C));
	checkCudaErrors(cufftPlan2d(&plan2, iSubsetH, iSubsetW, CUFFT_C2R));
	dim3 block(iNumbersize);
	dim3 threadnew(iSubsetH, iSubsetW / 2 + 1);
	dim3 thread(iSubsetH, iSubsetW);
	for (int i = 0;i < iNumbersize;i++)
	{
		checkCudaErrors(cufftExecR2C(plan1, (cufftReal *)(subset_aveR + 1 + i * (subsetsize + 1)), (cuFloatComplex *)(CU_FT_R + i*subsetsize)));
		checkCudaErrors(cufftExecR2C(plan1, (cufftReal *)(subset_aveT + 1 + i * (subsetsize + 1)), (cuFloatComplex *)(CU_FT_T + i*subsetsize)));
	}
	conjugate << <block, threadnew >> > (CU_FT_R);
	multiplication << <block, threadnew >> > (CU_FT_R, CU_FT_T);

	for (int j = 0;j < iNumbersize;j++)
	{
		checkCudaErrors(cufftExecC2R(plan2, (cuFloatComplex *)CU_FT_R, (cufftReal *)(CuFFT + subsetsize)));
	}


	latched_position << <block, thread2 >> > (CuFFT, d_iU, d_iV, subsetsize);
}
