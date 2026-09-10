#include <cstdlib>
#include <stdint.h>
#include <iostream>
#include <cuda_runtime.h>

#define MAX_BORDERS 12
#define MIN_WHITE 170
#define MAX_BLACK 150

__global__ void analyseLineKernel(uint8_t *gray, uint8_t *canny, int width, int height, int *result)
{
    //int tid = blockIdx.x * blockDim.x + threadIdx.x;
    //int y = tid * 10;
    //int y = threadIdx.x ;
    int y = blockIdx.x * blockDim.x + threadIdx.x;

    if (y >= height) return;
    int delta[MAX_BORDERS];
    int bord[MAX_BORDERS];
    float mean_color[MAX_BORDERS];
    int previous_x = 0;
    int nb_bord = 0; // nombre de bords
    int sum = 0; 
    for (int x = 0; x < width; x++)
	{
		sum += gray[y * width + x];
		if (canny[y * width + x] != 0 && x-previous_x > 9) {
			bord[nb_bord] = x;
			delta[nb_bord] = x - previous_x;
			mean_color[nb_bord] = (float) sum / (x - previous_x + 1);
			sum = 0;
			previous_x = x;
			if (nb_bord >= MAX_BORDERS-1) break;
			nb_bord++;
		}
	}

    if (nb_bord > 3) //y==344
    {
	    for (int idx = 2; idx < nb_bord; idx++)
	    {
            if (mean_color[idx] > MIN_WHITE && mean_color[idx - 1] < MAX_BLACK && mean_color[idx + 1] < MAX_BLACK &&
                abs(delta[idx] - delta[idx - 1]) < delta[idx] / 2 && abs(delta[idx] - delta[idx + 1]) < delta[idx] / 2)
            {
                result[y] = (bord[idx] + bord[idx - 1]) / 2;
                return;
            }
	    }
    }
    result[y] = -1;
    return;
}

void submitLineAnalysis(uint8_t *d_gray, uint8_t *d_canny, int width, int height, cudaStream_t &stream, int *d_result)
{
	// image 720 x 1280
    /*
	std::cout << width << "\n";
	std::cout << height << "\n";
	cudaPointerAttributes attr;
	cudaPointerGetAttributes(&attr, d_result);
	printf("Pointeur sur le GPU ? %d\n", attr.type == cudaMemoryTypeDevice);
	cudaPointerGetAttributes(&attr, d_gray);
	printf("Pointeur sur le GPU ? %d\n", attr.type == cudaMemoryTypeDevice);
	cudaPointerGetAttributes(&attr, d_canny);
	printf("Pointeur sur le GPU ? %d\n", attr.type == cudaMemoryTypeDevice);

	std::cout << "height" << height <<"\n";
    */
	int nbLines = height;   // 72
	int threads = 256;
	int blocks = (nbLines + threads - 1) / threads;

	analyseLineKernel<<<blocks, threads, 0, stream>>>(d_gray, d_canny, width, height, d_result);
	//std::cout << "sortie" << "\n";
	cudaError_t err = cudaGetLastError();
    if (err) std::cout << "erreur:" << err << "\n";
}
