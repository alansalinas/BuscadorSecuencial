
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

cudaError_t evaluarCuda(float *c, float vi, int rl, float vd, size_t size);

__global__ void objetivo(float *c, float vi, int rl, float vd)
{
	// obtener coordenada del espacio de busqueda
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	// obtener r1 y r2 en formato R1_ R1_ R1_ R1_ R2_ R2_ R2_ R2_

	int r1 = i / 1000;
	int r2 = i % 1000;

	// getOhms()
	int banda1 = r1 / 100;
	int banda2 = (r1 / 10) % 10;
	int m = r1 % 10;
	int multiplicador = 1;

	for (int k = 0; k < m; k++){
		multiplicador *= 10;
	}

	int ohmsR1 = ((banda1 * 10) + banda2) * multiplicador;

	// getOhms para R2
	banda1 = r2 / 100;
	banda2 = (r2 / 10) % 10;
	m = r2 % 10;
	multiplicador = 1;

	for (int k = 0; k < m; k++){
		multiplicador *= 10;
	}

	int ohmsR2 = ((banda1 * 10) + banda2) * multiplicador;
	
	int req = (ohmsR1*ohmsR2) / (ohmsR1 + ohmsR2);	// resistencia equivalente

	float vl = vi * ((ohmsR1*rl) / (ohmsR1 + rl)) / (ohmsR2 + (ohmsR1*rl) / (ohmsR1 + rl));	// voltaje en la carga

	// minimizar funcion objetivo
	float f = 0.4*(req - rl)*(req - rl) + (vl - vd)*(vl - vd);
	c[i] = f;

	//c[i] = i;
	//printf("thread %d: \n", i);

}



//Rutina 'lanzadora' del "kernel"
cudaError_t evaluarCuda(float *c, float vi, int rl, float vd, size_t size)
{
	//float *dev_a = 0;
	float *dev_c = 0;
	cudaError_t cudaStatus;

	//Seleccionar el GPU a utilizar
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		printf("Error: cudaSetDevice\n");
		goto Error;
	}

	//Solicitar memoria para los vectores a utilizar dentro del GPU
	cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		printf("Error: cudaMalloc\n");
		goto Error;
	}
	/*
	cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		printf("Error: cudaMalloc\n");
		goto Error;
	}
	


	//Copiar los datos a la memoria del GPU
	cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		printf("Error: cudaMalloc\n");
		goto Error;
	}
	*/

	int BLOCK_SIZE, grid;

	if (size < 512){
		BLOCK_SIZE = size;
		grid = 1;
	}
	else
	{
		BLOCK_SIZE = 512;
		grid = size / BLOCK_SIZE;
	}

	//dim3 threadBlock(BLOCK_SIZE, BLOCK_SIZE);
	//dim3 grid(K, K);

	//'Lanzar' el "kernel"
	objetivo << <grid, BLOCK_SIZE >> >(dev_c, vi, rl, vd);

	//Esperar a que terminen los "threads"
	cudaStatus = cudaThreadSynchronize();
	if (cudaStatus != cudaSuccess) {
		printf("Error: cudaThreadSynchronize (código de error: %d reportado por objetivo)\n", cudaStatus);
		goto Error;
	}

	//Recuperar los resultados de la memoria del GPU
	cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		printf("Error: cudaMemcpy\n");
		goto Error;
	}

Error:
	cudaFree(dev_c);
	//cudaFree(dev_a);

	return cudaStatus;
}
