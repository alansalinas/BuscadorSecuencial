// Buscador.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"

#include <math.h>

#include "gpu.h"





int getOhms(int R){
	
	int banda1 = R / 100;
	int banda2 = (R / 10)%10;
	int m = R % 10;
	int multiplicador = 1;

	for (int k = 0; k < m; k++){
		multiplicador *= 10;
	}
	// intsssss
	int ohms = ((banda1 * 10) + banda2)* multiplicador;

	return ohms;
}



int _tmain(int argc, _TCHAR* argv[])
{

	const int size = 999999;	// tamaño del espacio de busqueda

	float *r = new float[size];
	//
	// Preparar espacio de busqueda

	printf("Buscador:\n");


	int RL;
	float Vi;
	float Vd;

	printf("RL (int): ");
	scanf_s("%d", &RL);
	printf("Vi (float): ");
	scanf_s("%f", &Vi);
	printf("V deseado (float): ");
	scanf_s("%f", &Vd);

	
	//Invocar la rutina para funcion objetivo con CUDA
	cudaError_t cudaStatus = evaluarCuda(r, Vi, RL, Vd, size);
	if (cudaStatus != cudaSuccess) {
		printf("Error: addWithCuda\n");
		return 1;
	}

	//Asegurar la finalización de los "threads", con propósitos de rastreo de errores ("profilin")
	cudaStatus = cudaThreadExit();
	if (cudaStatus != cudaSuccess) {
		printf("Error: cudaThreadExit\n");
		return 1;
	}
	

	printf("EXITO!!: \n");

	float min = 999999.9;
	float max = 0;
	int maxPos = -1;
	int minPos = -1;

	for (int k = 0; k < size; k++)
	{
		if (r[k] < min && r[k]>0.001)
		{
			min = r[k];
			minPos = k;
		}

		if (r[k] > max)
		{
			max = r[k];
			maxPos = k;
		}
	}

	printf("results:\n");
	printf("mejor encontrado[%d](minimo): %f, con R1: %d y R2: %d\n", minPos, min, getOhms(minPos/1000), getOhms(minPos%1000));
	printf("mejor encontrado[%d](maximo): %f, con R1: %d y R2: %d\n", maxPos, max, getOhms(maxPos / 1000), getOhms(maxPos % 1000));

	
	delete[] r;

	getchar();
	getchar();
	return 0;
}

