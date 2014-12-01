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

int getR1(int n)
{
	return getOhms(n / 1000);
}

int getR2(int n)
{
	return getOhms(n % 1000);
}

int getBand3(int R)
{
	unsigned int mult = 0;
	unsigned int n = R;

	do{
		if (n % 10 == 0)
			++mult;

		n /= 10;
	} while (n);


	return mult;
}

int getBand2(int R)
{
	int m = getBand3(R);
	int mult = 1;

	for (int i = 0; i < m; i++)
		mult *= 10;

	int n = R / mult;

	return n % 10;

}

int getBand1(int R)
{
	int m = getBand3(R);
	int mult = 1;

	for (int i = 0; i < m; i++)
		mult *= 10;

	int n = R / mult;

	return n / 10;
}

void getr1r2(int i, int& r1, int& r2, int R1, int R2, int m1, int m2){
	int ohms1 = 0;
	int ohms2 = 0;

	for (int k = 0; k < m2; k++)
	{
		ohms2 += i % 10;
		i = i / 10;
	}

	for (int k = 0; k < m1; k++)
	{
		ohms1 += i % 10;
		i = i / 10;
	}

	// obtener r1 r2


	r1 = R1 + ohms1;
	r2 = R2 + ohms2;

}

int getDimensionSize(int r){

	unsigned int digits = 0;

	unsigned int n = r;

	do{
		if (n % 10 == 0)
			++digits;

		n /= 10;
	} while (n);

	return digits;
}

// Obtiene tamaño de dimension necesario para calcular los valores en ohm en el gpu dados r1 y r2 comerciales
int getTotalDimensionSize(int r1, int r2){

	int digitsR1 = getDimensionSize(r1);

	int digitsR2 = getDimensionSize(r2);

	return digitsR1 + digitsR2;

}

int lanzaBandas(float *c, float vi, int rl, float vd, size_t size){
	//Invocar la rutina para funcion objetivo con CUDA
	cudaError_t cudaStatus = bandasCuda(c, vi, rl, vd, size);
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

	return 0;
}

int lanzaOhms(float *c, float vi, int rl, float vd, int r1, int r2, int m1, int m2, size_t size){
	//Invocar la rutina para funcion objetivo con CUDA
	cudaError_t cudaStatus = ohmsCuda(c, vi, rl, vd, r1, r2, m1, m2, size);
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

	return 0;
}

int mejorEncontradoPos(float *r, size_t size){

	float min = 999999.9;
	float max = 0;
	int maxPos = -1;
	int minPos = -1;
	
	for (int k = 0; k < size; k++)
	{
		if (r[k] < min && r[k] >= 0)
		{
			min = r[k];
			minPos = k;
		}

		if (r[k] > max)
		{
			max = r[k];
			maxPos = k;
		}

		printf("min %f\n", r[k]);
	}

	printf("results:\n");
	printf("mejor encontrado[%d](minimo): %f, con R1: %d y R2: %d\n", minPos, min, getR1(minPos), getR2(minPos));
	//printf("mejor encontrado[%d](maximo): %f, con R1: %d y R2: %d\n", maxPos, max, getOhms(maxPos / 1000), getOhms(maxPos % 1000));

	return minPos;
}

int _tmain(int argc, _TCHAR* argv[])
{

	const int size = 999999;	// cantidad de intentos a evaluar
	const int ere1 = 1650;
	const int ere2 = 37;

	//float *a = new float[size];
	//float r[size] = { 0 };
	float *r = new float[size];
	//
	// Preparar espacio de busqueda


	int RL = 300;
	float Vi = 10.0;
	float Vd = 4.14;



	int dimsize = getTotalDimensionSize(ere1, ere2);
	int dim = 1;

	for (int i = 0; i < dimsize; i++)
		dim *= 10;

	printf("ere dim:%d", dim);

	int status = lanzaOhms(r, Vi, RL, Vd, ere1, ere2, getDimensionSize(ere1), getDimensionSize(ere2), dim);

	//int status = lanzaBandas(r, Vi, RL, Vd, size);

	if (status == 0){
		printf("SUCESS!!: \n");


		int min = mejorEncontradoPos(r, dim);

		int r1, r2;

		getr1r2(min, r1, r2, ere1, ere2, getDimensionSize(ere1), getDimensionSize(ere2));

		printf("R1: %d, R2: %d\n", r1, r2);
		//printf("b2 :%d\n", getBand2(getR1(min)));

		
		// imprimir vector de resultados
		for (int i = 0; i < dim; i++)
		{
		printf("Evaluacion: %f, R1: %d, R2:%d \n", r[i], getOhms(i/1000), getOhms(i%1000));
		}
	}
	else
	{
		printf("ERROR\n");
	}
	
	delete[] r;

	getchar();
	return 0;
}

