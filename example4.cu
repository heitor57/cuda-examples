#include <stdio.h>
#include <stdlib.h>

#define N 32

//código device
// blockDim.x é a dimensão do bloco, ou seja,
// a quantidade de threads por bloco.


__global__ void soma_numeros(int *a, int* res){
	__shared__ int temp[N/4];
	int ind = threadIdx.x;
	if(ind < N/4){
		int soma = 0;
		for(int i = 4*ind; i < 4*ind + 4; i++){
			soma = soma + a[i];
		}
		temp[ind] = soma;
	}

	__syncthreads();

	if(ind == 0){
		int soma_final = 0;
		for(int i = 0; i< N/4; i++)
			soma_final += temp[i];
		*res = soma_final;
	}
}

//código host
int main(){
	int a[N];
	int res;
	int* dev_a;
	int* dev_res;

	int tam = N*sizeof(int);

	//inicializando as variaveis do host:
	for(int i=0; i < N; i++){
		a[i] = i;
		printf("%d ",a[i]);
	}

	printf("\n");
	//alocando espaço para as variaveis da GPU:
	cudaMalloc((void**)&dev_a,tam);
	cudaMalloc((void**)&dev_res,sizeof(int));

	//copiando as variaveis da CPU para a GPU:
	cudaMemcpy(dev_a, &a, tam, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_res, &res, sizeof(int), cudaMemcpyHostToDevice);
	//chamada da função da gpu (kernel):
	// Número de blocos é igual a dimensão do vetor
	// divida pela dimensão do bloco. N/M

	// O tipo dim3 permite definir a quantidade de
	// blocos e threads por dimensão
	//dim3 numBlocos(2,2);// número de blocos é igual a 2x2 = 4
	//dim3 numThreads(1,2);// número de threads por bloco = 2x2 = 4

	dim3 numBlocos(1);
	dim3 numThreads(N/4);
	soma_numeros<<<numBlocos,numThreads>>>(dev_a, dev_res);

	//copiando o resultado da GPU para a CPU:
	cudaMemcpy(&res, dev_res, sizeof(int), cudaMemcpyDeviceToHost);

	//vizualizando o resultado:
	printf("%d",res);

	printf("\n\n");

	//liberando a memoria na GPU:
	cudaFree(dev_a);
	cudaFree(dev_res);

	return 0;
}
