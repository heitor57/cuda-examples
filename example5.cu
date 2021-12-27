#include <stdio.h>
#include <stdlib.h>

#define N 6
#define R 2
//código device
// blockDim.x é a dimensão do bloco, ou seja,
// a quantidade de threads por bloco.


__global__ void soma_numeros_adjacentes(int *a, int* res){
   __shared__ int temp[N];
	
	int ind = threadIdx.x;
	if(ind < N){
		int initial_pos = ind - R;
		if(initial_pos < 0){
			initial_pos = 0;
		}

		int final_pos = ind + R + 1;
		if(final_pos >= N){
			final_pos = N;
		}
		temp[ind] = 0;
		for(int i = initial_pos; i < final_pos; i++){
			temp[ind] += a[i];
		}
	}
	res[ind] = temp[ind];

}

//código host
int main(){
	int a[N];
	int res[N];
	int* dev_a;
	int* dev_res;

	int tam = N*sizeof(int);

	//inicializando as variaveis do host:
	for(int i=0; i < N; i++){
		a[i] = i+1;
		printf("%d ",a[i]);
	}
	printf("\n");
	//alocando espaço para as variaveis da GPU:
	cudaMalloc((void**)&dev_a,tam);
	cudaMalloc((void**)&dev_res,tam);

	//copiando as variaveis da CPU para a GPU:
	cudaMemcpy(dev_a, &a, tam, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_res, &res, tam, cudaMemcpyHostToDevice);
	//chamada da função da gpu (kernel):
	// Número de blocos é igual a dimensão do vetor
	// divida pela dimensão do bloco. N/M

	// O tipo dim3 permite definir a quantidade de
	// blocos e threads por dimensão
	//dim3 numBlocos(2,2);// número de blocos é igual a 2x2 = 4
	//dim3 numThreads(1,2);// número de threads por bloco = 2x2 = 4

	dim3 numBlocos(1);
	dim3 numThreads(N);
	soma_numeros_adjacentes<<<numBlocos,numThreads>>>(dev_a, dev_res);

	//copiando o resultado da GPU para a CPU:
	cudaMemcpy(&res, dev_res, tam, cudaMemcpyDeviceToHost);

	//vizualizando o resultado:

	for(int i=0; i < N; i++){
		a[i] = i;
		printf("%d ",res[i]);
	}

	printf("\n\n");

	//liberando a memoria na GPU:
	cudaFree(dev_a);
	cudaFree(dev_res);

	return 0;
}
