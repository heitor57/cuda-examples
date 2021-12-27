#include <stdio.h>
#include <stdlib.h>


// R é o raio de soma de cada thread
#define R 2

// N é o tamanho do vetor
#define N 32


__global__ void soma_numeros(int *a, int* res){
   __shared__ int temp[N/R];
	int count = 1;
	int ind = threadIdx.x;
	int num_threads = N/R;
	int remainder_elements = N;
	while(num_threads > 1){
		if(ind < num_threads){
			int soma = 0;
			if(count == 1){
				for(int i = R*ind; i < R*ind + R; i++){
					soma = soma + a[i];
				}
			}else{
				for(int i = R*ind; i < R*ind + R; i++){
					soma = soma + temp[i];
				}
			}
			temp[ind] = soma;
		}
		__syncthreads();
		count++;
		remainder_elements = num_threads;
		num_threads=num_threads/R;
	}
	if(ind == 0){
		*res = 0;
		if(count == 1){
			for(int i=0; i < remainder_elements;i++)
				*res+=a[i];
		}else{
			for(int i=0; i < remainder_elements;i++)
				*res+=temp[i];
		}
	}
}

//código host
int main(){
	int a[N];
	int res=0;
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
	dim3 numThreads((N+R-1)/R);
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
