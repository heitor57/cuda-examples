#include <stdio.h>
#include <stdlib.h>

#define N 4 // Número de colunas das matrizes
#define M 6// Número de linhas das matrizes
#define T 8 // Número de threads por bloco

//código device
// blockDim.x é a dimensão do bloco, ou seja,
// a quantidade de threads por bloco.
__global__ void soma_vetor(int *a, int *b, int *c ){
	int indice = blockIdx.x*blockDim.x + threadIdx.x;
	if(indice < N)
		c[indice] = a[indice] + b[indice];
}


__global__ void soma_matriz(int *a, int* b,int *c){
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int j = blockIdx.y*blockDim.y + threadIdx.y;
	if(i<M && j<N)
		c[N*i + j] = a[N*i + j] + b[N*i + j];
}


//código host
int main(){
	int a[M*N],b[M*N],c[M*N];
	int* dev_a;
	int* dev_b;
	int* dev_c;

	int tam = M*N*sizeof(int);

	//inicializando as variaveis do host:
	for(int i=0; i < N*M; i++){
		a[i] = i;
		b[i]= i*2;
	}

	//alocando espaço para as variaveis da GPU:
	cudaMalloc((void**)&dev_a,tam);
	cudaMalloc((void**)&dev_b,tam);
	cudaMalloc((void**)&dev_c,tam);

	//copiando as variaveis da CPU para a GPU:
	cudaMemcpy(dev_a, &a, tam, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, &b, tam, cudaMemcpyHostToDevice);
	//chamada da função da gpu (kernel):
	// Número de blocos é igual a dimensão do vetor
	// divida pela dimensão do bloco. N/M

	// O tipo dim3 permite definir a quantidade de
	// blocos e threads por dimensão
	//dim3 numBlocos(2,2);// número de blocos é igual a 2x2 = 4
	//dim3 numThreads(2,2);// número de threads por bloco = 2x2 = 4

	dim3 numBlocos((M+T-1)/T, (N+T-1)/T);
	dim3 numThreads(T,T);
	soma_matriz<<<numBlocos,numThreads>>>(dev_a, dev_b, dev_c);

	//copiando o resultado da GPU para a CPU:
	cudaMemcpy(&c, dev_c, tam, cudaMemcpyDeviceToHost);

	//vizualizando o resultado:
	for(int i=0; i<M; i++){
		for(int j=0; j<N;j++)
			printf("%d ",c[i*N+j]);
		printf("\n");
	}

	printf("\n\n");

	//liberando a memoria na GPU:
	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);

	return 0;
}
