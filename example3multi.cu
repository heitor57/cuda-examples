#include <stdio.h>
#include <stdlib.h>

#define N 5 // Número de colunas da matriz 1 e numero de linhas da matriz 2
#define M 3// Número de linhas da matriz 1
#define D 4 // numero de colunas da segunda matriz
#define T 8 // Número de threads por bloco

//código device
// blockDim.x é a dimensão do bloco, ou seja,
// a quantidade de threads por bloco.

__global__ void soma_matriz(int *a, int* b,int *c){
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int j = blockIdx.y*blockDim.y + threadIdx.y;
	if(i<M && j<D){
		c[D*i+j]=0;
		int e1=N*i;
		int e2=j;
		for(int x=0;x<N;x++){
			c[D*i + j] += a[e1]*b[e2];
			e1++;
			e2+=D;
		}
	}
}


//código host
int main(){
	int a[M*N],b[D*N],c[M*D];
	int* dev_a;
	int* dev_b;
	int* dev_c;

	int tam = M*N*sizeof(int);
	int tam_2 = N*D*sizeof(int);
	int tam_3 = M*D*sizeof(int);

	//inicializando as variaveis do host:
	for(int i=0; i < N*M; i++){
		a[i] = i;
	}
	for(int i=0; i < N*D; i++){
		b[i]= i*2;
	}

	for(int i=0; i<M; i++){
		for(int j=0; j<N;j++)
			printf("%d ",a[i*N+j]);
		printf("\n");
	}

	printf("\n\n");
	for(int i=0; i<N; i++){
		for(int j=0; j<D;j++)
			printf("%d ",b[i*D+j]);
		printf("\n");
	}

	printf("\n\n");
	//alocando espaço para as variaveis da GPU:
	cudaMalloc((void**)&dev_a,tam);
	cudaMalloc((void**)&dev_b,tam_2);
	cudaMalloc((void**)&dev_c,tam_3);

	//copiando as variaveis da CPU para a GPU:
	cudaMemcpy(dev_a, &a, tam, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, &b, tam_2, cudaMemcpyHostToDevice);
	//chamada da função da gpu (kernel):
	// Número de blocos é igual a dimensão do vetor
	// divida pela dimensão do bloco. N/M

	// O tipo dim3 permite definir a quantidade de
	// blocos e threads por dimensão
	//dim3 numBlocos(2,2);// número de blocos é igual a 2x2 = 4
	//dim3 numThreads(2,2);// número de threads por bloco = 2x2 = 4

	dim3 numBlocos((M+T-1)/T, (D+T-1)/T);
	dim3 numThreads(T,T);
	soma_matriz<<<numBlocos,numThreads>>>(dev_a, dev_b, dev_c);

	//copiando o resultado da GPU para a CPU:
	cudaMemcpy(&c, dev_c, tam_3, cudaMemcpyDeviceToHost);

	//vizualizando o resultado:
	for(int i=0; i<M; i++){
		for(int j=0; j<D;j++)
			printf("%d ",c[i*D+j]);
		printf("\n");
	}

	printf("\n\n");

	//liberando a memoria na GPU:
	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);

	return 0;
}
