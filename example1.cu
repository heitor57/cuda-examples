#include <stdio.h>
#include <stdlib.h>

#define N 32

//código device
__global__ void soma_vetor(int *a, int *b, int *c ){
	int indice = blockIdx.x;
	if(indice < N)
		c[indice] = a[indice] + b[indice];
}

//código host
int main(){
	int a[N],b[N],c[N];
	int* dev_a;
	int* dev_b;
	int* dev_c;

	int tam = N*sizeof(int);

	//inicializando as variaveis do host:
	for(int i=0; i < N; i++){
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
	soma_vetor<<<N,1>>>(dev_a, dev_b, dev_c);

	//copiando o resultado da GPU para a CPU:
	cudaMemcpy(&c, dev_c, tam, cudaMemcpyDeviceToHost);

	//vizualizando o resultado:
	for(int i=0; i<N; i++)
		printf("%d ",c[i]);
	printf("\n\n");

	//liberando a memoria na GPU:
	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);

	return 0;
}
