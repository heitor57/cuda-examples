#include <stdio.h>
#include <stdlib.h>

#define N 2 // Número de colunas das matrizes
#define M 2 // Número de linhas das matrizes
#define T 8 // Número de threads por bloco

//código device
// blockDim.x é a dimensão do bloco, ou seja,
// a quantidade de threads por bloco.
__global__ void soma_vetor(int *a, int *b, int *c ){
  int indice = blockIdx.x*blockDim.x + threadIdx.x;
  if(indice < N)
    c[indice] = a[indice] + b[indice];
}


__global__ void soma_matriz(int** a, int** b,int** c){
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  int j = blockIdx.y*blockDim.y + threadIdx.y;
  if(i<M && j<N)
    c[i][j] = a[i][j] + b[i][j];
}


//código host
int main(){
  int a[M][N],b[M][N],c[M][N];
  int** dev_a;
  int** dev_b;
  int** dev_c;

  int tam = M*sizeof(int*);
  int tam_inside = N*sizeof(int);
  //inicializando as variaveis do host:
  for(int i=0; i < N; i++){
    for(int j=0; j < M; j++){
      a[i][j] = i*j;
      b[i][j] = i*2*j;
    }
  }

  //alocando espaço para as variaveis da GPU:
  cudaMalloc((void***)&dev_a,tam);
  cudaMalloc((void***)&dev_b,tam);
  cudaMalloc((void***)&dev_c,tam);
  for(int i = 0; i < M; i++){
    cudaMalloc((void**)&(dev_a[i]),tam_inside);
    cudaMalloc((void**)&(dev_b[i]),tam_inside);
    cudaMalloc((void**)&(dev_c[i]),tam_inside);
  }
	  

  //copiando as variaveis da CPU para a GPU:

  for(int i = 0; i < M; i++){
    cudaMemcpy(dev_a[i], &(a[i]), tam_inside, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b[i], &(b[i]), tam_inside, cudaMemcpyHostToDevice);
  }
  //chamada da função da gpu (kernel):
  // Número de blocos é igual a dimensão do vetor
  // divida pela dimensão do bloco. N/M
  dim3 numBlocos((M+T-1)/T, (N+T-1)/T);
  dim3 numThreads(T,T);
  soma_matriz<<<numBlocos,numThreads>>>(dev_a, dev_b, dev_c);
			

  //copiando o resultado da GPU para a CPU:

  for(int i = 0; i < M; i++)
    cudaMemcpy(&(c[i]), dev_c[i], tam_inside, cudaMemcpyDeviceToHost);

  //vizualizando o resultado:
  for(int i=0; i<M; i++){
    for(int j=0; j<N; j++)
      printf("%d ",c[i][j]);
    printf("\n");
  }
  printf("\n\n");

  //liberando a memoria na GPU:
  for(int i = 0; i < M; i++){
    cudaFree(dev_a[i]);
    cudaFree(dev_b[i]);
    cudaFree(dev_c[i]);
  }

  cudaFree(dev_a);
  cudaFree(dev_b);
  cudaFree(dev_c);

  return 0;
}
