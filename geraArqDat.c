#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>

int main(int argc, char *argv[]){
	unsigned long int dimA_M, dimA_N, dimB_M, dimB_N, tamanhoA, tamanhoB, tamanhoC;
	int i;
	float *matrizA, *matrizB, *matrizC;
	char *eptr = NULL;
	
	if (argc != 5) {
		printf("Usage: %s <DimA_M> <DimA_N> <DimB_M> <DimB_N>\n", argv[0]);
		return 0;
	}
	
	dimA_M = strtol(argv[1], &eptr, 10);
	dimA_N = strtol(argv[2], &eptr, 10);
	dimB_M = strtol(argv[3], &eptr, 10);
	dimB_N = strtol(argv[4], &eptr, 10);

	tamanhoA = dimA_M*dimA_N;
	tamanhoB = dimB_M*dimB_N;
	tamanhoC = dimA_M*dimB_N;

	matrizA = aligned_alloc(32, tamanhoA * sizeof(float));
	matrizB = aligned_alloc(32, tamanhoB * sizeof(float));
	matrizC = aligned_alloc(32, tamanhoC * sizeof(float));

	for (i=0; i < tamanhoA; i++)
		matrizA[i] = 2.0 ;

	for (i=0; i < tamanhoB; i++)
		matrizB[i] = 5.0 ;

	for (i=0; i < tamanhoC; i++)
		matrizC[i] = 0.0 ;

	FILE* fp1= fopen("floats_256_2.0f.dat", "wb");
	FILE* fp2= fopen("floats_256_5.0f.dat", "wb");
	FILE* fp3= fopen("floats_256_0.0f.dat", "wb");

	for ( i = 0; i < tamanhoA; i += 8, matrizA += 8) {
		fwrite(matrizA,sizeof(float),8,fp1);
	}

	for ( i = 0; i < tamanhoB; i += 8, matrizB += 8) {
		fwrite(matrizB,sizeof(float),8,fp2);
	}

	for ( i = 0; i < tamanhoC; i += 8, matrizC += 8) {
		fwrite(matrizC,sizeof(float),8,fp3);
	}
	
	fclose(fp1);
	fclose(fp2);
	fclose(fp3);
}
