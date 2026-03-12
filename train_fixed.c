//compile with < gcc fwd3.c -o Gary -O3 -Wno-unused-result -lm >


#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <time.h>

#define N 784 //number of neurons
#define T 3 //number of total layers
//#define ERROR 0.01

#define TRAINING_DATA_PATH "./Images/train-images-idx3-ubyte"
#define TRAINING_LABEL_PATH "./Images/train-labels-idx1-ubyte"

#define B_PARAMETER_PATH "./Parameters/B.neural"
#define R_PARAMETER_PATH "./Parameters/R.neural"


//prototypes for necessary functions
double sigma(double x);
void write3DArrayToFile(double array[N][N][T+1], const char* filename);
void write2DArrayToFile(double array[N][T+1], const char* filename);
void read3DArrayFromFile(double array[N][N][T+1], const char* filename);
void read2DArrayFromFile(double array[N][T+1], const char* filename);
void print3DArray(double array[N][N][T+1]);
void print2DArray(double array[N][T+1]);
void seed3D(double array[N][N][T+1]);
void seed2D(double array[N][T+1]);



//Variables
//-----------------------------------------------------------------//
float const ERROR = 0.001; //end when error reaches this
float EPS = 0.001;				//learning rate

int oldDogNewTricks = 1;
float output=0;
float miss=0;
float epochMiss=0;
float epochMissAverage=0;

double Y[N];				//Label Matrix (expected answer)
double R[N][N][T+1];		//Transfer Matrix
double B[N][T+1];			//Bias Function
double X[N][T+1];				//Resultant Matrix
double Z[N][T+1];				//input for the sigma function
double dB[N][T+1];


double cost = ERROR + 1.0;

uint64_t mu, nu, t=0, cycles = 0, epochs=0;




int main (void) {
	srand(time(NULL));
/*Data intake
----------------------------------------------------------------------
Image Matrix*/

	//Open image file
	FILE * trainingData = fopen(TRAINING_DATA_PATH, "rb");
	if(trainingData==NULL){
		printf("unable to open image file!\n");
		return 1;
	}
	
	//read header data
	uint32_t magic, cols, rows, records, Lmagic, Lrecords;
	fread(&magic, 4, 1, trainingData);
	fread(&records, 4, 1, trainingData);
	fread(&cols, 4, 1, trainingData);
	fread(&rows, 4, 1, trainingData);

	//big indian little indian swapping voodoo magic bullshit
	magic = __builtin_bswap32(magic);
	records = __builtin_bswap32(records);
	cols = __builtin_bswap32(cols);
	rows = __builtin_bswap32(rows);

	printf("Magic: %x\n", magic);
	printf("There are %d images.\n", records);
	printf("There are %d rows. \n", rows);
	printf("There are %d columns.\n", cols);
	
	//allocate memory for images matrix
	uint8_t (*images)[rows][cols] = malloc(records * sizeof(*images));

	//populate images matrix with file data
	for (uint32_t imageIndex = 0; imageIndex < records; imageIndex++){
		for (uint32_t rowIndex = 0; rowIndex < rows; rowIndex++){
			for (uint32_t columnIndex=0; columnIndex < cols; columnIndex++){
				fread(&images[imageIndex][rowIndex][columnIndex],1,1,trainingData);
			}
		}
	}
//End Image Matrix

/* Label Matrix
-----------------------------------------------------------*/	
	//open label file
	FILE * labelData = fopen(TRAINING_LABEL_PATH, "rb");
	if(labelData==NULL){
		printf("unable to open label file!\n");
		return 1;
	}
	
	//read header data
	fread(&Lmagic, 4, 1, labelData);
	fread(&Lrecords, 4, 1, labelData);

	//big indian little indian swapping voodoo magic bullshit
	Lmagic = __builtin_bswap32(Lmagic);
	Lrecords = __builtin_bswap32(Lrecords);

	printf("LMagic: %x\n", Lmagic);
	printf("There are %d labels.\n", Lrecords);
	
	//allocate memory for labels matrix
	uint8_t  labels[Lrecords];

	//populate label matrix with file data
	for (uint32_t labelIndex = 0; labelIndex < Lrecords; labelIndex++){
		fread(&labels[labelIndex],1,1,labelData);
	}
	//debugging loop
/*	for (int i = 0; i < 4; i++){
		printf()
	}
	*/
/*End Label Matrix
-------------------------------------------------------------*/

	//Choose to use random B and R matrices, or continue training from a file
	if (oldDogNewTricks == 1){
		//load B and R from file
		printf("Loading B and R arrays form file");
		read2DArrayFromFile(B, B_PARAMETER_PATH);
		read3DArrayFromFile(R, R_PARAMETER_PATH);
	}
	else {
		printf("Randomly Seeding B and R (like that one farmer from the parable.)");
		seed3D(R);
		seed2D(B);
	}
//end of data intake
	printf (" \n Welcome to Boot Camp!");
	epochMissAverage = ERROR + 1;

	while (epochMissAverage > ERROR){
		epochMiss=0;
		//loads current sample into the zeroth layer, and the label into the desired output
		for(int imageIndex= 0; imageIndex < records; imageIndex++){
			mu=0;
			for(int rowIndex=0; rowIndex < rows; rowIndex++){
				for(int columnIndex=0; columnIndex < cols; columnIndex++){
					X[mu][0] = (double)images[imageIndex][rowIndex][columnIndex]/255.00;
					Y[mu] = (double)labels[imageIndex]/10.0;
					mu++;
				}
			}

		//Forward propogate
			for ( t = 1; t <= T; t++){
				for ( mu = 0; mu < N; mu++){
				
					Z[mu][t] = B[mu][t];

					for (nu = 0; nu < N; nu++){
        		  		 Z[mu][t] = Z[mu][t] + R[mu][nu][t]*X[nu][t-1];
					}

					X[mu][t] = sigma(Z[mu][t]);
				//fprintf(stderr, "X[%d][%d] = %lf\n", mu, t+1, X[mu][t+1]);
				}
			}
		//end Forward Propogate

		//Back Propogate
			for ( mu=0; mu < N; mu++){
				dB[mu][T] = -EPS*(X[mu][T]-Y[mu]) * X[mu][T] * (1-X[mu][T]);
				B[mu][T] = B[mu][T] + dB[mu][T];
				for ( nu=0; nu<N; nu++){
					R[mu][nu][T] = R[mu][nu][T] + dB[mu][T]*X[nu][T-1];
				}
			}

			//k = 1...T-1 Layer
			for (int k=1; k<T; k++){
				for (mu=0; mu<N; mu++){
					dB[mu][T-k] = 0;
					for(int a=0; a<N; a++){
						dB[mu][T-k] = dB[mu][T-k] + dB[a][T-k+1] * R[a][mu][T-k+1]*X[mu][T-k]*(1-X[mu][T-k]);
					}
					B[mu][T-k] = B[mu][T-k] + dB[mu][T-k];
					for (nu = 0; nu < N; nu++){
						R[mu][nu][T-k] = R[mu][nu][T-k] + dB[mu][T-k] * X[nu][T-k-1];
					}

				}
			}

			float sum = 0;
			for(int mu = 0; mu<N; mu++){
				sum += X[mu][T];			//sum of all the neurons
			}
			output = sum/(double)N;
			miss = fabs(((double)labels[imageIndex]/10.0)-output);
			epochMiss += miss;

			if(imageIndex % 1000 == 0){
				printf("\nSample %d -> guessed %1.5f, answer %d, miss of %1.5f", imageIndex, output*10, labels[imageIndex], miss*10);
			}
		
			//calculate cost function
			cost = 0;
			for(mu=0; mu<N; mu++){
				cost = cost + (X[mu][T]-Y[mu])*(X[mu][T]-Y[mu]);
			}
			cost = 0.5*cost;
			cycles++;
		}	//end of training data loop

		//calculate average error from the last epoch
		epochMissAverage = (epochMiss/(float)records);
		printf("\n\n>>>>>>>>>>>>>>>>>Epoch %ld MISS AVERAGE: %f ", epochs, epochMissAverage);
		epochs++;

		//periodically saves B,R to file to prevent data lose from premature termination of the program
		printf("Saving B,R tmp parameters to file");
		write2DArrayToFile(B, "BTemp.fixed");
		write3DArrayToFile(R, "RTemp.fixed");
		
	}		//End Training

	//Save Parameters to file
	write2DArrayToFile(B, "B.fixed");
	write3DArrayToFile(R, "R.fixed");
	
	//Testing
	records=50100;

	for(int imageIndex = 50000; imageIndex < records; imageIndex++){
		//fill X with samples, Y with labels
		mu = 0;
		for (int rowIndex = 0; rowIndex<rows; rowIndex++){
			for (int columnIndex=0; columnIndex < cols; columnIndex++){
				X[mu][0] = (double)images[imageIndex][rowIndex][columnIndex];
				Y[mu] = (double)labels[imageIndex];
				mu++;
			}
		}

		//forward propogate
		for(t=1; t<=T; t++){
			for(mu=0; mu<N; mu++){
				Z[mu][t] = B[mu][t];
				for(nu=0; nu<N; nu++){
					Z[mu][t] = Z[mu][t] + R[mu][nu][t]*X[nu][t-1];
				}
				X[mu][t] = sigma(Z[mu][t]);
			}
		}

		float sum=0;
		for(int mu=0; mu<N; mu++){
			sum += X[mu][T];
		}
		output = sum/784.00;
		miss = fabs(((float)labels[imageIndex]/10) - output);
		printf("/nFor sample %d, the guess was %f, answer %d.  An error of %f", imageIndex, output*10, labels[imageIndex], miss);
	}


//End Testing


	//free memory from data matrices, then close the files
	free(images);
	fclose(trainingData);
	
	return 0;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////

double sigma(double input){
	return 1/(1+exp(-input));
}

//Writes 3D array of doubles to a file
void write3DArrayToFile(double array[N][N][T+1], const char* filename){
	FILE* file = fopen(filename, "wb");
	if (file == NULL){
		fprintf(stderr,"Error opening file!\n");
		return;
	}
	fwrite(array, sizeof(double), N*N*(T+1), file);
	fclose (file);
}

//Writes a 2D array of doubles to a file
void write2DArrayToFile(double array[N][T+1], const char* filename){
	FILE* file = fopen(filename, "wb");
	if (file == NULL){
		fprintf(stderr,"Error opening file!\n");
		return;
	}
	fwrite(array, sizeof(double), N * (T+1), file);
	fclose(file);
}

//Reads a 3D array of doubles from a file
void read3DArrayFromFile(double array[N][N][T+1], const char* filename){
	FILE* file = fopen(filename, "rb");
	if (file == NULL){
		printf("Error opening file!\n");
		return;
	}
	fread(array, sizeof(double), N*N*(T+1),file);
	fclose(file);
}

//Reads a 2D array of doubles from a file
void read2DArrayFromFile(double array[N][T+1], const char* filename){
	FILE* file = fopen(filename, "rb");
	if (file == NULL){
		printf("Error opening file!\n");
		return;
	}
	fread(array, sizeof(double), N*(T+1),file);
	fclose(file);
}

//Prints 3D Array to screen
void print3DArray(double array[N][N][T+1]){
	for (uint32_t x = 0; x<N; x++){
		printf("Layer %u:\n", x);
		for(uint32_t y=0; y<N; y++){
			for(uint32_t z=0; z<T+1; z++){
				printf("%0.03f", array[x][y][z]);
			}
			printf("\n");
		}
		printf("\n");
	}
}

void print2DArray(double array[N][T+1]){
	for(uint32_t x=0; x<N; x++){
		for(uint32_t y=0; y<T+1; y++){
			printf("%0.3f ", array[x][y]);
		}
		printf("\n");
	}
}

void seed3D(double array[N][N][T+1]){
	for (uint32_t x=0; x<N; x++){
		for (uint32_t y=0; y < N; y++){
			for (uint32_t z=0; z < T+1; z++){
				array[x][y][z] = ((double) rand() / RAND_MAX) - 0.5;
			}
		}
	}
}

void seed2D(double array[N][T+1]){
	for(int y=0; y<N; y++){
		for(int x=0; x<T; x++){
			array[y][x] = ((double)rand() / RAND_MAX) - 0.5;
		}
	}
}

