//compile with < gcc fwd3.c -o Gary -O3 -Wno-unused-result -lm >


#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <time.h>

#define N 784 //number of neurons
#define T 3 //number of total layers
//#define ERROR 0.01

#define TRAINING_DATA_PATH "./Images/t10k-images.idx3-ubyte"
#define TRAINING_LABEL_PATH "./Images/t10k-labels-idx1-ubyte"

//#define TRAINING_DATA_PATH "./Images/train-images-idx3-ubyte"
//#define TRAINING_LABEL_PATH "./Images/train-labels-idx1-ubyte"

#define B_PARAMETER_PATH "./Parameters/B.skynet"
#define R_PARAMETER_PATH "./Parameters/R.skynet"


//prototypes for necessary functions
double sigma(double x);
void read3DArrayFromFile(double array[N][N][T+1], const char* filename);
void read2DArrayFromFile(double array[N][T+1], const char* filename);




//Variables
//-----------------------------------------------------------------//

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


uint32_t correctCount = 0;
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
	uint32_t magic, cols, rows, records;
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
	
	uint32_t Lmagic, Lrecords;
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
/*End Label Matrix
-------------------------------------------------------------*/

    //load Parameters from file
    printf("Loading B and R arrays form file");
	read2DArrayFromFile(B, B_PARAMETER_PATH);
	read3DArrayFromFile(R, R_PARAMETER_PATH);

    printf (" \n It's Showtiem!!!");
    
    for(uint64_t imageIndex = 0; imageIndex < records; imageIndex++){
		//fill X with samples, Y with labels
		mu = 0;
		for (int rowIndex = 0; rowIndex<rows; rowIndex++){
			for (int columnIndex=0; columnIndex < cols; columnIndex++){
				X[mu][0] = (double)images[imageIndex][rowIndex][columnIndex]/255.00;
				Y[mu] = (double)labels[imageIndex]/10.0;
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
		output = sum/(double)N;
		miss = fabs(((float)labels[imageIndex]/10) - output);
        epochMiss += miss;

        if (round(output*10) == labels[imageIndex]){
            correctCount++;
        }

        if (imageIndex %100 == 0){
            printf("\nFor sample %ld, guessed %1.5f, answer %d.  miss of %1.5f", imageIndex, output*10, labels[imageIndex], miss);
        }
        cycles++ ;
    }

    printf("\n---------------------------------------------------------------");
	printf("\nCorrect Answers: %d of %d\n",correctCount,records);
    printf("Network has an accuracy of %3.2f %% accuracy\n", 100.0 * ((float)correctCount/(float)records));

	free(images);
	fclose(trainingData);
	return 0;
}

////////////////////////////////////////////////////////////////////
double sigma(double input){
	return 1/(1+exp(-input));
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
