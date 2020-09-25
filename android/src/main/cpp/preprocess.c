/*
 * preprocess.c
 *
 *  Created on: Nov 11, 2018
 *      Author: Omer Naeem omernaeem@gmail.com
 */


#include <math.h>
#include <complex.h>
#include "kiss_fftr.h"
#include "preprocess.h"

float aPeriodicHann[WINDOW_LENGTH];

void GeneratePeriodicHann(void)
{
	for(int i=0;i<WINDOW_LENGTH;i++)
	{
		aPeriodicHann[i] = 0.5 - (0.5* cosf((2*PI*i)/WINDOW_LENGTH));
	}
	//0.5 - (0.5 * np.cos(2 * np.pi / window_length *
	//                             np.arange(window_length)))
}

float** ppFrames; //Dimension of [numFrames] [Window_Length]
float** ppFFT;
int CreateFrames(float* pData,long num_samples,int window_length,int hop_length)
{
	int numFrames =  1 + (int)(floorf(((float)num_samples - (float)window_length) / (float)hop_length)); //type cast to float can be disabled
	//printf("\r frames  %d   num samp  %ld      window len %d      hoplen %d",numFrames,num_samples,window_length,hop_length);

	ppFrames = (float**)malloc(numFrames*sizeof(float*));

	for(int i=0;i<numFrames;i++)
	{
		ppFrames[i] = (float*)malloc(WINDOW_LENGTH*sizeof(float));
		memcpy(ppFrames[i],&pData[i*160],WINDOW_LENGTH*sizeof(float));
	}

	return numFrames;

}

void WindowFrames(int numFrames)
{
	for(int i=0;i<numFrames;i++)
	{
		for(int j=0;j<WINDOW_LENGTH;j++)
		{
			ppFrames[i][j] = ppFrames[i][j] * aPeriodicHann[j];
		}

	}
}

void RealFFT(int fftlength,int numFrames)
{

	kiss_fftr_cfg cfg = kiss_fftr_alloc(fftlength, 0, 0, 0);
	kiss_fft_scalar *cx_in = (kiss_fft_scalar*)malloc((fftlength+1)*sizeof(kiss_fft_scalar));
	kiss_fft_cpx *cx_out = (kiss_fft_cpx*)malloc(((fftlength/2)+1)*sizeof(kiss_fft_cpx));
	float complex *pFFTOut = (float complex*)malloc(((fftlength/2)+1)*sizeof(float complex));

	for(int j=0;j<numFrames;j++)
	{
		for(int i=WINDOW_LENGTH;i<fftlength;i++)
		{
			cx_in[i]=0;
		}
		memcpy(cx_in,&ppFrames[j][0],WINDOW_LENGTH*sizeof(float));

		kiss_fftr(cfg, cx_in, cx_out);

		for(int k=0;k<(fftlength/2) +1;k++)
		{
			pFFTOut[k] = (cx_out[k].r) + (cx_out[k].i *I);
			ppFFT[j][k] = cabsf(pFFTOut[k]);
		}
	}


	free(cfg);
	free(cx_in);
	free(cx_out);
	free(pFFTOut);
}
float* pSpectrogramBinsHertz;
float* pSpectrogramBinsMel;
float* pBandEdgesMel;
float** ppMelWeightsMatrix; // dimension [numSpectrogramBins][NUM_MEL_BINS] numSpectrogramBins = fftlength/2 +1
float** ppMelSpectrogram; //dimension [numFrames][NUM_MEL_BINS]
void HertzToMel(float* out, float* in, int n)
{
	for(int i=0;i<n;i++)
	{
		out[i] = MEL_HIGH_FREQUENCY_Q * logf( 1.0 + (in[i] / MEL_BREAK_FREQUENCY_HERTZ));
	}
}
void Linspace(float* out, float start, float end, long n)
{
	float stepSize = (end - start ) / (n-1);
	for(long i =0;i<n;i++)
	{
		out[i] = start + (stepSize*i);
	}
}
void SpectrogramToMelMatrix(int numSpectrogramBins)
{
	float nyquistHertz = SAMPLE_RATE/2;
	pSpectrogramBinsHertz = (float*)malloc(sizeof(float)*numSpectrogramBins);
	Linspace(pSpectrogramBinsHertz,0,nyquistHertz,numSpectrogramBins);

	pSpectrogramBinsMel = (float*)malloc(sizeof(float)*numSpectrogramBins);
	HertzToMel(pSpectrogramBinsMel,pSpectrogramBinsHertz, numSpectrogramBins);

	pBandEdgesMel = (float*)malloc(sizeof(float)*(NUM_MEL_BINS+2));


	float lowerEdgeMel = MEL_HIGH_FREQUENCY_Q * logf( 1.0 + (MEL_MIN_HZ / MEL_BREAK_FREQUENCY_HERTZ));
	float upperEdgeMel = MEL_HIGH_FREQUENCY_Q * logf( 1.0 + (MEL_MAX_HZ / MEL_BREAK_FREQUENCY_HERTZ));

	Linspace(pBandEdgesMel,lowerEdgeMel,upperEdgeMel,NUM_MEL_BINS+2);

	ppMelWeightsMatrix = (float**)malloc(numSpectrogramBins*sizeof(float*));

	for(int i=0;i<numSpectrogramBins;i++)
	{
		ppMelWeightsMatrix[i] = (float*)malloc(NUM_MEL_BINS*sizeof(float));
	}

	float centerMel,lowerSlope,upperSlope;
	for (int i=0;i<NUM_MEL_BINS;i++)
	{
		lowerEdgeMel = pBandEdgesMel[i];
		centerMel = pBandEdgesMel[i+1];
		upperEdgeMel = pBandEdgesMel[i+2];

		for(int j=0;j<numSpectrogramBins;j++)
		{
			lowerSlope = ((pSpectrogramBinsMel[j] - lowerEdgeMel) /(centerMel - lowerEdgeMel));
			upperSlope = ((upperEdgeMel - pSpectrogramBinsMel[j]) /(upperEdgeMel - centerMel));
			ppMelWeightsMatrix[j][i] = fmaxf(0,fminf(lowerSlope,upperSlope));
		}
	}

	for (int i=0;i<NUM_MEL_BINS;i++)
	{
		ppMelWeightsMatrix[0][i]=0.0;
	}



}

void MatrixMultiply(float** out, int m1, int m2, float** mat1,
            int n1, int n2, float** mat2)
{
    int x, i, j;
    for (i = 0; i < m1; i++)
    {
        for (j = 0; j < n2; j++)
        {
            out[i][j] = 0;
            for (x = 0; x < m2; x++)
            {
                *(*(out + i) + j) += *(*(mat1 + i) + x) *
                                    *(*(mat2 + x) + j);
            }
        }
    }
}
int Preprocess(int16* pWavData, long wavLen)
{
	/* Convert to float and scale to -1 to +1 */
	float * pWavFData = (float*)malloc(wavLen*sizeof(float));
	if(pWavFData==NULL)
	{
		//printf("malloc failed");
	}
	for(long i =0; i< wavLen; i++)
	{
		pWavFData[i] = ((float)pWavData[i])/32768.0f;
	}

	 int window_length_samples = SAMPLE_RATE * STFT_WINDOW_LENGTH_SECONDS;
	 int hop_length_samples = SAMPLE_RATE * STFT_HOP_LENGTH_SECONDS;
	 int fft_length = powf(2,ceilf(logf(window_length_samples)/log(2.0)));


	 int numFrames = CreateFrames(pWavFData,wavLen,window_length_samples,hop_length_samples);
	 //printf("\r%f",ppFrames[329][397]);

	 WindowFrames(numFrames);
	 ppFFT = (float**)malloc(numFrames*sizeof(float*));

	 	for(int i=0;i<numFrames;i++)
	 	{
	 		ppFFT[i] = (float*)malloc(((fft_length/2) +1)*sizeof(float));
	 	}

	 RealFFT(fft_length,numFrames); //Produces Spectrogram in ppFFT


	 SpectrogramToMelMatrix((fft_length/2) + 1); //Produces MelMatrix in ppMelWeightsMatrix

	 //Multiply matrices ppFFT[numFrames][fftlen/2 +1] * ppMelWeightMatrix[fftlen/2 +1][NUM_MEL_BINS]
	 ppMelSpectrogram = (float**)malloc(numFrames*sizeof(float*));

	 for(int i=0;i<numFrames;i++)
	 {
		 ppMelSpectrogram[i] = (float*)malloc(NUM_MEL_BINS*sizeof(float));
	 }

	 MatrixMultiply(ppMelSpectrogram,numFrames,(fft_length/2) +1,ppFFT,(fft_length/2) +1,NUM_MEL_BINS,ppMelWeightsMatrix);





	 for (int i=0;i<numFrames;i++)
	 {
		for(int j=0;j<NUM_MEL_BINS;j++)
		{
			ppMelSpectrogram[i][j] = logf(ppMelSpectrogram[i][j] + LOG_OFFSET);
		}
	 }

	 for(int i=0;i<numFrames;i++)
	 {
		 free(ppFrames[i]);
	 }
	 free(ppFrames);

	 for(int i=0;i<((fft_length/2) +1);i++)
	 {
	 	free(ppMelWeightsMatrix[i]);
	 }
	 free(ppMelWeightsMatrix);

	 for(int i=0;i<numFrames;i++)
	 {
		free(ppFFT[i]);
	 }

	 free(ppFFT);
	 free(pSpectrogramBinsHertz);
	 free(pSpectrogramBinsMel);
	 free(pBandEdgesMel);
	 free(pWavFData);


	 float features_sample_rate = 1.0 / STFT_HOP_LENGTH_SECONDS;
	 int example_window_length = roundf(EXAMPLE_WINDOW_SECONDS * features_sample_rate);
	 int example_hop_length = roundf(EXAMPLE_HOP_SECONDS * features_sample_rate);


	 int numExamples = 1 + (int)(floorf(((float)numFrames - (float)example_window_length) / (float)example_hop_length));

//	 printf("\r\n new float[][]{");
//	 for(int a=0;a<96;a++)
//	 {
//		 printf("\n{");
//		 for(int b=0;b<64;b++)
//		 {
//			 if(b==63)
//				 printf("%ff",ppMelSpectrogram[a][b]);
//			 else
//			 printf("%ff, ",ppMelSpectrogram[a][b]);
//		 }
//		 printf("},");
//	 }
	 //In case of 1 sec audio output in ppMelSpectrogram[0:95][0:63]

	 return numFrames;


}
