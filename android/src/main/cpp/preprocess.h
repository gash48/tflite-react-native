/*
 * preprocess.h
 *
 *  Created on: Nov 11, 2018
 *      Author: Omer Naeem omernaeem@gmail.com
 */

#ifndef PREPROCESS_H_
#define PREPROCESS_H_


/* Hyper parameters used in feature and example generation. */

#define NUM_FRAMES  96 // Frames in input mel-spectrogram patch.
#define NUM_BANDS  64  // Frequency bands in input mel-spectrogram patch.

#define SAMPLE_RATE  16000
#define STFT_WINDOW_LENGTH_SECONDS  0.025
#define STFT_HOP_LENGTH_SECONDS  0.010
#define NUM_MEL_BINS NUM_BANDS
#define MEL_MIN_HZ  125
#define MEL_MAX_HZ 7500
#define LOG_OFFSET 0.01  // Offset used for stabilized log of input mel-spectrogram.
#define EXAMPLE_WINDOW_SECONDS  0.96  // Each example contains 96 10ms frames
#define EXAMPLE_HOP_SECONDS 0.96     // with zero overlap.

#define MEL_BREAK_FREQUENCY_HERTZ  700.0f
#define MEL_HIGH_FREQUENCY_Q  1127.0f

#define WINDOW_LENGTH 400 //SAMPLE RATE * STFT_WINDOW_LENGTH_SECONDS
#define PI 3.14159265358979
/* ---------------------------------------------------------- */



typedef short int16;

void GeneratePeriodicHann(void);
void MatrixMultiply(float** out, int m1, int m2, float** mat1,
            int n1, int n2, float** mat2);
int Preprocess(int16* pWavData, long wavLen);
extern float** ppMelSpectrogram; // 2d array containing output of preprocessing

#endif /* PREPROCESS_H_ */
