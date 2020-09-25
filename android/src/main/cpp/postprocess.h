/*
 * postprocess.h
 *
 *  Created on: Nov 11, 2018
 *      Author: Omer Naeem omernaeem@gmail.com
 */

#ifndef POSTPROCESS_H_
#define POSTPROCESS_H_

extern float aPcaMeans[128];
extern float aaPcaMat[128][128];

#define QUANTIZE_MIN_VAL -2.0f
#define QUANTIZE_MAX_VAL 2.0f
#define QUANTIZE_RANGE 4.0f

#define EMBEDDING_SIZE 128

void PostProcess(unsigned char* out8, float* in);

#endif /* POSTPROCESS_H_ */
