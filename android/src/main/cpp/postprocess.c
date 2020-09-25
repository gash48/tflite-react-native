/*
 * postprocess.c
 *
 *  Created on: Nov 11, 2018
 *      Author: Omer Naeem omernaeem@gmail.com
 */

#include "preprocess.h"
#include "postprocess.h"


void PostProcess(unsigned char* out8, float* in)
{
	float out[EMBEDDING_SIZE]={0};
	for(int i=0;i<EMBEDDING_SIZE;i++)
	{
		in[i] = in[i] - aPcaMeans[i];
	}

	for (int j=0;j<EMBEDDING_SIZE;j++)
	{
		for (int k=0;k<EMBEDDING_SIZE;k++)
		{
			out[j] += in[k]*aaPcaMat[j][k];
		}
	}

	for(int i=0;i<EMBEDDING_SIZE;i++)
	{
		if(out[i]>QUANTIZE_MAX_VAL)out[i]=QUANTIZE_MAX_VAL;
		if(out[i]<QUANTIZE_MIN_VAL)out[i]=QUANTIZE_MIN_VAL;

		out8[i] = (unsigned char)(((out[i]- QUANTIZE_MIN_VAL)*255)/QUANTIZE_RANGE);
	}


}
