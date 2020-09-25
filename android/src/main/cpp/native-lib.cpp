//
// Created by eyraa on 2019-03-13.
//

#include <jni.h>
#include <string>
#include <stdio.h>

extern "C"
{
#include "preprocess.h"
#include "postprocess.h"
}


extern "C" JNIEXPORT jstring JNICALL
Java_com_reactlibrary_TfliteReactNativeModule_stringFromJNI(
        JNIEnv *env,
        jobject /* this */) {
    std::string hello = "Hello from C++";
    return env->NewStringUTF(hello.c_str());
}

extern "C" JNIEXPORT jobjectArray JNICALL
Java_com_reactlibrary_TfliteReactNativeModule_preProcessFromJNI(
        JNIEnv *env,
        jobject /* this */,
        jshortArray pcmData,
        jint pcmSize) {
    char tmpMsg[100];
    short* pcmDataE = env->GetShortArrayElements(pcmData,0);
    GeneratePeriodicHann();
    short numFrames = Preprocess(pcmDataE,pcmSize);
    //sprintf(tmpMsg,"pcmData[0] %d  pcmData[1] %d Size %d",pcmDataE[0],pcmDataE[1],pcmSize);

    // Get the 2D float array we want to "Cast"
    float** primitive2DArray =ppMelSpectrogram;

    // Get the float array class

    jclass floatArrayClass = env->FindClass("[F");

    // Check if we properly got the float array class
    if (floatArrayClass == NULL)
    {
    // Ooops
        return NULL;
    }

    // Create the returnable 2D array
    jobjectArray myReturnable2DArray = env->NewObjectArray((jsize) numFrames, floatArrayClass, NULL);

// Go through the firs dimension and add the second dimension arrays
    for (unsigned int i = 0; i < numFrames; i++)
    {
        jfloatArray floatArray = env->NewFloatArray(64);
        env->SetFloatArrayRegion(floatArray, (jsize) 0, (jsize) 64, (jfloat*) primitive2DArray[i]);
        env->SetObjectArrayElement(myReturnable2DArray, (jsize) i, floatArray);
        env->DeleteLocalRef(floatArray);
    }

// Return a Java consumable 2D float array
    //return myReturnable2DArray;

    env->ReleaseShortArrayElements(pcmData, pcmDataE, 0);
    return myReturnable2DArray;
}

extern "C" JNIEXPORT jbyteArray JNICALL
Java_com_reactlibrary_TfliteReactNativeModule_postProcessFromJNI(
        JNIEnv *env,
        jobject /* this */,
        jfloatArray vggishOut) {

    float* vggishOutE = env->GetFloatArrayElements(vggishOut,0);

    unsigned char postProcessed[128];
    PostProcess(postProcessed,vggishOutE);

    jbyteArray jbArray  = env->NewByteArray(128);
    env->SetByteArrayRegion(jbArray, (jsize) 0, (jsize) 128, (jbyte *)postProcessed);


    env->ReleaseFloatArrayElements(vggishOut, vggishOutE, 0);
    return jbArray;
}

//extern "C" JNIEXPORT jshort JNICALL
//Java_omer_allears_Listening_numFramesFromJNI(
//        JNIEnv *env,
//        jobject /* this */,
//        jshortArray pcmData,
//        jint pcmSize) {
//    char tmpMsg[100];
//    short* pcmDataE = env->GetShortArrayElements(pcmData,0);
//    short numFrames = Preprocess(pcmDataE,pcmSize);
//    //sprintf(tmpMsg,"pcmData[0] %d  pcmData[1] %d Size %d",pcmDataE[0],pcmDataE[1],pcmSize);
//    env->ReleaseShortArrayElements(pcmData, pcmDataE, 0);
//    return numFrames;
//}


