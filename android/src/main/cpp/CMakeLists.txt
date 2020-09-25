cmake_minimum_required(VERSION 3.4.1)

file(GLOB SOURCES "*.h")
add_library( 
        vggsup

        kiss_fft.c
        kiss_fft.h
        kiss_fftr.c
        kiss_fftr.h
        _kiss_fft_guts.h
        pca.c
        preprocess.c
        preprocess.h
        postprocess.c
        preprocess.h
        )

add_library(native-lib

       
        SHARED

       
        native-lib.cpp
 )


find_library( 
        log-lib

       
        log)



target_link_libraries( 
        native-lib

       
        ${log-lib}
          vggsup)