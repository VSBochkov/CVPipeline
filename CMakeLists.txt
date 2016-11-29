cmake_minimum_required(VERSION 3.5)
project(cv_pipeline)

find_package( OpenCV REQUIRED )

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(SOURCE_FILES
    main.cpp
    json11.cpp
    json11.hpp
    cvpipeline.cpp
    cvpipeline.h
    modules/firebbox.cpp
    modules/firebbox.h
    modules/firevalidation.cpp
    modules/firevalidation.h
    modules/fireweight.cpp
    modules/fireweight.h
    modules/flamesrcbbox.cpp
    modules/flamesrcbbox.h
    modules/rfiremaskingmodel.cpp
    modules/rfiremaskingmodel.h
)

add_executable(cv_pipeline ${SOURCE_FILES})
target_link_libraries( cv_pipeline ${OpenCV_LIBS} )
