cmake_minimum_required(VERSION 3.5)
project(cv_pipeline)

find_package( OpenCV REQUIRED )

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(SOURCE_FILES main.cpp json11.cpp json11.hpp cvpipeline.cpp cvpipeline.h modules/fgbg.cpp modules/fgbg.h modules/fgsegm.cpp modules/fgsegm.h modules/object_tracker.cpp modules/object_tracker.h modules/intrusion_area.cpp modules/intrusion_area.h modules/image_part.cpp modules/image_part.h modules/object_extracting.cpp modules/object_extracting.h)

add_executable(cv_pipeline ${SOURCE_FILES})
target_link_libraries( cv_pipeline ${OpenCV_LIBS} )
