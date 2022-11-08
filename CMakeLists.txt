cmake_minimum_required(VERSION 3.2)
project(pysot)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fopenmp ")
set(CMAKE_CXX_STANDARD 14)
#set(CMAKE_CXX_STANDARD 11)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
set(Torch_DIR /home/jjyjb/softwares/libtorchcu113/libtorch/share/cmake/Torch)
find_package(Torch REQUIRED)

include_directories(
        ${EIGEN3_INCLUDE_DIR}
        ${PROJECT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/SiamRPNTracker
        ${OpenCV_INCLUDE_DIRS}

)



add_executable(siamrpntracker SiamRPNTracker/main.cpp
        SiamRPNTracker/SiamRPNTracker.cpp SiamRPNTracker/SiamRPNTracker.h
        SiamRPNTracker/Hanning.cpp SiamRPNTracker/Hanning.h SiamRPNTracker/Anchors.cpp SiamRPNTracker/Anchors.h
        SiamRPNTracker/base_tracker.cpp SiamRPNTracker/base_tracker.h
        )

target_link_libraries(siamrpntracker
        ${OpenCV_LIBS}
        ${TORCH_LIBRARIES}
        )


