cmake_minimum_required(VERSION 3.16)
project(my_BA_pro)
set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)
# find_package(g2o REQUIRED)
find_package(Sophus REQUIRED)
find_package(Eigen3 REQUIRED)
set(SOURCES
    ${PROJECT_SOURCE_DIR}/src/my_bundle_adjustment.cc
    ${PROJECT_SOURCE_DIR}/src/BA_3d2d.cc
)
add_executable(my_BA ${SOURCES})

target_include_directories(my_BA PRIVATE /usr/include/glog ${PROJECT_SOURCE_DIR}/include ${EIGEN3_INCLUDE_DIR})
link_directories(/usr/lib/x86_64-linux-gnu)
target_link_libraries(my_BA glog)
target_link_libraries(my_BA ${OpenCV_LIBS} ${Sophus_LIBRARIES} ${EIGEN3_LIBRARIES})