message("--------------------------------------------------------------------")
message("CMake: vi_mapper")
cmake_minimum_required(VERSION 2.8.3)
project(vi_mapper)

#ds modules
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

message("module path: ${CMAKE_MODULE_PATH}")

#ds find packages
find_package(catkin REQUIRED COMPONENTS roscpp std_msgs genmsg sensor_msgs fps_mapper)
#find_package(Boost REQUIRED COMPONENTS thread program_options)
find_package(Threads REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED )
find_package(G2O REQUIRED)

#ds log found packages
message("using OpenCV version: ${OpenCV_VERSION}")

#ds declare the catkin packages
catkin_package(CATKIN_DEPENDS roscpp std_msgs genmsg sensor_msgs fps_mapper)

#ds add c++11 support, no optimization, all warnings
add_definitions(-O3 -Wall -pedantic -std=c++11 -pthread)

#ds include dirs
include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${OpenCV_INCLUDE_DIRS})
#include_directories(${Boost_INCLUDE_DIRS})
include_directories(${catkin_INCLUDE_DIRS})
include_directories(${G2O_INCLUDE_DIR})
#include_directories(${CSPARSE_INCLUDE_DIR})
#include_directories(${CHOLMOD_INCLUDE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)

#ds print all defined cmake variables
message("--------------------------------------------------------------------")
message("custom CMake variables:")
get_cmake_property(_variableNames VARIABLES)
foreach(_variableName ${_variableNames})
    if(_variableName MATCHES "G2O*")
        message("${_variableName}=${${_variableName}}")
    endif()
endforeach()

#ds BINARIES
#add_executable(message_dumper_node src/streaming/message_dumper_node.cpp)
#add_executable(stereo_detector_node src/analysis/stereo_detector_node.cpp)
#add_executable(stereo_detector_blocking_node src/analysis/stereo_detector_blocking_node.cpp)
add_executable(stereo_detector_alberto_node src/main/stereo_detector_alberto_node.cpp src/core/CEpipolarDetectorBRIEF.cpp src/utility/CMiniVisionToolbox.cpp)

#ds link the binaries against catkin
#target_link_libraries(message_dumper_node ${catkin_LIBRARIES} ${Boost_LIBRARIES})
#target_link_libraries(stereo_detector_node ${catkin_LIBRARIES} ${OpenCV_LIBRARIES})
#target_link_libraries(stereo_detector_blocking_node ${catkin_LIBRARIES} ${OpenCV_LIBRARIES})
target_link_libraries(stereo_detector_alberto_node 
                      ${catkin_LIBRARIES} 
                      ${OpenCV_LIBRARIES}     
                      ${G2O_OPENGL_HELPER_LIBRARY}
                      ${G2O_SOLVER_CSPARSE_EXTENSION}
                      ${G2O_TYPES_SLAM3D} 
                      ${G2O_CORE_LIBRARY} 
                      ${G2O_STUFF_LIBRARY}
                      ${G2O_TYPES_SBA})

message("--------------------------------------------------------------------")
