message("--------------------------------------------------------------------")
message("CMake: vi_mapper")
cmake_minimum_required(VERSION 2.8.3)
project(vi_mapper)

#ds modules
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

#ds find packages
find_package(catkin REQUIRED COMPONENTS roscpp std_msgs genmsg sensor_msgs fps_mapper)
find_package(Boost REQUIRED COMPONENTS thread program_options)
find_package(Threads REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED )

#ds log found packages
message("using OpenCV version: ${OpenCV_VERSION}")

#ds declare the catkin packages
catkin_package(CATKIN_DEPENDS roscpp std_msgs genmsg sensor_msgs fps_mapper)

#ds add c++11 support, no optimization, all warnings
add_definitions(-march=native -O0 -g3 -fmessage-length=0 -MMD -MP -Wall -pedantic -std=c++11 -pthread -lm)

#ds include dirs
include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${Boost_INCLUDE_DIRS})
include_directories(${catkin_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)

#ds BINARIES
add_executable(message_dumper_node src/streaming/message_dumper_node.cpp)
#add_executable(stereo_detector_node src/analysis/stereo_detector_node.cpp)
#add_executable(stereo_detector_blocking_node src/analysis/stereo_detector_blocking_node.cpp)
add_executable(stereo_detector_alberto_node src/analysis/stereo_detector_alberto_node.cpp src/utility/CNaiveStereoDetector.cpp src/utility/CMiniVisionToolbox.cpp)

#ds link the binaries against catkin
target_link_libraries(message_dumper_node ${catkin_LIBRARIES} ${Boost_LIBRARIES})
#target_link_libraries(stereo_detector_node ${catkin_LIBRARIES} ${OpenCV_LIBRARIES})
#target_link_libraries(stereo_detector_blocking_node ${catkin_LIBRARIES} ${OpenCV_LIBRARIES})
target_link_libraries(stereo_detector_alberto_node ${catkin_LIBRARIES} ${OpenCV_LIBRARIES})

message("--------------------------------------------------------------------")
