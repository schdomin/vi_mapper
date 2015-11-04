message("--------------------------------------------------------------------")
message("CMake: vi_mapper")
cmake_minimum_required(VERSION 2.8.3)
project(vi_mapper)

#ds modules
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake_modules" ${CMAKE_MODULE_PATH})
set(CMAKE_BUILD_TYPE Debug)
#set(CMAKE_BUILD_TYPE Release)

message("module path: ${CMAKE_MODULE_PATH}")
message("build type: ${CMAKE_BUILD_TYPE}")

#ds find packages
find_package(catkin REQUIRED COMPONENTS roscpp std_msgs genmsg sensor_msgs fps_mapper)
#find_package(Threads REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED )
find_package(G2O REQUIRED)
find_package(Qt4 REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)
find_package(QGLViewer REQUIRED)
find_package(Cholmod REQUIRED)

#ds kdtree NN search
find_package(PkgConfig REQUIRED)
pkg_check_modules(FLANN flann)
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(FLANN DEFAULT_MSG
                                  FLANN_LIBRARY 
                                  FLANN_INCLUDE_DIR)

#ds log found packages
message("using OpenCV version: ${OpenCV_VERSION}")

#ds declare the catkin packages
catkin_package(CATKIN_DEPENDS roscpp std_msgs genmsg sensor_msgs fps_mapper)

#ds add c++11 support, no optimization, all warnings
add_definitions(-O3 -march=native -Wall -pedantic -std=c++11 -pthread)

#ds include dirs
include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${catkin_INCLUDE_DIRS})
include_directories(${G2O_INCLUDE_DIR})
#include_directories(${CSPARSE_INCLUDE_DIR})
include_directories(${CHOLMOD_INCLUDE_DIR})
include_directories(${QT_INCLUDES})
include_directories(${OPENGL_INCLUDE_DIRS})
include_directories(${GLUT_INCLUDE_DIRS})
include_directories(${QGLVIEWER_INCLUDE_DIR})
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
set(SOURCEFILES
src/configuration/CConfigurationCamera.cpp
src/configuration/CConfigurationCameraKITTI.cpp
src/types/CLandmark.cpp
src/types/CKeyFrame.cpp
src/utility/CLogger.cpp
src/utility/CIMUInterpolator.cpp
src/vision/CMiniVisionToolbox.cpp
src/core/CTriangulator.cpp
src/core/CFundamentalMatcher.cpp
src/optimization/Cg2oOptimizer.cpp
src/optimization/closure_buffer.cpp
src/optimization/closure_checker.cpp
src/gui/CViewerScene.cpp)

set(SOURCEFILES_MOCKED
src/configuration/CConfigurationCamera.cpp
src/types/CLandmark.cpp
src/types/CKeyFrame.cpp
src/utility/CLogger.cpp
src/vision/CMiniVisionToolbox.cpp
src/core/CMockedMatcherEpipolar.cpp
src/optimization/CBridgeG2O.cpp)

set(LIBRARIES_GENERAL
${catkin_LIBRARIES}
${OpenCV_LIBRARIES}
${G2O_CORE_LIBRARY}
${G2O_TYPES_SLAM3D}
${G2O_SOLVER_CHOLMOD}
${CHOLMOD_LIBRARY}
${G2O_STUFF_LIBRARY}
${FLANN_LIBRARIES}
-fopenmp
-pthread)

set(LIBRARIES_VISUALIZATION
${OPENGL_LIBRARIES}
${GLUT_LIBRARY}
${QGLVIEWER_LIBRARY}
${QT_QTXML_LIBRARY}
${QT_QTOPENGL_LIBRARY}
${QT_QTGUI_LIBRARY}
${QT_QTCORE_LIBRARY}
GLU)

set(LIBRARIES_MOCKED
${catkin_LIBRARIES} 
${OpenCV_LIBRARIES}     
${G2O_OPENGL_HELPER_LIBRARY}
${G2O_SOLVER_CSPARSE_EXTENSION}
${G2O_TYPES_SLAM3D} 
${G2O_CORE_LIBRARY} 
${G2O_STUFF_LIBRARY}
${G2O_TYPES_SBA})

add_executable(message_dumper_node src/streaming/message_dumper_node.cpp)
add_executable(test_cloud_matching src/runnable/test_cloud_matching.cpp src/vision/CMiniVisionToolbox.cpp src/gui/CViewerCloud.cpp src/types/CKeyFrame.cpp)
#add_executable(stereo_detector_node src/analysis/stereo_detector_node.cpp)
#add_executable(stereo_detector_blocking_node src/analysis/stereo_detector_blocking_node.cpp)
#add_executable(stereo_detector_alberto_node src/main/stereo_detector_alberto_node.cpp src/core/CEpipolarDetectorBRIEF.cpp src/utility/CMiniVisionToolbox.cpp)
#add_executable(stereo_detector_alberto_node src/runnable/stereo_detector_alberto_node.cpp src/core/CDetectorTestStereoDepth.cpp ${CUSTOMVARIABLE_SOURCEFILES})

#add_executable(stereo_detector_alberto_node src/runnable/stereo_detector_alberto_node.cpp src/core/CTrackerStereo.cpp ${SOURCEFILES})
#add_executable(stereo_detector_mocked src/runnable/stereo_detector_mocked.cpp src/core/CMockedTrackerStereo.cpp ${SOURCEFILES_MOCKED})
#add_executable(test_depth src/runnable/test_depth.cpp ${SOURCEFILES})
#add_executable(stereo_single src/runnable/stereo_single.cpp src/core/CTrackerStereoMotionModel.cpp ${SOURCEFILES})
#add_executable(stereo_parallel src/runnable/stereo_parallel.cpp src/core/CTrackerStereoMotionModel.cpp ${SOURCEFILES})
add_executable(stereo_fps src/runnable/stereo_fps.cpp src/core/CTrackerStereoMotionModel.cpp ${SOURCEFILES})
add_executable(stereo_fps_kitti src/runnable/stereo_fps_kitti.cpp src/core/CTrackerStereoMotionModelKITTI.cpp ${SOURCEFILES})
add_executable(republisher_node src/runnable/republisher_node.cpp)
add_executable(republisher_kitti src/runnable/republisher_kitti.cpp)
add_executable(create_trajectory_kitti src/streaming/create_trajectory_kitti.cpp src/configuration/CConfigurationCameraKITTI.cpp)
add_executable(evaluate_trajectory src/runnable/evaluate_trajectory.cpp src/utility/CLogger.cpp)

#ds link the binaries against catkin
target_link_libraries(message_dumper_node ${catkin_LIBRARIES})
target_link_libraries(test_cloud_matching ${OpenCV_LIBRARIES} ${LIBRARIES_VISUALIZATION})
#target_link_libraries(stereo_detector_node ${catkin_LIBRARIES} ${OpenCV_LIBRARIES})
#target_link_libraries(stereo_detector_blocking_node ${catkin_LIBRARIES} ${OpenCV_LIBRARIES})
#target_link_libraries(stereo_detector_alberto_node ${LIBRARIES_GENERAL})
#target_link_libraries(stereo_detector_mocked ${LIBRARIES_MOCKED})
#target_link_libraries(test_depth ${catkin_LIBRARIES} ${OpenCV_LIBRARIES})
#target_link_libraries(stereo_single ${LIBRARIES_GENERAL})
#target_link_libraries(stereo_parallel ${LIBRARIES_GENERAL})
target_link_libraries(stereo_fps ${LIBRARIES_GENERAL} ${LIBRARIES_VISUALIZATION})
target_link_libraries(stereo_fps_kitti ${LIBRARIES_GENERAL} ${LIBRARIES_VISUALIZATION})
target_link_libraries(republisher_node ${catkin_LIBRARIES} ${OpenCV_LIBRARIES})
target_link_libraries(republisher_kitti ${catkin_LIBRARIES} ${OpenCV_LIBRARIES})
target_link_libraries(create_trajectory_kitti ${catkin_LIBRARIES} ${LIBRARIES_GENERAL})

message("--------------------------------------------------------------------")
