#ifndef CLOGGER_H
#define CLOGGER_H

#include <chrono>
#include "types/CLandmark.h"

class CLogger
{

public:

    static void openBox( )
    {
        std::printf( "|''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''|\n" );
    }

    static void closeBox( )
    {
        std::printf( "|..........................................................................................................|\n" );
    }

    static const std::string getTimestamp( )
    {
        //ds current time
        const std::time_t tmCurrent = std::time( NULL );

        //ds compute stamp and return
        char chBufferTimestamp[100];
        std::strftime( chBufferTimestamp, sizeof( chBufferTimestamp ), "%Y-%m-%d-%H%M%S", std::localtime( &tmCurrent ) );
        return chBufferTimestamp;
    }

    static const double getTimeSeconds( )
    {
        return std::chrono::system_clock::now( ).time_since_epoch( ).count( )/1e9;
    }

//ds logging
public:

    static struct CLogLandmarkCreation
    {
        static std::FILE* m_pFile;

        static void open( )
        {
            m_pFile = std::fopen( "/home/dominik/workspace_catkin/src/vi_mapper/logs/landmarks_creation.txt", "w" );
            assert( 0 != m_pFile );
            std::fprintf( m_pFile, "ID_FRAME | ID_LANDMARK |      X      Y      Z :  DEPTH | U_LEFT V_LEFT | U_RIGHT V_RIGHT | KEYPOINT_SIZE\n" );
        }
        static void addEntry( const uint64_t& p_uFrame, const CLandmark* p_pLandmark, const double& p_dDepthMeters, const cv::Point2d& p_ptLandmarkLEFT, const cv::Point2d& p_ptLandmarkRIGHT )
        {
            std::fprintf( m_pFile, "    %04lu |      %06lu | %6.2f %6.2f %6.2f : %6.2f | %6.2f %6.2f |  %6.2f  %6.2f |        %6.2f\n",
                p_uFrame,
                p_pLandmark->uID,
                p_pLandmark->vecPointXYZOptimized.x( ),
                p_pLandmark->vecPointXYZOptimized.y( ),
                p_pLandmark->vecPointXYZOptimized.z( ),
                p_dDepthMeters,
                p_ptLandmarkLEFT.x,
                p_ptLandmarkLEFT.y,
                p_ptLandmarkRIGHT.x,
                p_ptLandmarkRIGHT.y,
                p_pLandmark->dKeyPointSize );
        }
        static void close( ){ if( 0 != m_pFile ){ std::fclose( m_pFile ); } }

    } CLandmarkCreation;

    static struct CLogLandmarkCreationMocked
    {
        static std::FILE* m_pFile;

        static void open( )
        {
            m_pFile = std::fopen( "/home/dominik/workspace_catkin/src/vi_mapper/logs/landmarks_creation_mocked.txt", "w" );
            assert( 0 != m_pFile );
            std::fprintf( m_pFile, "FRAME | ID_LANDMARK |      X      Y      Z :  DEPTH | U_LEFT V_LEFT : NOISE_U NOISE_V | U_RIGHT V_RIGHT : NOISE_U NOISE_V | KEYPOINT_SIZE\n" );

        }
        static void addEntry( const uint64_t& p_uFrame, const CLandmark* p_pLandmark, const double& p_dDepthMeters, const cv::Point2d& p_ptLandmarkLEFT, const cv::Point2d& p_ptLandmarkRIGHT, const CMockedDetection& p_cMockedDetection )
        {
            std::fprintf( m_pFile, " %04lu |      %06lu | %6.2f %6.2f %6.2f : %6.2f | %6.2f %6.2f :   %5.2f   %5.2f |  %6.2f  %6.2f :   %5.2f   %5.2f |        %6.2f\n", p_uFrame,
                p_pLandmark->uID,
                p_pLandmark->vecPointXYZOptimized.x( ),
                p_pLandmark->vecPointXYZOptimized.y( ),
                p_pLandmark->vecPointXYZOptimized.z( ),
                p_dDepthMeters,
                p_ptLandmarkLEFT.x,
                p_ptLandmarkLEFT.y,
                p_cMockedDetection.dNoiseULEFT,
                p_cMockedDetection.dNoiseV,
                p_ptLandmarkRIGHT.x,
                p_ptLandmarkRIGHT.y,
                p_cMockedDetection.dNoiseURIGHT,
                p_cMockedDetection.dNoiseV,
                p_pLandmark->dKeyPointSize );
        }
        static void close( ){ if( 0 != m_pFile ){ std::fclose( m_pFile ); } }

    } CLogLandmarkCreationMocked;

    static struct CLogTrajectory
    {
        static std::FILE* m_pFile;

        static void open( )
        {
            m_pFile = std::fopen( "/home/dominik/workspace_catkin/src/vi_mapper/logs/trajectory.txt", "w" );
            assert( 0 != m_pFile );
            std::fprintf( m_pFile, "ID_FRAME |      X      Y      Z | QUAT_X QUAT_Y QUAT_Z QUAT_W\n" );
        }
        static void addEntry( const uint64_t& p_uFrame, const CPoint3DInWorldFrame& p_vecPosition, const Eigen::Quaterniond& p_vecQuaternion )
        {
            std::fprintf( m_pFile, "    %04lu | %6.2f %6.2f %6.2f | %6.3f %6.3f %6.3f %6.3f\n",
                p_uFrame,
                p_vecPosition.x( ),
                p_vecPosition.y( ),
                p_vecPosition.z( ),
                p_vecQuaternion.x( ),
                p_vecQuaternion.y( ),
                p_vecQuaternion.z( ),
                p_vecQuaternion.w( ) );
        }
        static void close( ){ if( 0 != m_pFile ){ std::fclose( m_pFile ); } }

    } CLogTrajectory;

    static struct CLogLandmarkFinal
    {
        static std::FILE* m_pFile;

        static void open( )
        {
            m_pFile = std::fopen( "/home/dominik/workspace_catkin/src/vi_mapper/logs/landmarks_final.txt", "w" );
            assert( 0 != m_pFile );
            std::fprintf( m_pFile, "ID_LANDMARK | X_INITIAL Y_INITIAL Z_INITIAL | X_FINAL Y_FINAL Z_FINAL | DELTA_X DELTA_Y DELTA_Z DELTA_TOTAL | MEASUREMENTS | OPTIMIZATIONS | MEAN_X MEAN_Y MEAN_Z | KEYFRAMES\n" );
        }
        static void addEntry( const CLandmark* p_pLandmark )
        {
            //ds compute errors
            const double dErrorX = std::fabs( ( p_pLandmark->vecPointXYZOptimized.x( )-p_pLandmark->vecPointXYZInitial.x( ) )/( 1.0+std::fabs( p_pLandmark->vecPointXYZInitial.x( ) ) ) );
            const double dErrorY = std::fabs( ( p_pLandmark->vecPointXYZOptimized.y( )-p_pLandmark->vecPointXYZInitial.y( ) )/( 1.0+std::fabs( p_pLandmark->vecPointXYZInitial.y( ) ) ) );
            const double dErrorZ = std::fabs( ( p_pLandmark->vecPointXYZOptimized.z( )-p_pLandmark->vecPointXYZInitial.z( ) )/( 1.0+std::fabs( p_pLandmark->vecPointXYZInitial.z( ) ) ) );
            const double dErrorTotal = dErrorX + dErrorY + dErrorZ;

            //ds write final state to file before deleting
            std::fprintf( m_pFile, "     %06lu |    %6.2f    %6.2f    %6.2f |  %6.2f  %6.2f  %6.2f |   %5.2f   %5.2f   %5.2f       %5.2f |       %06lu |        %06u | %6.2f %6.2f %6.2f |        %02u\n",
                p_pLandmark->uID,
                p_pLandmark->vecPointXYZInitial.x( ),
                p_pLandmark->vecPointXYZInitial.y( ),
                p_pLandmark->vecPointXYZInitial.z( ),
                p_pLandmark->vecPointXYZOptimized.x( ),
                p_pLandmark->vecPointXYZOptimized.y( ),
                p_pLandmark->vecPointXYZOptimized.z( ),
                dErrorX,
                dErrorY,
                dErrorZ,
                dErrorTotal,
                p_pLandmark->getNumberOfMeasurements( ),
                p_pLandmark->uOptimizationsSuccessful,
                p_pLandmark->vecPointXYZMean.x( ),
                p_pLandmark->vecPointXYZMean.y( ),
                p_pLandmark->vecPointXYZMean.z( ),
                p_pLandmark->uNumberOfKeyFramePresences );
        }
        static void close( ){ if( 0 != m_pFile ){ std::fclose( m_pFile ); } }

    } CLogLandmarkFinal;

    static struct CLogLandmarkFinalOptimized
    {
        static std::FILE* m_pFile;

        static void open( )
        {
            m_pFile = std::fopen( "/home/dominik/workspace_catkin/src/vi_mapper/logs/landmarks_final_optimized.txt", "w" );
            assert( 0 != m_pFile );
            std::fprintf( m_pFile, "ID_LANDMARK | X_INITIAL Y_INITIAL Z_INITIAL | X_FINAL Y_FINAL Z_FINAL | DELTA_X DELTA_Y DELTA_Z DELTA_TOTAL | MEASUREMENTS | OPTIMIZATIONS | MEAN_X MEAN_Y MEAN_Z | KEYFRAMES\n" );
        }
        static void addEntry( const CLandmark* p_pLandmark )
        {
            //ds compute errors
            const double dErrorX = std::fabs( ( p_pLandmark->vecPointXYZOptimized.x( )-p_pLandmark->vecPointXYZInitial.x( ) )/( 1.0+std::fabs( p_pLandmark->vecPointXYZInitial.x( ) ) ) );
            const double dErrorY = std::fabs( ( p_pLandmark->vecPointXYZOptimized.y( )-p_pLandmark->vecPointXYZInitial.y( ) )/( 1.0+std::fabs( p_pLandmark->vecPointXYZInitial.y( ) ) ) );
            const double dErrorZ = std::fabs( ( p_pLandmark->vecPointXYZOptimized.z( )-p_pLandmark->vecPointXYZInitial.z( ) )/( 1.0+std::fabs( p_pLandmark->vecPointXYZInitial.z( ) ) ) );
            const double dErrorTotal = dErrorX + dErrorY + dErrorZ;

            //ds write final state to file before deleting
            std::fprintf( m_pFile, "     %06lu |    %6.2f    %6.2f    %6.2f |  %6.2f  %6.2f  %6.2f |   %5.2f   %5.2f   %5.2f       %5.2f |       %06lu |        %06u | %6.2f %6.2f %6.2f |        %02u\n",
                p_pLandmark->uID,
                p_pLandmark->vecPointXYZInitial.x( ),
                p_pLandmark->vecPointXYZInitial.y( ),
                p_pLandmark->vecPointXYZInitial.z( ),
                p_pLandmark->vecPointXYZOptimized.x( ),
                p_pLandmark->vecPointXYZOptimized.y( ),
                p_pLandmark->vecPointXYZOptimized.z( ),
                dErrorX,
                dErrorY,
                dErrorZ,
                dErrorTotal,
                p_pLandmark->getNumberOfMeasurements( ),
                p_pLandmark->uOptimizationsSuccessful,
                p_pLandmark->vecPointXYZMean.x( ),
                p_pLandmark->vecPointXYZMean.y( ),
                p_pLandmark->vecPointXYZMean.z( ),
                p_pLandmark->uNumberOfKeyFramePresences );
        }
        static void close( ){ if( 0 != m_pFile ){ std::fclose( m_pFile ); } }

    } CLogLandmarkFinalOptimized;

    static struct CLogDetectionEpipolar
    {
        static std::FILE* m_pFile;

        static void open( )
        {
            m_pFile = std::fopen( "/home/dominik/workspace_catkin/src/vi_mapper/logs/detection_epipolar.txt", "w" );
            assert( 0 != m_pFile );
            std::fprintf( m_pFile, "ID_FRAME | DETECTION_POINT | LANDMARKS_TOTAL LANDMARKS_ACTIVE LANDMARKS_VISIBLE\n" );
        }
        static void addEntry( const uint64_t& p_uFrame, const UIDDetectionPoint& p_uID, const UIDLandmark& p_uNumberLandmarksTotal, const UIDLandmark& p_uNumberLandmarksActive, const UIDLandmark& p_uNumberLandmarksVisible )
        {
            std::fprintf( m_pFile, "    %04lu | %06lu | %03lu %03lu %03lu\n", p_uFrame, p_uID, p_uNumberLandmarksTotal, p_uNumberLandmarksActive, p_uNumberLandmarksVisible );
        }
        static void close( ){ if( 0 != m_pFile ){ std::fclose( m_pFile ); } }

    } CLogDetectionEpipolar;

    static struct CLogOptimizationOdometry
    {
        static std::FILE* m_pFile;

        static void open( )
        {
            m_pFile = std::fopen( "/home/dominik/workspace_catkin/src/vi_mapper/logs/optimization_odometry.txt", "w" );
            assert( 0 != m_pFile );
            std::fprintf( m_pFile, "ID_FRAME | ITERATION | TOTAL_POINTS INLIERS REPROJECTIONS | ERROR_RSS |      X      Y      Z |  DELTA | MOTION |       RISK" );
        }
        static void addEntry( const uint64_t& p_uFrame, const UIDLandmark& p_uNumberOfLandmarksInOptimization, const UIDLandmark& p_uNumberOfInliers, const UIDLandmark& p_uNumberOfReprojections, const double& p_dErrorCurrent  )
        {
            std::fprintf( m_pFile, "\n    %04lu |    INLIER |          %03lu     %03lu           %03lu | %9.2f |", p_uFrame, p_uNumberOfLandmarksInOptimization, p_uNumberOfInliers, p_uNumberOfReprojections, p_dErrorCurrent );
        }
        static void close( ){ if( 0 != m_pFile ){ std::fclose( m_pFile ); } }

    } CLogOptimizationOdometry;

    static void closeOpenLogFiles( )
    {
        //ds close all open files
        CLogLandmarkCreation::close( );
        CLogLandmarkCreationMocked::close( );
        CLogTrajectory::close( );
        CLogLandmarkFinal::close( );
        CLogLandmarkFinalOptimized::close( );
        CLogDetectionEpipolar::close( );
        CLogOptimizationOdometry::close( );
    }

};

#endif //CLOGGER_H
