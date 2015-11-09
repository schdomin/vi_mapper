#ifndef CLOGGER_H
#define CLOGGER_H

#include <chrono>
#include <fstream>

#include "types/CLandmark.h"
#include "types/TypesMocked.h"

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

    template < class T > static void writeDatum( std::ostream& p_osStream, const T& p_tValue )
    {
        const char * pValue = reinterpret_cast< const char* >( &p_tValue );
        p_osStream.write( pValue, sizeof( T ) );
    }

    template < class T > static void readDatum( std::istream& p_isStream, T& p_tValue )
    {
        char * pValue = reinterpret_cast< char* >( &p_tValue );
        p_isStream.read( pValue, sizeof( T ) );
    }

//ds logging
public:

    static struct CLogLandmarkCreation
    {
        static std::FILE* m_pFile;

        static void open( )
        {
            m_pFile = std::fopen( "logs/landmarks_creation.txt", "w" );
            assert( 0 != m_pFile );
            std::fprintf( m_pFile, "ID_FRAME | ID_LANDMARK |      X      Y      Z :  DEPTH | U_LEFT V_LEFT | U_RIGHT V_RIGHT | KEYPOINT_SIZE\n" );
        }
        static void addEntry( const UIDFrame& p_uFrame, const CLandmark* p_pLandmark, const double& p_dDepthMeters, const cv::Point2d& p_ptLandmarkLEFT, const cv::Point2d& p_ptLandmarkRIGHT )
        {
            assert( 0 != m_pFile );
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

    } CLogLandmarkCreation;

    static struct CLogLandmarkCreationMocked
    {
        static std::FILE* m_pFile;

        static void open( )
        {
            m_pFile = std::fopen( "logs/landmarks_creation_mocked.txt", "w" );
            assert( 0 != m_pFile );
            std::fprintf( m_pFile, "FRAME | ID_LANDMARK |      X      Y      Z :  DEPTH | U_LEFT V_LEFT : NOISE_U NOISE_V | U_RIGHT V_RIGHT : NOISE_U NOISE_V | KEYPOINT_SIZE\n" );

        }
        static void addEntry( const UIDFrame& p_uFrame, const CLandmark* p_pLandmark, const double& p_dDepthMeters, const cv::Point2d& p_ptLandmarkLEFT, const cv::Point2d& p_ptLandmarkRIGHT, const CMockedDetection& p_cMockedDetection )
        {
            assert( 0 != m_pFile );
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
            m_pFile = std::fopen( "logs/trajectory.txt", "w" );
            assert( 0 != m_pFile );
            std::fprintf( m_pFile, "ID_FRAME |      X      Y      Z | QUAT_X QUAT_Y QUAT_Z\n" );
        }
        static void addEntry( const UIDFrame& p_uFrame, const CPoint3DWORLD& p_vecPosition, const Eigen::Quaterniond& p_vecQuaternion )
        {
            assert( 0 != m_pFile );
            std::fprintf( m_pFile, "    %04lu | %6.2f %6.2f %6.2f | %6.3f %6.3f %6.3f\n",
                p_uFrame,
                p_vecPosition.x( ),
                p_vecPosition.y( ),
                p_vecPosition.z( ),
                p_vecQuaternion.x( )/p_vecQuaternion.w( ),
                p_vecQuaternion.y( )/p_vecQuaternion.w( ),
                p_vecQuaternion.z( )/p_vecQuaternion.w( ) );
        }
        static void close( ){ if( 0 != m_pFile ){ std::fclose( m_pFile ); } }

    } CLogTrajectory;

    static struct CLogLandmarkFinal
    {
        static std::FILE* m_pFile;

        static void open( )
        {
            m_pFile = std::fopen( "logs/landmarks_final.txt", "w" );
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

            assert( 0 != m_pFile );

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
            m_pFile = std::fopen( "logs/landmarks_final_optimized.txt", "w" );
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

            assert( 0 != m_pFile );

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
            m_pFile = std::fopen( "logs/detection_epipolar.txt", "w" );
            assert( 0 != m_pFile );
            std::fprintf( m_pFile, "ID_FRAME | ID_DETECTION | LANDMARKS: TOTAL ACTIVE VISIBLE\n" );
        }
        static void addEntry( const UIDFrame& p_uFrame, const UIDDetectionPoint& p_uID, const UIDLandmark& p_uNumberLandmarksTotal, const UIDLandmark& p_uNumberLandmarksActive, const UIDLandmark& p_uNumberLandmarksVisible )
        {
            assert( 0 != m_pFile );
            std::fprintf( m_pFile, "    %04lu |       %06lu |              %03lu    %03lu     %03lu\n", p_uFrame, p_uID, p_uNumberLandmarksTotal, p_uNumberLandmarksActive, p_uNumberLandmarksVisible );
        }
        static void close( ){ if( 0 != m_pFile ){ std::fclose( m_pFile ); } }

    } CLogDetectionEpipolar;

    static struct CLogOptimizationOdometry
    {
        static std::FILE* m_pFile;

        static void open( )
        {
            m_pFile = std::fopen( "logs/optimization_odometry.txt", "w" );
            assert( 0 != m_pFile );
            std::fprintf( m_pFile, "ID_FRAME | ITERATION | TOTAL_POINTS INLIERS | ERROR_AVERAGE | ERROR_RSS |      X      Y      Z | DELTA: T      R |       RISK" );
        }
        static void addEntryIteration( const UIDFrame& p_uFrame, const uint8_t& p_uIteration, const UIDLandmark& p_uNumberOfLandmarksInOptimization, const UIDLandmark& p_uNumberOfInliers, const double& p_dErrorAverage, const double& p_dErrorCurrent )
        {
            assert( 0 != m_pFile );
            std::fprintf( m_pFile, "\n    %04lu |        %02u |          %03lu     %03lu |        %6.2f | %9.2f |", p_uFrame, p_uIteration, p_uNumberOfLandmarksInOptimization, p_uNumberOfInliers, p_dErrorAverage, p_dErrorCurrent );
        }
        static void addEntryInliers( const UIDFrame& p_uFrame, const UIDLandmark& p_uNumberOfLandmarksInOptimization, const UIDLandmark& p_uNumberOfInliers, const UIDLandmark& p_uNumberOfReprojections, const double& p_dErrorCurrent )
        {
            assert( 0 != m_pFile );
            std::fprintf( m_pFile, "\n    %04lu |    INLIER |          %03lu     %03lu           %03lu | %9.2f |", p_uFrame, p_uNumberOfLandmarksInOptimization, p_uNumberOfInliers, p_uNumberOfReprojections, p_dErrorCurrent );
        }
        static void addEntryResult( const CPoint3DWORLD& p_vecPosition, const double& p_dOptimizationDeltaTranslation, const double& p_dOptimizationDeltaRotation, const double& p_dOptimizationRisk )
        {
            assert( 0 != m_pFile );
            std::fprintf( m_pFile, " %6.2f %6.2f %6.2f |   %6.4f %6.4f |     %6.4f", p_vecPosition.x( ), p_vecPosition.y( ), p_vecPosition.z( ), p_dOptimizationDeltaTranslation, p_dOptimizationDeltaRotation, p_dOptimizationRisk );
        }
        static void close( ){ if( 0 != m_pFile ){ std::fclose( m_pFile ); } }

    } CLogOptimizationOdometry;

    static struct CLogIMUInput
    {
        static std::FILE* m_pFile;

        static void open( )
        {
            m_pFile = std::fopen( "logs/imu_input.txt", "w" );
            assert( 0 != m_pFile );
            std::fprintf( m_pFile, "ID_FRAME | LINEAR ACCELERATION:     X      Y      Z | FILTERED:     X      Y      Z | ANGULAR VELOCITY:     X      Y      Z | FILTERED:     X      Y      Z\n" );
        }
        static void addEntry( const UIDFrame& p_uFrame, const CLinearAccelerationWORLD& p_vecLinearAcceleration, const CLinearAccelerationWORLD& p_vecLinearAccelerationFiltered, const CAngularVelocityLEFT& p_vecAngularVelocity, const CAngularVelocityLEFT& p_vecAngularVelocityFiltered )
        {
            assert( 0 != m_pFile );
            std::fprintf( m_pFile, "    %04lu |                     %6.2f %6.2f %6.2f |          %6.2f %6.2f %6.2f |                  %6.2f %6.2f %6.2f |          %6.2f %6.2f %6.2f\n", p_uFrame,
                          p_vecLinearAcceleration.x( ), p_vecLinearAcceleration.y( ), p_vecLinearAcceleration.z( ), p_vecLinearAccelerationFiltered.x( ), p_vecLinearAccelerationFiltered.y( ), p_vecLinearAccelerationFiltered.z( ),
                          p_vecAngularVelocity.x( ), p_vecAngularVelocity.y( ), p_vecAngularVelocity.z( ), p_vecAngularVelocityFiltered.x( ), p_vecAngularVelocityFiltered.y( ), p_vecAngularVelocityFiltered.z( ) );
        }
        static void close( ){ if( 0 != m_pFile ){ std::fclose( m_pFile ); } }

    } CLogIMUInput;

    static struct CLogTrajectoryKITTI
    {
        static std::ofstream ofOutfile;

        static void open( )
        {
            ofOutfile.open( "logs/trajectory_final_KITTI.txt", std::ofstream::out );
            assert( ofOutfile.good( ) );
        }
        static void addEntry( const UIDFrame& p_uFrame, const Eigen::Isometry3d& p_matTransformationLEFTtoWORLD, const Eigen::Vector3d& p_vecTranslationToG2o )
        {
            //ds clear translation to original frame
            Eigen::Isometry3d matTransformationLEFTtoWORLDShifted( p_matTransformationLEFTtoWORLD );
            matTransformationLEFTtoWORLDShifted.translation( ) += p_vecTranslationToG2o;

            //ds get to matrix representation
            const Eigen::Matrix4d matTransformationLEFTtoWORLD( matTransformationLEFTtoWORLDShifted.matrix( ) );

            //ds KITTI 3x4 format
            assert( ofOutfile.good( ) );

            //ds frame number
            ofOutfile << p_uFrame;

            //ds transformation matrix
            for( uint8_t u = 0; u < 3; ++u )
            {
                for( uint8_t v = 0; v < 4; ++v )
                {
                    ofOutfile << " " << matTransformationLEFTtoWORLD(u,v);
                }
            }

            //ds done
            ofOutfile << "\n";
        }
        static void close( ){ if( ofOutfile.good( ) ){ ofOutfile.close( ); } }

    } CLogTrajectoryKITTI;

};

#endif //CLOGGER_H
