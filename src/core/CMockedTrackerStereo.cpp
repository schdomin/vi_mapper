#include "CMockedTrackerStereo.h"

#include <opencv/highgui.h>
#include <Eigen/Core>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/features2d.hpp>

#include "configuration/CConfigurationOpenCV.h"
#include "configuration/CConfigurationCamera.h"
#include "utility/CWrapperOpenCV.h"
#include "exceptions/CExceptionNoMatchFound.h"
#include "vision/CMiniVisionToolbox.h"
#include "utility/CMiniTimer.h"

CMockedTrackerStereo::CMockedTrackerStereo( const uint32_t& p_uFrequencyPlaybackHz,
                                  const EPlaybackMode& p_eMode,
                                  const std::string& p_strLandmarksMocked,
                                  const uint32_t& p_uWaitKeyTimeout ): m_vecLandmarksMocked( std::make_shared< std::vector< CMockedLandmark > >( ) ),
                                                                           m_uWaitKeyTimeout( p_uWaitKeyTimeout ),
                                                                           m_pCameraLEFT( std::make_shared< CPinholeCamera >( CConfigurationCamera::LEFT::cPinholeCamera ) ),
                                                                           m_pCameraRIGHT( std::make_shared< CPinholeCamera >( CConfigurationCamera::RIGHT::cPinholeCamera ) ),
                                                                           m_pCameraSTEREO( std::make_shared< CMockedStereoCamera >( p_strLandmarksMocked, m_pCameraLEFT, m_pCameraRIGHT ) ),

                                                                           m_uFrameCount( 0 ),
                                                                           m_vecTranslationLast( 1.0, 1.0, 1.0 ),
                                                                           m_dTranslationDeltaForMAPMeters( 0.5 ),

                                                                           m_uMaximumFailedSubsequentTrackingsPerLandmark( 5 ),
                                                                           m_uVisibleLandmarksMinimum( 1 ),
                                                                           m_dMinimumDepthMeters( 0.5 ),
                                                                           m_dMaximumDepthMeters( 100.0 ),

                                                                           m_cMatcherEpipolar( m_pCameraLEFT, m_pCameraRIGHT, m_pCameraSTEREO, m_uMaximumFailedSubsequentTrackingsPerLandmark ),

                                                                           m_uAvailableLandmarkID( 0 ),
                                                                           m_vecLandmarks( std::make_shared< std::vector< CLandmark* > >( ) ),
                                                                           m_uNumberofLastVisibleLandmarks( 0 ),

                                                                           m_eMode( p_eMode ),
                                                                           m_bIsShutdownRequested( false ),
                                                                           m_dFrequencyPlaybackHz( p_uFrequencyPlaybackHz ),
                                                                           m_uFrequencyPlaybackDeltaHz( 50 ),
                                                                           m_iPlaybackSpeedupCounter( 0 ),
                                                                           m_cRandomGenerator( 1337 ),

                                                                           m_ptPositionXY( 0.0, 0.0 ),
                                                                           m_uOffsetTrajectoryU( 180 ),
                                                                           m_uOffsetTrajectoryV( 360 ),
                                                                           m_uTimingToken( 0 ),
                                                                           m_uFramesCurrentCycle( 0 ),
                                                                           m_dPreviousFrameRate( 0.0 ),
                                                                           m_uTotalMeasurementPoints( 0 ),
                                                                           m_uMAPPoints( 0 )
{
    //ds debug logging
    m_pFileLandmarkCreation = std::fopen( "/home/dominik/workspace_catkin/src/vi_mapper/logs/landmarks_creation.txt", "w" );
    m_pFileLandmarkFinal    = std::fopen( "/home/dominik/workspace_catkin/src/vi_mapper/logs/landmarks_final.txt", "w" );

    //ds dump file format
    std::fprintf( m_pFileLandmarkCreation, "FRAME | ID_LANDMARK |      X      Y      Z |  DEPTH | U_LEFT V_LEFT | U_RIGHT V_RIGHT | KEYPOINT_SIZE\n" );
    std::fprintf( m_pFileLandmarkFinal, "ID_LANDMARK | X_INITIAL Y_INITIAL Z_INITIAL | X_FINAL Y_FINAL Z_FINAL | DELTA_X DELTA_Y DELTA_Z DELTA_TOTAL | CALIBRATIONS | MEAN_X MEAN_Y MEAN_Z\n" );

    //ds initialize reference frames with black images
    m_matDisplayLowerReference = cv::Mat::zeros( m_pCameraSTEREO->m_uPixelHeight, 2*m_pCameraSTEREO->m_uPixelWidth, CV_8UC3 );

    //ds trajectory maps
    m_matTrajectoryXY = cv::Mat( 720, 720, CV_8UC3, CColorCodeBGR( 255, 255, 255 ) );

    //ds draw meters grid
    for( uint32_t u = 0; u < 720; u += 10 )
    {
        cv::line( m_matTrajectoryXY, cv::Point( u, 0 ),cv::Point( u, 720 ), CColorCodeBGR( 175, 175, 175 ) );
        cv::line( m_matTrajectoryXY, cv::Point( 0, u ),cv::Point( 720, u ), CColorCodeBGR( 175, 175, 175 ) );
    }

    //ds initialize the window
    cv::namedWindow( "stereo matching MOCKED", cv::WINDOW_AUTOSIZE );

    CLogger::openBox( );
    std::printf( "<CMockedTrackerStereo>(CMockedTrackerStereo) instance allocated\n" );
    CLogger::closeBox( );
}

CMockedTrackerStereo::~CMockedTrackerStereo( )
{
    //ds free all landmarks
    for( const CLandmark* pLandmark: *m_vecLandmarks )
    {
        //ds compute errors
        const double dErrorX = std::fabs( ( pLandmark->vecPointXYZCalibrated.x( )-pLandmark->vecPointXYZInitial.x( ) )/( 1.0+std::fabs( pLandmark->vecPointXYZInitial.x( ) ) ) );
        const double dErrorY = std::fabs( ( pLandmark->vecPointXYZCalibrated.y( )-pLandmark->vecPointXYZInitial.y( ) )/( 1.0+std::fabs( pLandmark->vecPointXYZInitial.y( ) ) ) );
        const double dErrorZ = std::fabs( ( pLandmark->vecPointXYZCalibrated.z( )-pLandmark->vecPointXYZInitial.z( ) )/( 1.0+std::fabs( pLandmark->vecPointXYZInitial.z( ) ) ) );
        const double dErrorTotal = dErrorX + dErrorY + dErrorZ;

        //ds write final state to file before deleting
        std::fprintf( m_pFileLandmarkFinal, "     %06lu |    %6.2f    %6.2f    %6.2f |  %6.2f  %6.2f  %6.2f |   %5.2f   %5.2f   %5.2f       %5.2f |       %06u | %6.2f %6.2f %6.2f\n", pLandmark->uID,
                                                                              pLandmark->vecPointXYZInitial.x( ),
                                                                              pLandmark->vecPointXYZInitial.y( ),
                                                                              pLandmark->vecPointXYZInitial.z( ),
                                                                              pLandmark->vecPointXYZCalibrated.x( ),
                                                                              pLandmark->vecPointXYZCalibrated.y( ),
                                                                              pLandmark->vecPointXYZCalibrated.z( ),
                                                                              dErrorX,
                                                                              dErrorY,
                                                                              dErrorZ,
                                                                              dErrorTotal,
                                                                              pLandmark->uCalibrations,
                                                                              pLandmark->vecMeanMeasurement.x( ),
                                                                              pLandmark->vecMeanMeasurement.y( ),
                                                                              pLandmark->vecMeanMeasurement.z( ) );

        delete pLandmark;
    }

    //ds debug
    std::fclose( m_pFileLandmarkCreation );
    std::fclose( m_pFileLandmarkFinal );

    std::printf( "<CMockedTrackerStereo>(~CMockedTrackerStereo) instance deallocated\n" );
}

void CMockedTrackerStereo::receivevDataVIWithPose( const std::shared_ptr< txt_io::PinholeImageMessage > p_pImageLEFT,
                                             const std::shared_ptr< txt_io::PinholeImageMessage > p_pImageRIGHT,
                                             const txt_io::CIMUMessage& p_cIMU,
                                             const std::shared_ptr< txt_io::CPoseMessage > p_cPose )
{
    //ds flush all output
    std::fflush( stdout );

    //ds preprocessed images
    cv::Mat matPreprocessedLEFT;
    cv::Mat matPreprocessedRIGHT;

    //ds preprocess images
    cv::equalizeHist( p_pImageLEFT->image( ), matPreprocessedLEFT );
    cv::equalizeHist( p_pImageRIGHT->image( ), matPreprocessedRIGHT );
    m_pCameraSTEREO->undistortAndrectify( matPreprocessedLEFT, matPreprocessedRIGHT );

    //ds pose information
    Eigen::Isometry3d matTransformationIMUToWORLD;
    matTransformationIMUToWORLD.translation( ) = p_cPose->getPosition( );
    matTransformationIMUToWORLD.linear( )      = p_cPose->getOrientationMatrix( );

    //ds compute LEFT camera transformation
    const Eigen::Isometry3d matTransformationLEFTToWORLD( matTransformationIMUToWORLD*m_pCameraLEFT->m_matTransformationToIMU );

    //ds process images
    _trackLandmarks( matPreprocessedLEFT, matPreprocessedRIGHT, matTransformationLEFTToWORLD, p_cIMU.getAngularVelocity( ), p_cIMU.getLinearAcceleration( ) );

    //ds flush all output
    std::fflush( stdout );
}

void CMockedTrackerStereo::_trackLandmarks( const cv::Mat& p_matImageLEFT,
                                      const cv::Mat& p_matImageRIGHT,
                                      const Eigen::Isometry3d& p_matTransformationLEFTtoWORLD,
                                      const Eigen::Vector3d& p_vecAngularVelocity,
                                      const Eigen::Vector3d& p_vecLinearAcceleration )
{
    //ds current translation
    const CPoint3DInWorldFrame vecTranslationCurrent( p_matTransformationLEFTtoWORLD.translation( ) );
    m_ptPositionXY = cv::Point2d( vecTranslationCurrent.x( ), vecTranslationCurrent.y( ) );

    //ds normalized gravity
    const Eigen::Vector3d vecLinearAccelerationNormalized( p_vecLinearAcceleration.normalized( ) );

    //ds draw position on trajectory mat
    cv::circle( m_matTrajectoryXY, cv::Point2d( m_uOffsetTrajectoryU+m_ptPositionXY.x*10, m_uOffsetTrajectoryV-m_ptPositionXY.y*10 ), 2, cv::Scalar( 255, 0, 0 ), -1 );

    //ds get images into triple channel mats (display only)
    cv::Mat matDisplayLEFT;
    cv::Mat matDisplayRIGHT;

    //ds get images to triple channel for colored display
    cv::cvtColor( p_matImageLEFT, matDisplayLEFT, cv::COLOR_GRAY2BGR );
    cv::cvtColor( p_matImageRIGHT, matDisplayRIGHT, cv::COLOR_GRAY2BGR );

    //ds get clean copies
    const cv::Mat matDisplayLEFTClean( matDisplayLEFT.clone( ) );
    const cv::Mat matDisplayRIGHTClean( matDisplayRIGHT.clone( ) );

    //ds temporary trajectory handle to allow temporal drawing
    cv::Mat matDisplayTrajectory( m_matTrajectoryXY.clone( ) );

    //ds get currently visible landmarks
    const std::shared_ptr< std::vector< const CMeasurementLandmark* > > vecVisibleLandmarks( m_cMatcherEpipolar.getVisibleLandmarksEssential( m_uFrameCount, matDisplayLEFT, matDisplayRIGHT, p_matImageLEFT, p_matImageRIGHT, p_matTransformationLEFTtoWORLD, matDisplayTrajectory ) );
    m_uNumberofLastVisibleLandmarks = vecVisibleLandmarks->size( );

    //ds add to data structure if delta is sufficiently high
    if( m_dTranslationDeltaForMAPMeters < ( vecTranslationCurrent-m_vecTranslationLast ).squaredNorm( ) && m_uVisibleLandmarksMinimum < m_uNumberofLastVisibleLandmarks )
    {
        m_vecTranslationLast = vecTranslationCurrent;
        m_vecLogG2OMeasurementPoints.push_back( CMeasurementPose( p_matTransformationLEFTtoWORLD, vecLinearAccelerationNormalized, vecVisibleLandmarks ) );
        //std::printf( "<CTrackerStereo>(_trackLandmarks) stashed measurement %lu with landmarks (%lu)\n", m_vecLogMeasurementPoints.size( ), vecVisibleLandmarks->size( ) );

        ++m_uMAPPoints;
    }

    //ds check if we have to detect new landmarks
    if( m_uVisibleLandmarksMinimum > m_uNumberofLastVisibleLandmarks )
    {
        //ds clean the lower display
        cv::hconcat( matDisplayLEFTClean, matDisplayRIGHTClean, m_matDisplayLowerReference );

        //ds detect landmarks
        const std::shared_ptr< std::vector< CLandmark* > > vecNewLandmarks( _getNewLandmarksTriangulated( m_uFrameCount, m_matDisplayLowerReference, matDisplayTrajectory, m_ptPositionXY, p_matTransformationLEFTtoWORLD ) );

        //ds add to permanent reference holder
        m_vecLandmarks->insert( m_vecLandmarks->end( ), vecNewLandmarks->begin( ), vecNewLandmarks->end( ) );

        //ds add this measurement point to the epipolar matcher
        m_cMatcherEpipolar.addMeasurementPoint( p_matTransformationLEFTtoWORLD, vecNewLandmarks );

        ++m_uTotalMeasurementPoints;

        cv::circle( m_matTrajectoryXY, cv::Point2d( m_uOffsetTrajectoryU+p_matTransformationLEFTtoWORLD.translation( )( 0 )*10, m_uOffsetTrajectoryV-p_matTransformationLEFTtoWORLD.translation( )( 1 )*10 ), 20, CColorCodeBGR( 0, 255, 0 ), 1 );
    }

    //ds build display mat
    _drawInfoBox( matDisplayLEFT );
    cv::Mat matDisplayUpper = cv::Mat( m_pCameraSTEREO->m_uPixelHeight, 2*m_pCameraSTEREO->m_uPixelWidth, CV_8UC3 );
    cv::hconcat( matDisplayLEFT, matDisplayRIGHT, matDisplayUpper );
    cv::Mat matDisplayComplete = cv::Mat( 2*m_pCameraSTEREO->m_uPixelHeight, 2*m_pCameraSTEREO->m_uPixelWidth, CV_8UC3 );
    cv::vconcat( matDisplayUpper, m_matDisplayLowerReference, matDisplayComplete );

    //ds display
    cv::imshow( "stereo matching MOCKED", matDisplayComplete );
    cv::imshow( "trajectory (x,y)", matDisplayTrajectory );

    //ds if there was a keystroke
    int iLastKeyStroke( cv::waitKey( m_uWaitKeyTimeout ) );
    if( -1 != iLastKeyStroke )
    {
        //ds increment count
        ++m_uFrameCount;

        //ds user input - reset frame rate counting
        m_uFramesCurrentCycle = 0;

        //ds evaluate keystroke
        switch( iLastKeyStroke )
        {
            case CConfigurationOpenCV::KeyStroke::iEscape:
            {
                _shutDown( );
                return;
            }
            case CConfigurationOpenCV::KeyStroke::iNumpadMinus:
            {
                _slowDown( );
                break;
            }
            case CConfigurationOpenCV::KeyStroke::iNumpadPlus:
            {
                _speedUp( );
                break;
            }
            default:
            {
                //std::printf( "<>(_trackLandmarksAuto) unknown keystroke: %i\n", iLastKeyStroke );
                break;
            }
        }
    }
    else
    {
        _updateFrameRateForInfoBox( );

        //ds increment count
        ++m_uFrameCount;
    }
}

const std::shared_ptr< std::vector< CLandmark* > > CMockedTrackerStereo::_getNewLandmarksTriangulated( const uint64_t& p_uFrame,
                                                                                                       cv::Mat& p_matDisplay,
                                                                                                       cv::Mat& p_matDisplayTrajectory,
                                                                                                       const cv::Point2d& p_ptPositionXY,
                                                                                                       const Eigen::Isometry3d& p_matTransformationLEFTtoWORLD )
{
    //ds precompute extrinsics
    const Eigen::Isometry3d matTransformationWORLDtoLEFT( p_matTransformationLEFTtoWORLD.inverse( ) );
    const Eigen::Matrix3d matKRotation( m_pCameraLEFT->m_matIntrinsic*matTransformationWORLDtoLEFT.linear( ) );
    const Eigen::Vector3d vecCameraPosition( matTransformationWORLDtoLEFT.translation( ) );
    const Eigen::Vector3d vecKTranslation( m_pCameraLEFT->m_matIntrinsic*vecCameraPosition );

    //ds solution holder
    std::shared_ptr< std::vector< CLandmark* > > vecNewLandmarks( std::make_shared< std::vector< CLandmark* > >( ) );

    //ds get currently visible landmarks
    const std::map< UIDLandmark, CMockedDetection > mapVisibleLandmarks( m_pCameraSTEREO->getDetectedLandmarks( p_ptPositionXY, matTransformationWORLDtoLEFT, p_matDisplayTrajectory ) );

    //ds process visible landmarks
    for( const std::pair< UIDLandmark, CMockedDetection > cDetection: mapVisibleLandmarks )
    {
        //ds naming consistency
        const cv::Point2d& ptLandmarkLEFT( cDetection.second.ptUVLEFT );

        //ds triangulate the point
        const CPoint3DInCameraFrame vecPointTriangulatedLEFT( CMiniVisionToolbox::getPointStereoLinearTriangulationSVDLS( cDetection.second.ptUVLEFT,
                                                                                                                          cDetection.second.ptUVRIGHT,
                                                                                                                          m_pCameraLEFT->m_matProjection,
                                                                                                                          m_pCameraRIGHT->m_matProjection ) );

        const double& dDepthMeters( vecPointTriangulatedLEFT.z( ) );

        //ds check if point is in front of camera an not more than a defined distance away
        if( m_dMinimumDepthMeters < dDepthMeters && m_dMaximumDepthMeters > dDepthMeters )
        {
            //ds compute triangulated point in world frame
            const CPoint3DInWorldFrame vecPointTriangulatedWORLD( p_matTransformationLEFTtoWORLD*vecPointTriangulatedLEFT );

            //ds draw reprojection of triangulation
            const cv::Point2d ptLandmarkRIGHT( m_pCameraRIGHT->getProjection( vecPointTriangulatedLEFT ) );

            //ds epipolar constraint
            assert( std::round( ptLandmarkLEFT.y ) == std::round( ptLandmarkRIGHT.y ) );

            //ds allocate a new landmark and add the current position (HACK: set mocked landmark id into keypoint size field to avoid another landmark class)
            CLandmark* cLandmark( new CLandmark( m_uAvailableLandmarkID,
                                                 CDescriptor( ),
                                                 cDetection.first,
                                                 vecPointTriangulatedWORLD,
                                                 m_pCameraLEFT->getHomogenized( ptLandmarkLEFT ),
                                                 ptLandmarkLEFT,
                                                 ptLandmarkRIGHT,
                                                 vecPointTriangulatedLEFT,
                                                 vecCameraPosition,
                                                 matKRotation,
                                                 vecKTranslation,
                                                 p_uFrame ) );

            //ds log creation
            std::fprintf( m_pFileLandmarkCreation, " %04lu |      %06lu | %6.2f %6.2f %6.2f | %6.2f | %6.2f %6.2f |  %6.2f  %6.2f |        %6.2f\n", p_uFrame,
                                                                                          cLandmark->uID,
                                                                                          cLandmark->vecPointXYZCalibrated.x( ),
                                                                                          cLandmark->vecPointXYZCalibrated.y( ),
                                                                                          cLandmark->vecPointXYZCalibrated.z( ),
                                                                                          dDepthMeters,
                                                                                          ptLandmarkLEFT.x,
                                                                                          ptLandmarkLEFT.y,
                                                                                          ptLandmarkRIGHT.x,
                                                                                          ptLandmarkRIGHT.y,
                                                                                          cLandmark->dKeyPointSize );

            //ds add to newly detected
            vecNewLandmarks->push_back( cLandmark );

            //ds new landmark
            ++m_uAvailableLandmarkID;

            //ds draw detected point
            cv::line( p_matDisplay, ptLandmarkLEFT, cv::Point2f( ptLandmarkLEFT.x+m_pCameraSTEREO->m_uPixelWidth, ptLandmarkLEFT.y ), CColorCodeBGR( 175, 175, 175 ) );
            cv::circle( p_matDisplay, ptLandmarkLEFT, 2, CColorCodeBGR( 0, 255, 0 ), -1 );
            //cv::circle( p_matDisplay, ptLandmarkLEFT, cLandmark->dKeyPointSize, CColorCodeBGR( 255, 0, 0 ), 1 );
            cv::putText( p_matDisplay, std::to_string( cLandmark->uID ) , cv::Point2d( ptLandmarkLEFT.x+10, ptLandmarkLEFT.y+10 ), cv::FONT_HERSHEY_PLAIN, 0.5, CColorCodeBGR( 0, 0, 255 ) );

            //ds draw landmark in world (2d)
            cv::circle( m_matTrajectoryXY, cv::Point2d( m_uOffsetTrajectoryU+vecPointTriangulatedWORLD(0)*10, m_uOffsetTrajectoryV-vecPointTriangulatedWORLD(1)*10 ), 3, CColorCodeBGR( 0, 165, 255 ), -1 );

            //ds draw reprojection of triangulation
            cv::circle( p_matDisplay, cv::Point2d( ptLandmarkRIGHT.x+m_pCameraSTEREO->m_uPixelWidth, ptLandmarkRIGHT.y ), 2, CColorCodeBGR( 255, 0, 0 ), -1 );
        }
        else
        {
            cv::circle( p_matDisplay, ptLandmarkLEFT, 2, CColorCodeBGR( 0, 0, 255 ), -1 );
            //cv::circle( p_matDisplay, ptLandmarkLEFT, cKeyPoint.size, CColorCodeBGR( 0, 0, 255 ) );

            std::printf( "<CTrackerStereo>(_getNewLandmarks) could not find match for keypoint (invalid depth: %f m)\n", vecPointTriangulatedLEFT(2) );
        }
    }

    std::printf( "<CMockedTrackerStereo>(_getNewLandmarks) added new landmarks: %lu/%lu\n", vecNewLandmarks->size( ), mapVisibleLandmarks.size( ) );

    //ds return found landmarks
    return vecNewLandmarks;
}

void CMockedTrackerStereo::_shutDown( )
{
    m_bIsShutdownRequested = true;
    std::printf( "<CMockedTrackerStereo>(_shutDown) termination requested, <CTrackerStereo> disabled\n" );
}

void CMockedTrackerStereo::_speedUp( )
{
    ++m_iPlaybackSpeedupCounter;
    m_dFrequencyPlaybackHz += std::pow( 2, m_iPlaybackSpeedupCounter )*m_uFrequencyPlaybackDeltaHz;
    std::printf( "<CMockedTrackerStereo>(_speedUp) increased playback speed to: %.0f Hz \n", m_dFrequencyPlaybackHz );
}

void CMockedTrackerStereo::_slowDown( )
{
    m_dFrequencyPlaybackHz -= std::pow( 2, m_iPlaybackSpeedupCounter )*m_uFrequencyPlaybackDeltaHz;

    //ds 12 fps minimum (one of 2 images 10 imu messages and 1 pose: 13-1)
    if( 1 < m_dFrequencyPlaybackHz )
    {
        std::printf( "<CMockedTrackerStereo>(_slowDown)  reduced playback speed to: %.0f Hz \n", m_dFrequencyPlaybackHz );
        --m_iPlaybackSpeedupCounter;
    }
    else
    {
        m_dFrequencyPlaybackHz = 1;
        std::printf( "<CMockedTrackerStereo>(_slowDown)  reduced playback speed to: %.0f Hz \n", m_dFrequencyPlaybackHz );
    }
}

void CMockedTrackerStereo::_updateFrameRateForInfoBox( const uint32_t& p_uFrameProbeRange )
{
    //ds check if we can compute the frame rate
    if( p_uFrameProbeRange == m_uFramesCurrentCycle )
    {
        //ds get time delta
        const double dDuration( CMiniTimer::toc( m_uTimingToken ) );

        //ds compute framerate
        m_dPreviousFrameRate = p_uFrameProbeRange/dDuration;

        //ds enable new measurement
        m_uFramesCurrentCycle = 0;
    }

    //ds check if its the first frame since the last count
    if( 0 == m_uFramesCurrentCycle )
    {
        //ds stop time
        m_uTimingToken = CMiniTimer::tic( );
    }

    //ds count frames
    ++m_uFramesCurrentCycle;
}

void CMockedTrackerStereo::_drawInfoBox( cv::Mat& p_matDisplay ) const
{
    char chBuffer[256];

    switch( m_eMode )
    {
        case ePlaybackStepwise:
        {
            std::snprintf( chBuffer, 256, "[%04lu] STEPWISE | POSITION XY(%5.1f,%5.1f) | LANDMARKS: %2lu(%4lu) | DETECTIONS: %1lu(%2lu) | POSES: %2lu", m_uFrameCount, m_ptPositionXY.x, m_ptPositionXY.y, m_uNumberofLastVisibleLandmarks, m_vecLandmarks->size( ), m_cMatcherEpipolar.getNumberOfActiveMeasurementPoints( ), m_uTotalMeasurementPoints, m_uMAPPoints );
            break;
        }
        case ePlaybackBenchmark:
        {
            std::snprintf( chBuffer, 256, "[%04lu] FPS %4.1f(BENCHMARK) | POSITION XY(%5.1f,%5.1f) | LANDMARKS: %2lu(%4lu) | DETECTIONS: %1lu(%2lu) | POSES: %2lu", m_uFrameCount, m_dPreviousFrameRate, m_ptPositionXY.x, m_ptPositionXY.y, m_uNumberofLastVisibleLandmarks, m_vecLandmarks->size( ), m_cMatcherEpipolar.getNumberOfActiveMeasurementPoints( ), m_uTotalMeasurementPoints, m_uMAPPoints );
            break;
        }
        case ePlaybackInteractive:
        {
            std::snprintf( chBuffer, 256, "[%04lu] FPS %4.1f(%4.2f Hz) | POSITION XY(%5.1f,%5.1f) | LANDMARKS: %0lu(%4lu) | DETECTIONS: %1lu(%2lu) | POSES: %2lu", m_uFrameCount, m_dPreviousFrameRate, m_dFrequencyPlaybackHz, m_ptPositionXY.x, m_ptPositionXY.y, m_uNumberofLastVisibleLandmarks, m_vecLandmarks->size( ), m_cMatcherEpipolar.getNumberOfActiveMeasurementPoints( ), m_uTotalMeasurementPoints, m_uMAPPoints );
            break;
        }
        default:
        {
            std::printf( "<CMockedTrackerStereo>(_drawInfoBox) unsupported playback mode, no info box displayed\n" );
            break;
        }
    }


    p_matDisplay( cv::Rect( 0, 0, m_pCameraLEFT->m_iWidthPixel, 17 ) ).setTo( CColorCodeBGR( 0, 0, 0 ) );
    cv::putText( p_matDisplay, chBuffer , cv::Point2i( 2, 12 ), cv::FONT_HERSHEY_PLAIN, 0.8, CColorCodeBGR( 0, 0, 255 ) );
}
