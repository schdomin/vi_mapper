#include "CTrackerStereo.h"

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

CTrackerStereo::CTrackerStereo( const uint32_t& p_uFrequencyPlaybackHz,
                                  const EPlaybackMode& p_eMode,
                                  const uint32_t& p_uWaitKeyTimeout ): m_uWaitKeyTimeout( p_uWaitKeyTimeout ),
                                                                           m_pCameraLEFT( std::make_shared< CPinholeCamera >( CConfigurationCamera::LEFT::cPinholeCamera ) ),
                                                                           m_pCameraRIGHT( std::make_shared< CPinholeCamera >( CConfigurationCamera::RIGHT::cPinholeCamera ) ),
                                                                           m_pCameraSTEREO( std::make_shared< CStereoCamera >( m_pCameraLEFT, m_pCameraRIGHT ) ),

                                                                           m_uFrameCount( 0 ),
                                                                           m_vecTranslationLast( 1.0, 1.0, 1.0 ),
                                                                           m_dTranslationDeltaForMAPMeters( 0.5 ),

                                                                           /*//ds ORB
                                                                           m_pDetector( std::make_shared< cv::OrbFeatureDetector >( 100, 1.2, 3, 20, 0, 4, cv::ORB::HARRIS_SCORE, 20 ) ),
                                                                           //m_pDetector( std::make_shared< cv::GoodFeaturesToTrackDetector >( 100, 0.01, 18.0, 15, true ) ),
                                                                           m_pExtractor( std::make_shared< cv::OrbDescriptorExtractor >( 100, 1.2, 3, 20, 0, 4, cv::ORB::HARRIS_SCORE, 20 ) ),
                                                                           m_pMatcher( std::make_shared< cv::BFMatcher >( ) ),
                                                                           m_dMatchingDistanceCutoffTriangulation( 400.0 ),
                                                                           m_dMatchingDistanceCutoffTracking( 350.0 ),*/

                                                                           //ds BRIEF (calibrated 2015-05-31)
                                                                           // m_uKeyPointSize( 7 ),
                                                                           m_pDetector( std::make_shared< cv::GoodFeaturesToTrackDetector >( 10, 0.01, 20, 7, true ) ),
                                                                           m_pExtractor( std::make_shared< cv::BriefDescriptorExtractor >( 64 ) ),
                                                                           m_pMatcher( std::make_shared< cv::BFMatcher >( cv::NORM_HAMMING ) ),
                                                                           m_dMatchingDistanceCutoffTriangulation( 40.0 ),
                                                                           m_dMatchingDistanceCutoffPoseOptimization( 30.0 ),
                                                                           m_dMatchingDistanceCutoffTracking( 20.0 ),

                                                                           /*//ds BRISK (calibrated 2015-06-09)
                                                                           //m_uKeyPointSize( 8 ),
                                                                           m_pDetector( std::make_shared< cv::BRISK >( 40, 2, 1.1 ) ), //std::make_shared< cv::GoodFeaturesToTrackDetector >( 100, 0.01, 20.0, m_uKeyPointSize, true ) ),
                                                                           m_pExtractor( std::make_shared< cv::BRISK >( 40, 2, 1.1 ) ),
                                                                           m_pMatcher( std::make_shared< cv::BFMatcher >( ) ),
                                                                           m_dMatchingDistanceCutoffTriangulation( 450.0 ),
                                                                           m_dMatchingDistanceCutoffTracking( 400.0 ),*/

                                                                           /*//ds SURF
                                                                           m_pDetector( std::make_shared< cv::SurfFeatureDetector >( 5000 ) ),
                                                                           m_pExtractor( std::make_shared< cv::SurfDescriptorExtractor >( 5000 ) ),
                                                                           m_pMatcher( std::make_shared< cv::FlannBasedMatcher >( ) ),
                                                                           m_fMatchingDistanceCutoffTriangulation( 0.25 ),
                                                                           m_fMatchingDistanceCutoffTracking( 0.25 ),*/

                                                                           /*//ds SIFT
                                                                           //m_pDetector( std::make_shared< cv::SiftFeatureDetector >( 100 ) ),
                                                                           m_pDetector( std::make_shared< cv::GoodFeaturesToTrackDetector >( 50, 0.01, 18.0, 7, true ) ),
                                                                           m_pExtractor( std::make_shared< cv::SiftDescriptorExtractor >( 50 ) ),
                                                                           m_pMatcher( std::make_shared< cv::FlannBasedMatcher >( ) ),
                                                                           m_fMatchingDistanceCutoffTriangulation( 250.0 ),
                                                                           m_fMatchingDistanceCutoffTracking( 200.0 ),*/

                                                                           m_uMaximumFailedSubsequentTrackingsPerLandmark( 5 ),
                                                                           m_uVisibleLandmarksMinimum( 1 ),
                                                                           m_dMinimumDepthMeters( 0.05 ),
                                                                           m_dMaximumDepthMeters( 100.0 ),


                                                                           //m_cDetector( m_pCameraLEFT, m_pDetector ),
                                                                           m_pTriangulator( std::make_shared< CTriangulator >( m_pCameraSTEREO, m_pExtractor, m_pMatcher, m_dMatchingDistanceCutoffTriangulation ) ),
                                                                           m_cMatcherEpipolar( m_pTriangulator, 0, m_dMinimumDepthMeters, m_dMaximumDepthMeters, m_dMatchingDistanceCutoffPoseOptimization, m_dMatchingDistanceCutoffTracking, m_uMaximumFailedSubsequentTrackingsPerLandmark ),

                                                                           m_uAvailableLandmarkID( 0 ),
                                                                           m_vecLandmarks( std::make_shared< std::vector< CLandmark* > >( ) ),
                                                                           m_uNumberofLastVisibleLandmarks( 0 ),

                                                                           m_eMode( p_eMode ),
                                                                           m_bIsShutdownRequested( false ),
                                                                           m_dFrequencyPlaybackHz( p_uFrequencyPlaybackHz ),
                                                                           m_uFrequencyPlaybackDeltaHz( 50 ),
                                                                           m_iPlaybackSpeedupCounter( 0 ),
                                                                           m_cRandomGenerator( 1337 ),

                                                                           m_uOffsetTrajectoryU( 180 ),
                                                                           m_uOffsetTrajectoryV( 360 ),
                                                                           m_uTimingToken( 0 ),
                                                                           m_uFramesCurrentCycle( 0 ),
                                                                           m_dPreviousFrameRate( 0.0 )
{
    //ds debug logging
    m_pFileLandmarkCreation = std::fopen( "/home/dominik/workspace_catkin/src/vi_mapper/logs/landmarks_creation.txt", "w" );
    m_pFileLandmarkFinal    = std::fopen( "/home/dominik/workspace_catkin/src/vi_mapper/logs/landmarks_final.txt", "w" );
    m_pFileTrajectory       = std::fopen( "/home/dominik/workspace_catkin/src/vi_mapper/logs/trajectory.txt", "w" );

    //ds dump file format
    std::fprintf( m_pFileLandmarkCreation, "FRAME | ID_LANDMARK |      X      Y      Z :  DEPTH | U_LEFT V_LEFT | U_RIGHT V_RIGHT | KEYPOINT_SIZE\n" );
    std::fprintf( m_pFileLandmarkFinal, "ID_LANDMARK | X_INITIAL Y_INITIAL Z_INITIAL | X_FINAL Y_FINAL Z_FINAL | DELTA_X DELTA_Y DELTA_Z DELTA_TOTAL | MEASUREMENTS | CALIBRATIONS | MEAN_X MEAN_Y MEAN_Z\n" );
    std::fprintf( m_pFileTrajectory, "ID_FRAME |      X      Y      Z | QUAT_X QUAT_Y QUAT_Z QUAT_W\n" );

    //ds initialize reference frames with black images
    m_matDisplayLowerReference = cv::Mat::zeros( m_pCameraSTEREO->m_uPixelHeight, 2*m_pCameraSTEREO->m_uPixelWidth, CV_8UC3 );

    //ds trajectory maps
    m_matTrajectoryXY = cv::Mat( 720, 720, CV_8UC3, CColorCodeBGR( 255, 255, 255 ) );
    m_matTrajectoryZ = cv::Mat( 350, 1500, CV_8UC3, CColorCodeBGR( 255, 255, 255 ) );

    //ds draw meters grid
    for( uint32_t u = 0; u < 720; u += 10 )
    {
        cv::line( m_matTrajectoryXY, cv::Point( u, 0 ),cv::Point( u, 720 ), CColorCodeBGR( 175, 175, 175 ) );
        cv::line( m_matTrajectoryXY, cv::Point( 0, u ),cv::Point( 720, u ), CColorCodeBGR( 175, 175, 175 ) );
    }

    for( uint32_t x = 0; x < 1500; x += 10 )
    {
        cv::line( m_matTrajectoryZ, cv::Point( x, 0 ),cv::Point( x, 350 ), CColorCodeBGR( 175, 175, 175 ) );
    }
    for( uint32_t y = 0; y < 350; y += 10 )
    {
        cv::line( m_matTrajectoryZ, cv::Point( 0, y ),cv::Point( 1500, y ), CColorCodeBGR( 175, 175, 175 ) );
    }

    //ds initialize the window
    cv::namedWindow( "stereo matching", cv::WINDOW_AUTOSIZE );

    CLogger::openBox( );
    std::printf( "<CTrackerStereo>(CTrackerStereo) feature detector: %s\n", m_pDetector->name( ).c_str( ) );
    std::printf( "<CTrackerStereo>(CTrackerStereo) descriptor extractor: %s\n", m_pExtractor->name( ).c_str( ) );
    std::printf( "<CTrackerStereo>(CTrackerStereo) descriptor matcher: %s\n", m_pMatcher->name( ).c_str( ) );
    std::printf( "<CTrackerStereo>(CTrackerStereo) descriptor size: %i bytes\n", m_pExtractor->descriptorSize( ) );
    std::printf( "<CTrackerStereo>(CTrackerStereo) instance allocated\n" );
    CLogger::closeBox( );
}

CTrackerStereo::~CTrackerStereo( )
{
    //ds free all landmarks
    for( const CLandmark* pLandmark: *m_vecLandmarks )
    {
        //ds compute errors
        const double dErrorX = std::fabs( ( pLandmark->vecPointXYZOptimized.x( )-pLandmark->vecPointXYZInitial.x( ) )/( 1.0+std::fabs( pLandmark->vecPointXYZInitial.x( ) ) ) );
        const double dErrorY = std::fabs( ( pLandmark->vecPointXYZOptimized.y( )-pLandmark->vecPointXYZInitial.y( ) )/( 1.0+std::fabs( pLandmark->vecPointXYZInitial.y( ) ) ) );
        const double dErrorZ = std::fabs( ( pLandmark->vecPointXYZOptimized.z( )-pLandmark->vecPointXYZInitial.z( ) )/( 1.0+std::fabs( pLandmark->vecPointXYZInitial.z( ) ) ) );
        const double dErrorTotal = dErrorX + dErrorY + dErrorZ;

        //ds write final state to file before deleting
        std::fprintf( m_pFileLandmarkFinal, "     %06lu |    %6.2f    %6.2f    %6.2f |  %6.2f  %6.2f  %6.2f |   %5.2f   %5.2f   %5.2f       %5.2f |       %06lu |       %06u | %6.2f %6.2f %6.2f\n", pLandmark->uID,
                                                                              pLandmark->vecPointXYZInitial.x( ),
                                                                              pLandmark->vecPointXYZInitial.y( ),
                                                                              pLandmark->vecPointXYZInitial.z( ),
                                                                              pLandmark->vecPointXYZOptimized.x( ),
                                                                              pLandmark->vecPointXYZOptimized.y( ),
                                                                              pLandmark->vecPointXYZOptimized.z( ),
                                                                              dErrorX,
                                                                              dErrorY,
                                                                              dErrorZ,
                                                                              dErrorTotal,
                                                                              pLandmark->getNumberOfMeasurements( ),
                                                                              pLandmark->uOptimizationsSuccessful,
                                                                              pLandmark->vecPointXYZMean.x( ),
                                                                              pLandmark->vecPointXYZMean.y( ),
                                                                              pLandmark->vecPointXYZMean.z( ) );

        delete pLandmark;
    }

    //ds debug
    std::fclose( m_pFileLandmarkCreation );
    std::fclose( m_pFileLandmarkFinal );
    std::fclose( m_pFileTrajectory );

    std::printf( "<CTrackerStereo>(~CTrackerStereo) instance deallocated\n" );
}

void CTrackerStereo::receivevDataVIWithPose( const std::shared_ptr< txt_io::PinholeImageMessage > p_pImageLEFT,
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
    const Eigen::Isometry3d matTransformationLEFTToWORLD( matTransformationIMUToWORLD*m_pCameraLEFT->m_matTransformationCAMERAtoIMU );

    //ds process images
    _trackLandmarks( matPreprocessedLEFT, matPreprocessedRIGHT, matTransformationLEFTToWORLD, p_cIMU.getAngularVelocity( ), p_cIMU.getLinearAcceleration( ) );

    //ds flush all output
    std::fflush( stdout );
}

void CTrackerStereo::_trackLandmarks( const cv::Mat& p_matImageLEFT,
                                      const cv::Mat& p_matImageRIGHT,
                                      const Eigen::Isometry3d& p_matTransformationLEFTtoWORLD,
                                      const Eigen::Vector3d& p_vecAngularVelocity,
                                      const Eigen::Vector3d& p_vecLinearAcceleration )
{
    //ds current translation
    const CPoint3DWORLD vecTranslationCurrent( p_matTransformationLEFTtoWORLD.translation( ) );
    const cv::Point2d ptPositionXY( vecTranslationCurrent.x( ), vecTranslationCurrent.y( ) );

    //ds draw position on trajectory mat
    cv::circle( m_matTrajectoryXY, cv::Point2d( m_uOffsetTrajectoryU+ptPositionXY.x*10, m_uOffsetTrajectoryV-ptPositionXY.y*10 ), 2, CColorCodeBGR( 175, 175, 175 ), -1 );

    //ds get images into triple channel mats (display only)
    cv::Mat matDisplayLEFT;
    cv::Mat matDisplayRIGHT;

    //ds get images to triple channel for colored display
    cv::cvtColor( p_matImageLEFT, matDisplayLEFT, cv::COLOR_GRAY2BGR );
    cv::cvtColor( p_matImageRIGHT, matDisplayRIGHT, cv::COLOR_GRAY2BGR );

    //ds get clean copies
    const cv::Mat matDisplayLEFTClean( matDisplayLEFT.clone( ) );
    const cv::Mat matDisplayRIGHTClean( matDisplayRIGHT.clone( ) );

    const double dDeltaIMUAngular( std::fabs( p_vecAngularVelocity.x( ) ) + std::fabs( p_vecAngularVelocity.y( ) ) +std::fabs( p_vecAngularVelocity.z( ) ) );
    //const double dDeltaIMULinear( std::fabs( p_vecLinearAcceleration(0) )+std::fabs( p_vecLinearAcceleration(1) )+std::fabs( p_vecLinearAcceleration(2) ) );

    //ds epipolar base line length
    const uint32_t uEpipolarBaseLineLength( 10+15*dDeltaIMUAngular );



    //ds get currently visible landmarks
    const std::shared_ptr< std::vector< const CMeasurementLandmark* > > vecVisibleLandmarks( m_cMatcherEpipolar.getVisibleLandmarksEssentialOptimized( m_uFrameCount, matDisplayLEFT, matDisplayRIGHT, p_matImageLEFT, p_matImageRIGHT, p_matTransformationLEFTtoWORLD, uEpipolarBaseLineLength, m_matTrajectoryXY ) );
    m_uNumberofLastVisibleLandmarks = vecVisibleLandmarks->size( );

    //ds add to data structure if delta is sufficiently high
    if( m_dTranslationDeltaForMAPMeters < ( vecTranslationCurrent-m_vecTranslationLast ).squaredNorm( ) && m_uVisibleLandmarksMinimum <= m_uNumberofLastVisibleLandmarks )
    {
        m_vecTranslationLast = vecTranslationCurrent;
        m_vecLogMeasurementPoints.push_back( CKeyFrame( p_matTransformationLEFTtoWORLD, p_vecLinearAcceleration.normalized( ), vecVisibleLandmarks ) );
        //std::printf( "<CTrackerStereo>(_trackLandmarks) stashed measurement %lu with landmarks (%lu)\n", m_vecLogMeasurementPoints.size( ), vecVisibleLandmarks->size( ) );
    }

    //ds check if we have to detect new landmarks
    if( m_uVisibleLandmarksMinimum > m_uNumberofLastVisibleLandmarks )
    {
        //ds clean the lower display
        cv::hconcat( matDisplayLEFTClean, matDisplayRIGHTClean, m_matDisplayLowerReference );

        //ds compute mask to avoid detecting similar features
        cv::Mat matMaskDetection( cv::Mat( m_pCameraSTEREO->m_uPixelHeight, m_pCameraSTEREO->m_uPixelWidth, CV_8UC1, cv::Scalar ( 255 ) ) );

        //ds draw black circles for existing landmark positions into the mask (avoid redetection of landmarks)
        for( const CMeasurementLandmark* cLandmarkInCameraFrame: *vecVisibleLandmarks )
        {
            cv::circle( matMaskDetection, cLandmarkInCameraFrame->ptUVLEFT, 10, cv::Scalar( 0 ), -1 );
        }

        //ds detect landmarks
        const std::shared_ptr< std::vector< CLandmark* > > vecNewLandmarks( _getNewLandmarksTriangulated( m_uFrameCount, m_matDisplayLowerReference, p_matImageLEFT, p_matImageRIGHT, p_matTransformationLEFTtoWORLD, matMaskDetection ) );

        //ds add to permanent reference holder
        m_vecLandmarks->insert( m_vecLandmarks->end( ), vecNewLandmarks->begin( ), vecNewLandmarks->end( ) );

        //ds add this measurement point to the epipolar matcher
        m_cMatcherEpipolar.addDetectionPoint( p_matTransformationLEFTtoWORLD, vecNewLandmarks );

        cv::circle( m_matTrajectoryXY, cv::Point2d( m_uOffsetTrajectoryU+ptPositionXY.x*10, m_uOffsetTrajectoryV-ptPositionXY.y*10 ), 20, CColorCodeBGR( 0, 255, 0 ), 1 );
    }



    //ds info
    //cv::circle( m_matTrajectoryZ, cv::Point2d( m_uFrameCount, 175-vecVisibleLandmarks->size( )*2 ), 1, cv::Scalar( 255, 0, 0 ), -1 );
    cv::circle( m_matTrajectoryZ, cv::Point2d( m_uFrameCount, 350-uEpipolarBaseLineLength*5 ), 3, CColorCodeBGR( 0, 0, 0 ), -1 );
    //cv::circle( m_matTrajectoryZ, cv::Point2d( m_uFrameCount, 175-p_vecAngularVelocity(0)*100 ), 3, CColorCodeBGR( 255, 0, 0 ), -1 );
    //cv::circle( m_matTrajectoryZ, cv::Point2d( m_uFrameCount, 175-p_vecAngularVelocity(1)*100 ), 3, CColorCodeBGR( 0, 255, 0 ), -1 );
    //cv::circle( m_matTrajectoryZ, cv::Point2d( m_uFrameCount, 175-p_vecAngularVelocity(2)*100 ), 3, CColorCodeBGR( 0, 0, 255 ), -1 );
    //cv::circle( m_matTrajectoryZ, cv::Point2d( m_uFrameCount, 350-dDeltaIMU*5 ), 1, CColorCodeBGR( 255, 0, 0 ), -1 );
    //cv::circle( m_matTrajectoryZ, cv::Point2d( m_uFrameCount, 350-dDeltaIMU*5 ), 1, CColorCodeBGR( 255, 0, 0 ), -1 );
    //cv::circle( m_matTrajectoryZ, cv::Point2d( m_uFrameCount, 350-dDeltaIMU*5 ), 1, CColorCodeBGR( 255, 0, 0 ), -1 );

    cv::Mat matTrajectoryXYTemporary( m_matTrajectoryXY.clone( ) );
    cv::circle( matTrajectoryXYTemporary, cv::Point2d( m_uOffsetTrajectoryU+ptPositionXY.x*10, m_uOffsetTrajectoryV-ptPositionXY.y*10 ), 2, CColorCodeBGR( 255, 0, 0 ), -1 );

    //ds build display mat
    _drawInfoBox( matDisplayLEFT );
    cv::Mat matDisplayUpper = cv::Mat( m_pCameraSTEREO->m_uPixelHeight, 2*m_pCameraSTEREO->m_uPixelWidth, CV_8UC3 );
    cv::hconcat( matDisplayLEFT, matDisplayRIGHT, matDisplayUpper );
    cv::Mat matDisplayComplete = cv::Mat( 2*m_pCameraSTEREO->m_uPixelHeight, 2*m_pCameraSTEREO->m_uPixelWidth, CV_8UC3 );
    cv::vconcat( matDisplayUpper, m_matDisplayLowerReference, matDisplayComplete );

    //ds display
    cv::imshow( "stereo matching", matDisplayComplete );
    cv::imshow( "trajectory (x,y)", matTrajectoryXYTemporary );
    //cv::imshow( "some stuff", m_matTrajectoryZ );

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

const std::shared_ptr< std::vector< CLandmark* > > CTrackerStereo::_getNewLandmarksTriangulated( const uint64_t& p_uFrame,
                                                                                     cv::Mat& p_matDisplay,
                                                                                     const cv::Mat& p_matImageLEFT,
                                                                                     const cv::Mat& p_matImageRIGHT,
                                                                                     const Eigen::Isometry3d& p_matTransformationLEFTtoWORLD,
                                                                                     const cv::Mat& p_matMask )
{
    //ds precompute extrinsics
    const Eigen::Isometry3d matTransformationWORLDtoLEFT( p_matTransformationLEFTtoWORLD.inverse( ) );
    const Eigen::Matrix3d matKRotation( m_pCameraLEFT->m_matIntrinsic*matTransformationWORLDtoLEFT.linear( ) );
    const Eigen::Vector3d vecCameraPosition( matTransformationWORLDtoLEFT.translation( ) );
    const Eigen::Vector3d vecKTranslation( m_pCameraLEFT->m_matIntrinsic*vecCameraPosition );
    const MatrixProjection matProjectionWORLDtoLEFT( m_pCameraLEFT->m_matProjection*matTransformationWORLDtoLEFT.matrix( ) );

    //ds solution holder
    std::shared_ptr< std::vector< CLandmark* > > vecNewLandmarks( std::make_shared< std::vector< CLandmark* > >( ) );

    //ds detect new keypoints
    std::vector< cv::KeyPoint > vecKeyPoints;
    m_pDetector->detect( p_matImageLEFT, vecKeyPoints, p_matMask );

    //ds compute descriptors for the keypoints
    CDescriptor matReferenceDescriptors;
    m_pExtractor->compute( p_matImageLEFT, vecKeyPoints, matReferenceDescriptors );

    //ds rightside keypoints buffer for descriptor computation
    std::vector< cv::KeyPoint > vecKeyPointRIGHT( 1 );

    //ds process the keypoints and see if we can use them as landmarks
    for( uint32_t u = 0; u < vecKeyPoints.size( ); ++u )
    {
        //ds current points
        const cv::KeyPoint cKeyPointLEFT( vecKeyPoints[u] );
        const cv::Point2f ptLandmarkLEFT( cKeyPointLEFT.pt );
        const CDescriptor& matDescriptorLEFT( matReferenceDescriptors.row(u) );

        try
        {
            //ds triangulate the point
            const CPoint3DCAMERA vecPointTriangulatedLEFT( m_pTriangulator->getPointTriangulatedLimited( p_matImageRIGHT, cKeyPointLEFT, matDescriptorLEFT ) );
            //const CPoint3DInCameraFrame vecPointTriangulatedRIGHT( m_pCameraSTEREO->m_matTransformLEFTtoRIGHT*vecPointTriangulatedLEFT );

            const double& dDepthMeters( vecPointTriangulatedLEFT.z( ) );

            //ds check if point is in front of camera an not more than a defined distance away
            if( m_dMinimumDepthMeters < dDepthMeters && m_dMaximumDepthMeters > dDepthMeters )
            {
                //ds compute triangulated point in world frame
                const CPoint3DWORLD vecPointTriangulatedWORLD( p_matTransformationLEFTtoWORLD*vecPointTriangulatedLEFT );

                //ds draw reprojection of triangulation
                cv::Point2d ptLandmarkRIGHT( m_pCameraRIGHT->getProjection( vecPointTriangulatedLEFT ) );

                //ds enforce epipolar constraint TODO integrate epipolar error
                const double dEpipolarError( ptLandmarkRIGHT.y-ptLandmarkLEFT.y );
                if( 0.1 < dEpipolarError )
                {
                    std::printf( "<CTrackerStereo>(_getNewLandmarksTriangulated) landmark [%lu] epipolar error: %f\n", m_uAvailableLandmarkID, dEpipolarError );
                }
                ptLandmarkRIGHT.y = ptLandmarkLEFT.y;

                //ds compute reference descriptor on right side as well
                vecKeyPointRIGHT[0] = cv::KeyPoint( ptLandmarkRIGHT, cKeyPointLEFT.size );
                CDescriptor matDescriptorRIGHT;
                m_pExtractor->compute( p_matImageRIGHT, vecKeyPointRIGHT, matDescriptorRIGHT );

                //ds allocate a new landmark and add the current position
                CLandmark* cLandmark( new CLandmark( m_uAvailableLandmarkID,
                                                     matDescriptorLEFT,
                                                     matDescriptorRIGHT,
                                                     cKeyPointLEFT.size,
                                                     vecPointTriangulatedWORLD,
                                                     m_pCameraLEFT->getNormalHomogenized( ptLandmarkLEFT ),
                                                     ptLandmarkLEFT,
                                                     ptLandmarkRIGHT,
                                                     vecPointTriangulatedLEFT,
                                                     vecCameraPosition,
                                                     matProjectionWORLDtoLEFT,
                                                     p_uFrame ) );

                //ds log creation
                std::fprintf( m_pFileLandmarkCreation, " %04lu |      %06lu | %6.2f %6.2f %6.2f : %6.2f | %6.2f %6.2f |  %6.2f  %6.2f |        %6.2f\n", p_uFrame,
                                                                                              cLandmark->uID,
                                                                                              cLandmark->vecPointXYZOptimized.x( ),
                                                                                              cLandmark->vecPointXYZOptimized.y( ),
                                                                                              cLandmark->vecPointXYZOptimized.z( ),
                                                                                              dDepthMeters,
                                                                                              ptLandmarkLEFT.x,
                                                                                              ptLandmarkLEFT.y,
                                                                                              ptLandmarkRIGHT.x,
                                                                                              ptLandmarkRIGHT.y,
                                                                                              cLandmark->dKeyPointSize );

                //ds add to newly detected
                vecNewLandmarks->push_back( cLandmark );

                //ds next landmark id
                ++m_uAvailableLandmarkID;

                //ds draw detected point
                cv::line( p_matDisplay, ptLandmarkLEFT, cv::Point2f( ptLandmarkLEFT.x+m_pCameraSTEREO->m_uPixelWidth, ptLandmarkLEFT.y ), CColorCodeBGR( 175, 175, 175 ) );
                cv::circle( p_matDisplay, ptLandmarkLEFT, 2, CColorCodeBGR( 0, 255, 0 ), -1 );
                //cv::circle( p_matDisplay, ptLandmarkLEFT, cLandmark->dKeyPointSize, CColorCodeBGR( 255, 0, 0 ), 1 );
                cv::putText( p_matDisplay, std::to_string( cLandmark->uID ) , cv::Point2d( ptLandmarkLEFT.x+cLandmark->dKeyPointSize, ptLandmarkLEFT.y+cLandmark->dKeyPointSize ), cv::FONT_HERSHEY_PLAIN, 0.5, CColorCodeBGR( 0, 0, 255 ) );

                //ds draw landmark in world (2d)
                cv::circle( m_matTrajectoryXY, cv::Point2d( m_uOffsetTrajectoryU+vecPointTriangulatedWORLD.x( )*10, m_uOffsetTrajectoryV-vecPointTriangulatedWORLD.y( )*10 ), 3, CColorCodeBGR( 0, 165, 255 ), -1 );

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
        catch( const CExceptionNoMatchFound& p_cException )
        {
            cv::circle( p_matDisplay, ptLandmarkLEFT, 2, CColorCodeBGR( 0, 0, 255 ), -1 );
            //cv::circle( p_matDisplay, ptLandmarkLEFT, cKeyPoint.size, CColorCodeBGR( 0, 0, 255 ) );

            std::printf( "<CTrackerStereo>(_getNewLandmarks) could not find match for keypoint (%s)\n", p_cException.what( ) );
        }
    }

    std::printf( "<CTrackerStereo>(_getNewLandmarks) added new landmarks: %lu/%lu\n", vecNewLandmarks->size( ), vecKeyPoints.size( ) );

    //ds return found landmarks
    return vecNewLandmarks;
}

void CTrackerStereo::_shutDown( )
{
    m_bIsShutdownRequested = true;
    std::printf( "<CTrackerStereo>(_shutDown) termination requested, <CTrackerStereo> disabled\n" );
}

void CTrackerStereo::_speedUp( )
{
    ++m_iPlaybackSpeedupCounter;
    m_dFrequencyPlaybackHz += std::pow( 2, m_iPlaybackSpeedupCounter )*m_uFrequencyPlaybackDeltaHz;
    std::printf( "<CTrackerStereo>(_speedUp) increased playback speed to: %.0f Hz \n", m_dFrequencyPlaybackHz );
}

void CTrackerStereo::_slowDown( )
{
    m_dFrequencyPlaybackHz -= std::pow( 2, m_iPlaybackSpeedupCounter )*m_uFrequencyPlaybackDeltaHz;

    //ds 12 fps minimum (one of 2 images 10 imu messages and 1 pose: 13-1)
    if( 1 < m_dFrequencyPlaybackHz )
    {
        std::printf( "<CTrackerStereo>(_slowDown)  reduced playback speed to: %.0f Hz \n", m_dFrequencyPlaybackHz );
        --m_iPlaybackSpeedupCounter;
    }
    else
    {
        m_dFrequencyPlaybackHz = 1;
        std::printf( "<CTrackerStereo>(_slowDown)  reduced playback speed to: %.0f Hz \n", m_dFrequencyPlaybackHz );
    }
}

void CTrackerStereo::_updateFrameRateForInfoBox( const uint32_t& p_uFrameProbeRange )
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

void CTrackerStereo::_drawInfoBox( cv::Mat& p_matDisplay ) const
{
    char chBuffer[100];

    switch( m_eMode )
    {
        case ePlaybackStepwise:
        {
            std::snprintf( chBuffer, 100, "[%04lu] STEPWISE | LANDMARKS: %2lu(%4lu) | KEYFRAMES: %1lu(%2lu) | POSES: %2lu", m_uFrameCount, m_uNumberofLastVisibleLandmarks, m_vecLandmarks->size( ), m_cMatcherEpipolar.getNumberOfDetectionPointsActive( ), m_cMatcherEpipolar.getNumberOfDetectionPointsTotal( ), m_vecLogMeasurementPoints.size( ) );
            break;
        }
        case ePlaybackBenchmark:
        {
            std::snprintf( chBuffer, 100, "[%04lu] FPS %4.1f(BENCHMARK) | LANDMARKS: %2lu(%4lu) | KEYFRAMES: %1lu(%2lu) | POSES: %2lu", m_uFrameCount, m_dPreviousFrameRate, m_uNumberofLastVisibleLandmarks, m_vecLandmarks->size( ), m_cMatcherEpipolar.getNumberOfDetectionPointsActive( ), m_cMatcherEpipolar.getNumberOfDetectionPointsTotal( ), m_vecLogMeasurementPoints.size( ) );
            break;
        }
        case ePlaybackInteractive:
        {
            std::snprintf( chBuffer, 100, "[%04lu] FPS %4.1f(%4.2f Hz) | LANDMARKS: %0lu(%4lu) | KEYFRAMES: %1lu(%2lu) | POSES: %2lu", m_uFrameCount, m_dPreviousFrameRate, m_dFrequencyPlaybackHz, m_uNumberofLastVisibleLandmarks, m_vecLandmarks->size( ), m_cMatcherEpipolar.getNumberOfDetectionPointsActive( ), m_cMatcherEpipolar.getNumberOfDetectionPointsTotal( ), m_vecLogMeasurementPoints.size( ) );
            break;
        }
        default:
        {
            std::printf( "<CTrackerStereo>(_drawInfoBox) unsupported playback mode, no info box displayed\n" );
            break;
        }
    }


    p_matDisplay( cv::Rect( 0, 0, m_pCameraLEFT->m_iWidthPixel, 17 ) ).setTo( CColorCodeBGR( 0, 0, 0 ) );
    cv::putText( p_matDisplay, chBuffer , cv::Point2i( 2, 12 ), cv::FONT_HERSHEY_PLAIN, 0.8, CColorCodeBGR( 0, 0, 255 ) );
}
