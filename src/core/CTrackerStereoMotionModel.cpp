#include "CTrackerStereoMotionModel.h"

#include <opencv/highgui.h>
#include <opencv2/features2d/features2d.hpp>

#include "configuration/CConfigurationCamera.h"
#include "configuration/CConfigurationOpenCV.h"
#include "types/CIMUInterpolator.h"
#include "exceptions/CExceptionPoseOptimization.h"
#include "exceptions/CExceptionNoMatchFound.h"
#include "utility/CCloudStreamer.h"

CTrackerStereoMotionModel::CTrackerStereoMotionModel( const EPlaybackMode& p_eMode,
                                                      const uint32_t& p_uWaitKeyTimeoutMS ): m_uWaitKeyTimeoutMS( p_uWaitKeyTimeoutMS ),
                                                                           m_pCameraLEFT( std::make_shared< CPinholeCamera >( CConfigurationCamera::LEFT::cPinholeCamera ) ),
                                                                           m_pCameraRIGHT( std::make_shared< CPinholeCamera >( CConfigurationCamera::RIGHT::cPinholeCamera ) ),
                                                                           m_pCameraSTEREO( std::make_shared< CStereoCamera >( m_pCameraLEFT, m_pCameraRIGHT ) ),

                                                                           m_matTransformationWORLDtoLEFTLAST( Eigen::Matrix4d( CConfigurationCamera::matTransformationIntialWORLDtoLEFT ).transpose( ) ),
                                                                           m_matTransformationLEFTLASTtoLEFTNOW( Eigen::Matrix4d::Identity( ) ),
                                                                           m_vecVelocityAngularFilteredLAST( 0.0, 0.0, 0.0 ),
                                                                           m_vecLinearAccelerationFilteredLAST( 0.0, 0.0, 0.0 ),
                                                                           m_vecTranslationLastKeyFrame( -1.0, -1.0, -1.0 ),
                                                                           m_dTranslationDeltaForKeyFrameMetersSquaredNorm( 0.1 ),
                                                                           m_vecPositionCurrent( 0.0, 0.0, 0.0 ),

                                                                           /*//ds ORB
                                                                           m_pDetector( std::make_shared< cv::OrbFeatureDetector >( 100, 1.2, 8, 25, 0, 2, cv::ORB::FAST_SCORE, 25 ) ),
                                                                           m_pExtractor( std::make_shared< cv::OrbDescriptorExtractor >( 100, 1.2, 8, 25, 0, 2, cv::ORB::FAST_SCORE, 25 ) ),
                                                                           m_pMatcher( std::make_shared< cv::BFMatcher >( cv::NORM_HAMMING ) ),
                                                                           m_dMatchingDistanceCutoffTriangulation( 100.0 ),
                                                                           m_dMatchingDistanceCutoffTracking( 350.0 ),*/

                                                                           //ds BRIEF (calibrated 2015-05-31)
                                                                           // m_uKeyPointSize( 7 ),
                                                                           m_pDetector( std::make_shared< cv::GoodFeaturesToTrackDetector >( 200, 0.01, 7.0, 7, true ) ),
                                                                           m_pExtractor( std::make_shared< cv::BriefDescriptorExtractor >( 64 ) ),
                                                                           m_pMatcher( std::make_shared< cv::BFMatcher >( cv::NORM_HAMMING ) ),
                                                                           m_dMatchingDistanceCutoffTriangulation( 100.0 ),
                                                                           m_dMatchingDistanceCutoffPoseOptimization( 60.0 ),
                                                                           m_dMatchingDistanceCutoffTracking( 40.0 ),

                                                                           m_uMaximumFailedSubsequentTrackingsPerLandmark( 5 ),
                                                                           m_uVisibleLandmarksMinimum( 100 ),
                                                                           m_dMinimumDepthMeters( 0.05 ),
                                                                           m_dMaximumDepthMeters( 100.0 ),

                                                                           m_pTriangulator( std::make_shared< CTriangulator >( m_pCameraSTEREO, m_pExtractor, m_pMatcher, m_dMatchingDistanceCutoffTriangulation ) ),
                                                                           m_cMatcherEpipolar( m_pTriangulator, m_pDetector, m_dMinimumDepthMeters, m_dMaximumDepthMeters, m_dMatchingDistanceCutoffPoseOptimization, m_dMatchingDistanceCutoffTracking, m_uMaximumFailedSubsequentTrackingsPerLandmark ),

                                                                           m_vecLandmarks( std::make_shared< std::vector< CLandmark* > >( ) ),

                                                                           m_eMode( p_eMode )
{
    //ds set opencv parallelization threads
    cv::setNumThreads( 0 );

    //ds initialize reference frames with black images
    m_matDisplayLowerReference = cv::Mat::zeros( m_pCameraSTEREO->m_uPixelHeight, 2*m_pCameraSTEREO->m_uPixelWidth, CV_8UC3 );

    //ds initialize the window
    cv::namedWindow( "stereo matching", cv::WINDOW_AUTOSIZE );

    CLogger::openBox( );
    std::printf( "<CTrackerStereoMotionModel>(CTrackerStereoMotionModel) <OpenCV> available CPUs: %i\n", cv::getNumberOfCPUs( ) );
    std::printf( "<CTrackerStereoMotionModel>(CTrackerStereoMotionModel) <OpenCV> available threads: %i\n", cv::getNumThreads( ) );
    std::printf( "<CTrackerStereoMotionModel>(CTrackerStereoMotionModel) feature detector: %s\n", m_pDetector->name( ).c_str( ) );
    std::printf( "<CTrackerStereoMotionModel>(CTrackerStereoMotionModel) descriptor extractor: %s\n", m_pExtractor->name( ).c_str( ) );
    std::printf( "<CTrackerStereoMotionModel>(CTrackerStereoMotionModel) descriptor matcher: %s\n", m_pMatcher->name( ).c_str( ) );
    std::printf( "<CTrackerStereoMotionModel>(CTrackerStereoMotionModel) descriptor size: %i bytes\n", m_pExtractor->descriptorSize( ) );
    std::printf( "<CTrackerStereoMotionModel>(CTrackerStereoMotionModel) minimum timestamp delta (IMU): %f\n", m_dMaximumDeltaTimestampSeconds );
    std::printf( "<CTrackerStereoMotionModel>(CTrackerStereoMotionModel) <Landmark> delta for optimization translation: %f\n", CLandmark::m_dDistanceDeltaForOptimizationMeters );
    std::printf( "<CTrackerStereoMotionModel>(CTrackerStereoMotionModel) <Landmark> delta for optimization rotation: %f\n", CLandmark::m_dAngleDeltaForOptimizationRadians );
    std::printf( "<CTrackerStereoMotionModel>(CTrackerStereoMotionModel) <Landmark> cap iterations: %u\n", CLandmark::m_uCapIterations );
    std::printf( "<CTrackerStereoMotionModel>(CTrackerStereoMotionModel) <Landmark> levenberg damping: %f\n", CLandmark::m_dLevenbergDamping );
    std::printf( "<CTrackerStereoMotionModel>(CTrackerStereoMotionModel) <Landmark> convergence delta: %f\n", CLandmark::m_dConvergenceDelta );
    std::printf( "<CTrackerStereoMotionModel>(CTrackerStereoMotionModel) <Landmark> kernel size: %f\n", CLandmark::m_dKernelMaximumError );
    std::printf( "<CTrackerStereoMotionModel>(CTrackerStereoMotionModel) instance allocated\n" );
    CLogger::closeBox( );
}

CTrackerStereoMotionModel::~CTrackerStereoMotionModel( )
{
    //ds free all landmarks
    for( const CLandmark* pLandmark: *m_vecLandmarks )
    {
        //ds write final state to file before deleting
        CLogger::CLogLandmarkFinal::addEntry( pLandmark );

        //ds save optimized landmarks to separate file
        if( CBridgeG2O::isOptimized( pLandmark ) && CBridgeG2O::isKeyFramed( pLandmark ) ){ CLogger::CLogLandmarkFinalOptimized::addEntry( pLandmark ); }

        delete pLandmark;
    }

    //ds close loggers
    CLogger::CLogLandmarkCreation::close( );
    CLogger::CLogLandmarkFinal::close( );
    CLogger::CLogLandmarkFinalOptimized::close( );
    CLogger::CLogTrajectory::close( );
    CLogger::CLogLinearAcceleration::close( );

    std::printf( "<CTrackerStereoMotionModel>(~CTrackerStereoMotionModel) instance deallocated\n" );
}

void CTrackerStereoMotionModel::receivevDataVI( const std::shared_ptr< txt_io::PinholeImageMessage > p_pImageLEFT,
                                                const std::shared_ptr< txt_io::PinholeImageMessage > p_pImageRIGHT,
                                                const std::shared_ptr< txt_io::CIMUMessage > p_pIMU )
{
    //ds preprocessed images
    cv::Mat matPreprocessedLEFT( p_pImageLEFT->image( ) );
    cv::Mat matPreprocessedRIGHT( p_pImageRIGHT->image( ) );

    //ds preprocess images
    cv::equalizeHist( p_pImageLEFT->image( ), matPreprocessedLEFT );
    cv::equalizeHist( p_pImageRIGHT->image( ), matPreprocessedRIGHT );
    m_pCameraSTEREO->undistortAndrectify( matPreprocessedLEFT, matPreprocessedRIGHT );

    //ds current timestamp
    const double dTimestampSeconds      = p_pIMU->timestamp( );
    const double dDeltaTimestampSeconds = dTimestampSeconds - m_dTimestampLASTSeconds;

    assert( 0 < dDeltaTimestampSeconds );

    //ds parallel transformation
    Eigen::Isometry3d matTransformationParallelLEFTLASTtoLEFTNOW( m_matTransformationLEFTLASTtoLEFTNOW );

    //ds compute total rotation
    const Eigen::Vector3d vecRotationTotal( m_vecVelocityAngularFilteredLAST*dDeltaTimestampSeconds );

    //ds if the delta is acceptable
    if( m_dMaximumDeltaTimestampSeconds > dDeltaTimestampSeconds )
    {
        //ds integrate imu input: overwrite rotation
        m_matTransformationLEFTLASTtoLEFTNOW.linear( ) = CMiniVisionToolbox::fromOrientationRodrigues( vecRotationTotal );

        //ds add acceleration
        m_matTransformationLEFTLASTtoLEFTNOW.translation( ) += 0.5*m_vecLinearAccelerationFilteredLAST*dDeltaTimestampSeconds*dDeltaTimestampSeconds;
    }
    else
    {
        //ds use full angular velocity
        std::printf( "<CTrackerStereoMotionModel>(receivevDataVI) using rotation-only IMU input, timestamp delta: %f \n", dDeltaTimestampSeconds );

        //ds integrate imu input: overwrite rotation only
        m_matTransformationLEFTLASTtoLEFTNOW.linear( ) = CMiniVisionToolbox::fromOrientationRodrigues( vecRotationTotal );
    }

    //ds process images (fed with IMU prior pose)
    _trackLandmarks( matPreprocessedLEFT,
                     matPreprocessedRIGHT,
                     m_matTransformationLEFTLASTtoLEFTNOW*m_matTransformationWORLDtoLEFTLAST,
                     matTransformationParallelLEFTLASTtoLEFTNOW*m_matTransformationWORLDtoLEFTLAST,
                     p_pIMU->getLinearAcceleration( ),
                     vecRotationTotal );

    //ds update references
    m_dTimestampLASTSeconds          = dTimestampSeconds;
    m_vecVelocityAngularFilteredLAST = m_pCameraLEFT->m_matRotationIMUtoCAMERA*CIMUInterpolator::getAngularVelocityFiltered( p_pIMU->getAngularVelocity( ) );
}

void CTrackerStereoMotionModel::_trackLandmarks( const cv::Mat& p_matImageLEFT,
                                                 const cv::Mat& p_matImageRIGHT,
                                                 const Eigen::Isometry3d& p_matTransformationEstimateWORLDtoLEFT,
                                                 const Eigen::Isometry3d& p_matTransformationEstimateParallelWORLDtoLEFT,
                                                 const CLinearAccelerationIMU& p_vecLinearAcceleration,
                                                 const Eigen::Vector3d& p_vecRotationTotal )
{
    //ds get images into triple channel mats (display only)
    cv::Mat matDisplayLEFT;
    cv::Mat matDisplayRIGHT;

    //ds get images to triple channel for colored display
    cv::cvtColor( p_matImageLEFT, matDisplayLEFT, cv::COLOR_GRAY2BGR );
    cv::cvtColor( p_matImageRIGHT, matDisplayRIGHT, cv::COLOR_GRAY2BGR );

    //ds get clean copies
    const cv::Mat matDisplayLEFTClean( matDisplayLEFT.clone( ) );
    const cv::Mat matDisplayRIGHTClean( matDisplayRIGHT.clone( ) );

    //ds compute motion scaling
    const double dMotionScaling = 1.0+100*p_vecRotationTotal.squaredNorm( );

    //ds refresh landmark states
    m_cMatcherEpipolar.resetVisibilityActiveLandmarks( );



    //ds initial transformation
    Eigen::Isometry3d matTransformationWORLDtoLEFT( p_matTransformationEstimateWORLDtoLEFT );

    try
    {
        //ds get the optimized pose
        matTransformationWORLDtoLEFT = m_cMatcherEpipolar.getPoseOptimizedSTEREO( m_uFrameCount, matDisplayLEFT, matDisplayRIGHT, p_matImageLEFT, p_matImageRIGHT, p_matTransformationEstimateWORLDtoLEFT, p_vecRotationTotal, dMotionScaling );
    }
    catch( const CExceptionPoseOptimization& p_cException )
    {
        std::printf( "<CTrackerStereoMotionModel>(_trackLandmarks) pose optimization failed: '%s' trying parallel damped transform\n", p_cException.what( ) );
        try
        {
            //ds get the optimized pose on damped motion
            matTransformationWORLDtoLEFT = m_cMatcherEpipolar.getPoseOptimizedSTEREO( m_uFrameCount, matDisplayLEFT, matDisplayRIGHT, p_matImageLEFT, p_matImageRIGHT, p_matTransformationEstimateParallelWORLDtoLEFT, p_vecRotationTotal, dMotionScaling );
        }
        catch( const CExceptionPoseOptimization& p_cException )
        {
            //ds stick to the original estimate
            std::printf( "<CTrackerStereoMotionModel>(_trackLandmarks) parallel pose optimization failed: '%s'\n", p_cException.what( ) );
            matTransformationWORLDtoLEFT = p_matTransformationEstimateWORLDtoLEFT;
        }
    }



    //ds respective camera transform
    const Eigen::Isometry3d matTransformationLEFTtoWORLD( matTransformationWORLDtoLEFT.inverse( ) );

    //ds estimate acceleration in current WORLD frame (necessary to filter for gravity)
    const Eigen::Isometry3d matTransformationIMUtoWORLD( matTransformationLEFTtoWORLD*m_pCameraLEFT->m_matTransformationIMUtoCAMERA );
    const CLinearAccelerationWORLD vecLinearAccelerationWORLD( matTransformationIMUtoWORLD.linear( )*p_vecLinearAcceleration );
    const CLinearAccelerationWORLD vecLinearAccelerationWORLDFiltered( CIMUInterpolator::getLinearAccelerationFiltered( vecLinearAccelerationWORLD ) );

    //ds update acceleration reference
    m_vecLinearAccelerationFilteredLAST = matTransformationWORLDtoLEFT.linear( )*vecLinearAccelerationWORLDFiltered;
    CLogger::CLogLinearAcceleration::addEntry( m_uFrameCount, vecLinearAccelerationWORLD, vecLinearAccelerationWORLDFiltered );

    //ds current translation
    m_vecPositionCurrent = matTransformationLEFTtoWORLD.translation( );
    CLogger::CLogTrajectory::addEntry( m_uFrameCount, m_vecPositionCurrent, Eigen::Quaterniond( matTransformationLEFTtoWORLD.linear( ) ) );


    //ds get currently visible landmarks (including landmarks already detected in the pose optimization)
    const std::shared_ptr< std::vector< const CMeasurementLandmark* > > vecVisibleLandmarks( m_cMatcherEpipolar.getVisibleLandmarksFundamental( matDisplayLEFT, matDisplayRIGHT, m_uFrameCount, p_matImageLEFT, p_matImageRIGHT, matTransformationLEFTtoWORLD, p_vecRotationTotal, dMotionScaling ) );

    //ds landmark lost since last
    const int32_t iLandmarksLost = m_uNumberofLastVisibleLandmarks-vecVisibleLandmarks->size( );

    //ds if we lose more than 75% landmarks in one frame
    if( 0.75 < static_cast< double >( iLandmarksLost )/m_uNumberofLastVisibleLandmarks )
    {
        std::printf( "<CTrackerStereoMotionModel>(_trackLandmarks) lost track (landmarks lost: %i), total delta: %f (%f %f %f)\n", iLandmarksLost, m_vecVelocityAngularFilteredLAST.squaredNorm( ), m_vecVelocityAngularFilteredLAST.x( ), m_vecVelocityAngularFilteredLAST.y( ), m_vecVelocityAngularFilteredLAST.z( ) );
        //cv::waitKey( 0 );
    }



    //ds update reference
    m_uNumberofLastVisibleLandmarks = vecVisibleLandmarks->size( );

    //ds draw the landmarks
    for( const CMeasurementLandmark* pMeasurement: *vecVisibleLandmarks )
    {
        //ds compute green brightness based on depth (further away -> darker)
        uint8_t uGreenValue = 255-std::sqrt( pMeasurement->vecPointXYZLEFT.z( ) )*20;
        cv::circle( matDisplayLEFT, pMeasurement->ptUVLEFT, 2, CColorCodeBGR( 0, uGreenValue, 0 ), -1 );
        cv::circle( matDisplayRIGHT, pMeasurement->ptUVRIGHT, 2, CColorCodeBGR( 0, uGreenValue, 0 ), -1 );

        //ds get 3d position in current camera frame
        const CPoint3DCAMERA vecXYZLEFT( matTransformationWORLDtoLEFT*pMeasurement->vecPointXYZWORLDOptimized );

        //ds also draw reprojections
        cv::circle( matDisplayLEFT, m_pCameraLEFT->getProjection( vecXYZLEFT ), 6, CColorCodeBGR( 0, uGreenValue, 0 ), 1 );
        cv::circle( matDisplayRIGHT, m_pCameraRIGHT->getProjection( vecXYZLEFT ), 6, CColorCodeBGR( 0, uGreenValue, 0 ), 1 );
    }

    //ds compute delta
    m_dTranslationDeltaSquaredNormCurrent = ( m_vecPositionCurrent-m_vecTranslationLastKeyFrame ).squaredNorm( );

    //ds add a keyframe if delta is sufficiently high
    if( m_dTranslationDeltaForKeyFrameMetersSquaredNorm < m_dTranslationDeltaSquaredNormCurrent )
    {
        m_cMatcherEpipolar.setKeyFrameToVisibleLandmarks( );
        m_vecTranslationLastKeyFrame = m_vecPositionCurrent;
        m_vecKeyFrames.push_back( CKeyFrame( matTransformationLEFTtoWORLD, p_vecLinearAcceleration.normalized( ), vecVisibleLandmarks ) );

        //ds update scene in viewer
        m_prFrameLEFTtoWORLD = std::pair< bool, Eigen::Isometry3d >( true, matTransformationLEFTtoWORLD );
        m_bIsFrameAvailable  = true;

        //ds current "id"
        const UIDKeyFrame uIDKeyFrameCurrent = m_vecKeyFrames.size( )-1;

        //ds check if optimization is required
        if( m_uIDDeltaKeyFrameForProcessing < uIDKeyFrameCurrent-m_uIDProcessedKeyFrameLAST )
        {
            //ds get keyframe chunk
            std::vector< CKeyFrame > vecKeyFrameChunkForOptimization( m_vecKeyFrames.begin( )+m_uIDProcessedKeyFrameLAST, m_vecKeyFrames.end( ) );

            //ds create file name
            char chBuffer[256];
            std::snprintf( chBuffer, 256, "/home/dominik/workspace_catkin/src/vi_mapper/logs/g2o/keyframes_%06lu-%06lu", m_uIDProcessedKeyFrameLAST, uIDKeyFrameCurrent );
            const std::string strFilePrefix( chBuffer );

            //ds feed sub vector to g2o
            CBridgeG2O::saveCOMBO( strFilePrefix, *m_pCameraSTEREO, *m_vecLandmarks, vecKeyFrameChunkForOptimization );
            CBridgeG2O::saveUVDepthOrDisparity( strFilePrefix, *m_pCameraSTEREO, *m_vecLandmarks, vecKeyFrameChunkForOptimization );
            CBridgeG2O::saveUVDepth( strFilePrefix, *m_pCameraSTEREO, *m_vecLandmarks, vecKeyFrameChunkForOptimization );
            CBridgeG2O::saveXYZ( strFilePrefix, *m_pCameraSTEREO, *m_vecLandmarks, vecKeyFrameChunkForOptimization );

            //ds update last
            m_uIDProcessedKeyFrameLAST = uIDKeyFrameCurrent;
            ++m_uOptimizationsG2O;
        }

        //ds save to cloud
        CCloudstreamer::saveLandmarksToCloud( uIDKeyFrameCurrent, *m_vecLandmarks );
    }
    else
    {
        //ds update scene in viewer: no keyframe
        m_prFrameLEFTtoWORLD = std::pair< bool, Eigen::Isometry3d >( false, matTransformationLEFTtoWORLD );
        m_bIsFrameAvailable  = true;
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
            cv::circle( matMaskDetection, cLandmarkInCameraFrame->ptUVLEFT, 10, cv::Scalar ( 0 ), -1 );
        }

        //ds detect landmarks
        const std::shared_ptr< std::vector< CLandmark* > > vecNewLandmarks( _getNewLandmarks( m_uFrameCount, m_matDisplayLowerReference, p_matImageLEFT, p_matImageRIGHT, matTransformationLEFTtoWORLD, matMaskDetection, p_vecRotationTotal ) );

        //ds all visible in this frame
        m_uNumberofLastVisibleLandmarks += vecNewLandmarks->size( );

        //ds add to permanent reference holder
        m_vecLandmarks->insert( m_vecLandmarks->end( ), vecNewLandmarks->begin( ), vecNewLandmarks->end( ) );

        //ds add this measurement point to the epipolar matcher
        m_cMatcherEpipolar.addDetectionPoint( matTransformationLEFTtoWORLD, vecNewLandmarks );
    }

    //ds build display mat
    cv::Mat matDisplayUpper = cv::Mat( m_pCameraSTEREO->m_uPixelHeight, 2*m_pCameraSTEREO->m_uPixelWidth, CV_8UC3 );
    cv::hconcat( matDisplayLEFT, matDisplayRIGHT, matDisplayUpper );
    _drawInfoBox( matDisplayUpper );
    cv::Mat matDisplayComplete = cv::Mat( 2*m_pCameraSTEREO->m_uPixelHeight, 2*m_pCameraSTEREO->m_uPixelWidth, CV_8UC3 );
    cv::vconcat( matDisplayUpper, m_matDisplayLowerReference, matDisplayComplete );

    //ds display
    cv::imshow( "stereo matching", matDisplayComplete );

    //ds if there was a keystroke
    int iLastKeyStroke( cv::waitKey( m_uWaitKeyTimeoutMS ) );
    if( -1 != iLastKeyStroke )
    {
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
            default:
            {
                break;
            }
        }
    }
    else
    {
        _updateFrameRateForInfoBox( );
    }

    //ds update references
    ++m_uFrameCount;
    m_matTransformationWORLDtoLEFTLAST   = matTransformationWORLDtoLEFT;
    m_matTransformationLEFTLASTtoLEFTNOW = matTransformationWORLDtoLEFT*m_matTransformationWORLDtoLEFTLAST.inverse( );

}

const std::shared_ptr< std::vector< CLandmark* > > CTrackerStereoMotionModel::_getNewLandmarks( const uint64_t& p_uFrame,
                                                                                     cv::Mat& p_matDisplay,
                                                                                     const cv::Mat& p_matImageLEFT,
                                                                                     const cv::Mat& p_matImageRIGHT,
                                                                                     const Eigen::Isometry3d& p_matTransformationLEFTtoWORLD,
                                                                                     const cv::Mat& p_matMask,
                                                                                     const Eigen::Vector3d& p_vecRotation )
{
    //ds precompute extrinsics
    const Eigen::Isometry3d matTransformationWORLDtoLEFT( p_matTransformationLEFTtoWORLD.inverse( ) );
    const MatrixProjection matProjectionWORLDtoLEFT( m_pCameraLEFT->m_matProjection*matTransformationWORLDtoLEFT.matrix( ) );

    //ds solution holder
    std::shared_ptr< std::vector< CLandmark* > > vecNewLandmarks( std::make_shared< std::vector< CLandmark* > >( ) );

    //ds detect new keypoints
    //const std::shared_ptr< std::vector< cv::KeyPoint > > vecKeyPoints( m_cDetector.detectKeyPointsTilewise( p_matImageLEFT, matMask ) );
    std::vector< cv::KeyPoint > vecKeyPoints;
    m_pDetector->detect( p_matImageLEFT, vecKeyPoints, p_matMask );

    //ds compute descriptors for the keypoints
    CDescriptor matReferenceDescriptors;
    //m_pExtractor->compute( p_matImageLEFT, *vecKeyPoints, matReferenceDescriptors );
    m_pExtractor->compute( p_matImageLEFT, vecKeyPoints, matReferenceDescriptors );

    //ds rightside keypoints buffer for descriptor computation
    std::vector< cv::KeyPoint > vecKeyPointRIGHT( 1 );

    //ds process the keypoints and see if we can use them as landmarks
    for( uint32_t u = 0; u < vecKeyPoints.size( ); ++u )
    {
        //ds current points
        const cv::KeyPoint cKeyPointLEFT( vecKeyPoints[u] );
        const cv::Point2f ptLandmarkLEFT( cKeyPointLEFT.pt );
        const CDescriptor matDescriptorLEFT( matReferenceDescriptors.row(u) );

        try
        {
            //ds triangulate the point
            const CMatchTriangulation cMatch( m_pTriangulator->getPointTriangulatedCompactInRIGHT( p_matImageRIGHT, cKeyPointLEFT, matDescriptorLEFT ) );
            const CPoint3DCAMERA vecPointTriangulatedLEFT( cMatch.vecPointXYZCAMERA );
            const CDescriptor matDescriptorRIGHT( cMatch.matDescriptorCAMERA );

            //ds check depth
            const double dDepthMeters( vecPointTriangulatedLEFT.z( ) );

            //ds check if point is in front of camera an not more than a defined distance away
            if( m_dMinimumDepthMeters < dDepthMeters && m_dMaximumDepthMeters > dDepthMeters )
            {
                //ds compute triangulated point in world frame
                const CPoint3DWORLD vecPointTriangulatedWORLD( p_matTransformationLEFTtoWORLD*vecPointTriangulatedLEFT );

                //ds landmark right
                const cv::Point2f ptLandmarkRIGHT( cMatch.ptUVCAMERA );

                //ds allocate a new landmark and add the current position
                CLandmark* pLandmark( new CLandmark( m_uAvailableLandmarkID,
                                                     matDescriptorLEFT,
                                                     cMatch.matDescriptorCAMERA,
                                                     cKeyPointLEFT.size,
                                                     vecPointTriangulatedWORLD,
                                                     m_pCameraLEFT->getNormalHomogenized( ptLandmarkLEFT ),
                                                     ptLandmarkLEFT,
                                                     ptLandmarkRIGHT,
                                                     vecPointTriangulatedLEFT,
                                                     p_matTransformationLEFTtoWORLD.translation( ),
                                                     p_vecRotation,
                                                     matProjectionWORLDtoLEFT,
                                                     p_uFrame ) );

                //ds log creation
                CLogger::CLogLandmarkCreation::addEntry( p_uFrame, pLandmark, dDepthMeters, ptLandmarkLEFT, ptLandmarkRIGHT );

                //ds add to newly detected
                vecNewLandmarks->push_back( pLandmark );

                //ds next landmark id
                ++m_uAvailableLandmarkID;

                //ds draw detected point
                cv::line( p_matDisplay, ptLandmarkLEFT, cv::Point2f( ptLandmarkLEFT.x+m_pCameraSTEREO->m_uPixelWidth, ptLandmarkLEFT.y ), CColorCodeBGR( 175, 175, 175 ) );
                cv::circle( p_matDisplay, ptLandmarkLEFT, 2, CColorCodeBGR( 0, 255, 0 ), -1 );
                //cv::circle( p_matDisplay, ptLandmarkLEFT, cLandmark->dKeyPointSize, CColorCodeBGR( 255, 0, 0 ), 1 );
                cv::putText( p_matDisplay, std::to_string( pLandmark->uID ) , cv::Point2d( ptLandmarkLEFT.x+pLandmark->dKeyPointSize, ptLandmarkLEFT.y+pLandmark->dKeyPointSize ), cv::FONT_HERSHEY_PLAIN, 0.5, CColorCodeBGR( 0, 0, 255 ) );

                //ds draw reprojection of triangulation
                cv::circle( p_matDisplay, cv::Point2d( ptLandmarkRIGHT.x+m_pCameraSTEREO->m_uPixelWidth, ptLandmarkRIGHT.y ), 2, CColorCodeBGR( 255, 0, 0 ), -1 );
            }
            else
            {
                cv::circle( p_matDisplay, ptLandmarkLEFT, 2, CColorCodeBGR( 0, 0, 255 ), -1 );
                //cv::circle( p_matDisplay, ptLandmarkLEFT, cKeyPoint.size, CColorCodeBGR( 0, 0, 255 ) );

                //std::printf( "<CTrackerStereoMotionModel>(_getNewLandmarks) could not find match for keypoint (invalid depth: %f m)\n", vecPointTriangulatedLEFT(2) );
            }
        }
        catch( const CExceptionNoMatchFound& p_cException )
        {
            cv::circle( p_matDisplay, ptLandmarkLEFT, 2, CColorCodeBGR( 0, 0, 255 ), -1 );
            //cv::circle( p_matDisplay, ptLandmarkLEFT, cKeyPoint.size, CColorCodeBGR( 0, 0, 255 ) );

            //std::printf( "<CTrackerStereoMotionModel>(_getNewLandmarks) could not find match for keypoint (%s)\n", p_cException.what( ) );
        }
    }

    std::printf( "<CTrackerStereoMotionModel>(_getNewLandmarks) added new landmarks: %lu/%lu\n", vecNewLandmarks->size( ), vecKeyPoints.size( ) );

    //ds return found landmarks
    return vecNewLandmarks;
}

void CTrackerStereoMotionModel::_shutDown( )
{
    m_bIsShutdownRequested = true;
    std::printf( "<CTrackerStereoMotionModel>(_shutDown) termination requested, <CTrackerStereoMotionModel> disabled\n" );
}

void CTrackerStereoMotionModel::_updateFrameRateForInfoBox( const uint32_t& p_uFrameProbeRange )
{
    //ds check if we can compute the frame rate
    if( p_uFrameProbeRange == m_uFramesCurrentCycle )
    {
        //ds get time delta
        const double dDuration = CLogger::getTimeSeconds( )-m_dPreviousFrameTime;

        //ds compute framerate
        m_dPreviousFrameRate = p_uFrameProbeRange/dDuration;

        //ds enable new measurement (will enter the following if case)
        m_uFramesCurrentCycle = 0;
    }

    //ds check if its the first frame since the last count
    if( 0 == m_uFramesCurrentCycle )
    {
        //ds stop time
        m_dPreviousFrameTime = CLogger::getTimeSeconds( );
    }

    //ds count frames
    ++m_uFramesCurrentCycle;
}

void CTrackerStereoMotionModel::_drawInfoBox( cv::Mat& p_matDisplay ) const
{
    char chBuffer[1024];

    switch( m_eMode )
    {
        case ePlaybackStepwise:
        {
            std::snprintf( chBuffer, 1024, "[%13.2f|%05lu] STEPWISE | X: %5.1f Y: %5.1f Z: %5.1f DELTA: %4.2f | LANDMARKS VISIBLE: %3lu (%3lu,%3lu) INVALID: %3lu TOTAL: %4lu | DETECTIONS: %1lu(%2lu) | KEYFRAMES: %2lu(%2lu) | G2OPTIMIZATIONS: %02u",
                           m_dTimestampLASTSeconds, m_uFrameCount,
                           m_vecPositionCurrent.x( ), m_vecPositionCurrent.y( ), m_vecPositionCurrent.z( ), m_dTranslationDeltaSquaredNormCurrent,
                           m_uNumberofLastVisibleLandmarks, m_cMatcherEpipolar.getNumberOfDetectionsPoseOptimization( ), m_cMatcherEpipolar.getNumberOfDetectionsEpipolar( ), m_cMatcherEpipolar.getNumberOfInvalidLandmarksTotal( ), m_vecLandmarks->size( ),
                           m_cMatcherEpipolar.getNumberOfDetectionPointsActive( ), m_cMatcherEpipolar.getNumberOfDetectionPointsTotal( ),
                           m_vecKeyFrames.size( ), m_vecKeyFrames.size( ), m_uOptimizationsG2O );
            break;
        }
        case ePlaybackBenchmark:
        {
            std::snprintf( chBuffer, 1024, "[%13.2f|%05lu] BENCHMARK FPS: %4.1f | X: %5.1f Y: %5.1f Z: %5.1f DELTA: %4.2f | LANDMARKS VISIBLE: %3lu (%3lu,%3lu) INVALID: %3lu TOTAL: %4lu | DETECTIONS: %1lu(%2lu) | KEYFRAMES: %2lu(%2lu) | G2OPTIMIZATIONS: %02u",
                           m_dTimestampLASTSeconds, m_uFrameCount, m_dPreviousFrameRate,
                           m_vecPositionCurrent.x( ), m_vecPositionCurrent.y( ), m_vecPositionCurrent.z( ), m_dTranslationDeltaSquaredNormCurrent,
                           m_uNumberofLastVisibleLandmarks, m_cMatcherEpipolar.getNumberOfDetectionsPoseOptimization( ), m_cMatcherEpipolar.getNumberOfDetectionsEpipolar( ), m_cMatcherEpipolar.getNumberOfInvalidLandmarksTotal( ), m_vecLandmarks->size( ),
                           m_cMatcherEpipolar.getNumberOfDetectionPointsActive( ), m_cMatcherEpipolar.getNumberOfDetectionPointsTotal( ),
                           m_vecKeyFrames.size( ), m_vecKeyFrames.size( ), m_uOptimizationsG2O );
            break;
        }
        default:
        {
            std::printf( "<CTrackerStereoMotionModel>(_drawInfoBox) unsupported playback mode, no info box displayed\n" );
            break;
        }
    }


    p_matDisplay( cv::Rect( 0, 0, 2*m_pCameraSTEREO->m_uPixelWidth, 17 ) ).setTo( CColorCodeBGR( 0, 0, 0 ) );
    cv::putText( p_matDisplay, chBuffer , cv::Point2i( 2, 12 ), cv::FONT_HERSHEY_PLAIN, 0.8, CColorCodeBGR( 0, 0, 255 ) );
}
