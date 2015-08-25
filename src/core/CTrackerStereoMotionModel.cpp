#include "CTrackerStereoMotionModel.h"

#include <opencv/highgui.h>
#include <opencv2/features2d/features2d.hpp>

#include "configuration/CConfigurationCamera.h"
#include "configuration/CConfigurationOpenCV.h"
#include "utility/CIMUInterpolator.h"
#include "exceptions/CExceptionPoseOptimization.h"
#include "exceptions/CExceptionNoMatchFound.h"

CTrackerStereoMotionModel::CTrackerStereoMotionModel( const EPlaybackMode& p_eMode,
                                                      const uint32_t& p_uWaitKeyTimeoutMS ): m_uWaitKeyTimeoutMS( p_uWaitKeyTimeoutMS ),
                                                                           m_pCameraLEFT( std::make_shared< CPinholeCamera >( CConfigurationCamera::LEFT::cPinholeCamera ) ),
                                                                           m_pCameraRIGHT( std::make_shared< CPinholeCamera >( CConfigurationCamera::RIGHT::cPinholeCamera ) ),
                                                                           m_pCameraSTEREO( std::make_shared< CStereoCamera >( m_pCameraLEFT, m_pCameraRIGHT ) ),

                                                                           m_matTransformationWORLDtoLEFTLAST( Eigen::Matrix4d( CConfigurationCamera::matTransformationIntialWORLDtoLEFT ).transpose( ) ),
                                                                           m_matTransformationLEFTLASTtoLEFTNOW( Eigen::Matrix4d::Identity( ) ),
                                                                           m_vecVelocityAngularFilteredLAST( 0.0, 0.0, 0.0 ),
                                                                           m_vecLinearAccelerationFilteredLAST( 0.0, 0.0, 0.0 ),
                                                                           m_vecPositionKeyFrameLAST( m_matTransformationWORLDtoLEFTLAST.inverse( ).translation( ) ),
                                                                           m_vecCameraOrientationAccumulated( 0.0, 0.0, 0.0 ),
                                                                           m_dTranslationDeltaForKeyFrameMetersL2( 0.1 ),
                                                                           m_dAngleDeltaForOptimizationRadiansL2( 0.25 ),
                                                                           m_vecPositionCurrent( m_matTransformationWORLDtoLEFTLAST.inverse( ).translation( ) ),
                                                                           m_vecPositionLAST( m_vecPositionCurrent ),

                                                                           //ds BRIEF (calibrated 2015-05-31)
                                                                           // m_uKeyPointSize( 7 ),
                                                                           m_pDetector( std::make_shared< cv::GoodFeaturesToTrackDetector >( 200, 0.01, 7.0, 7, true ) ),
                                                                           m_pExtractor( std::make_shared< cv::BriefDescriptorExtractor >( 64 ) ),
                                                                           m_pMatcher( std::make_shared< cv::BFMatcher >( cv::NORM_HAMMING ) ),
                                                                           m_dMatchingDistanceCutoffTriangulation( 100.0 ),
                                                                           m_dMatchingDistanceCutoffPoseOptimization( 60.0 ),
                                                                           m_dMatchingDistanceCutoffEpipolar( 40.0 ),

                                                                           m_uMaximumFailedSubsequentTrackingsPerLandmark( 5 ),
                                                                           m_uVisibleLandmarksMinimum( 100 ),
                                                                           m_dMinimumDepthMeters( 0.05 ),
                                                                           m_dMaximumDepthMeters( 100.0 ),

                                                                           m_pTriangulator( std::make_shared< CTriangulator >( m_pCameraSTEREO, m_pExtractor, m_pMatcher, m_dMatchingDistanceCutoffTriangulation ) ),
                                                                           m_cMatcher( m_pTriangulator, m_pDetector, m_dMinimumDepthMeters, m_dMaximumDepthMeters, m_dMatchingDistanceCutoffPoseOptimization, m_dMatchingDistanceCutoffEpipolar, m_uMaximumFailedSubsequentTrackingsPerLandmark ),

                                                                           m_vecLandmarks( std::make_shared< std::vector< CLandmark* > >( ) ),
                                                                           m_vecClouds( std::make_shared< std::vector< const CDescriptorPointCloud* > >( ) ),
                                                                           m_vecKeyFrames( std::make_shared< std::vector< CKeyFrame* > >( ) ),
                                                                           m_cOptimizer( m_pCameraSTEREO, m_vecLandmarks, m_vecKeyFrames ),

                                                                           m_vecTrees( std::make_shared< std::vector< const C67DTree* > >( ) ),

                                                                           m_eMode( p_eMode )
{
    m_vecLandmarks->clear( );
    m_vecClouds->clear( );
    m_vecKeyFrames->clear( );

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
    std::printf( "<CTrackerStereoMotionModel>(CTrackerStereoMotionModel) maximum timestamp delta (IMU): %f\n", CIMUInterpolator::dMaximumDeltaTimeSeconds );
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
        if( m_cOptimizer.isOptimized( pLandmark ) && m_cOptimizer.isKeyFramed( pLandmark ) )
        {
            CLogger::CLogLandmarkFinalOptimized::addEntry( pLandmark );
        }

        delete pLandmark;
    }

    //ds free keyframes
    for( const CKeyFrame* pKeyFrame: *m_vecKeyFrames )
    {
        delete pKeyFrame;
    }

    //ds free clouds
    for( const CDescriptorPointCloud* pCloud: *m_vecClouds )
    {
        delete pCloud;
    }

    //ds free trees
    for( const C67DTree* pTree: *m_vecTrees )
    {
        delete pTree;
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
    const Eigen::Isometry3d matTransformationParallelLEFTLASTtoLEFTNOW( m_matTransformationLEFTLASTtoLEFTNOW );

    //ds compute total rotation
    const Eigen::Vector3d vecRotationTotal( m_vecVelocityAngularFilteredLAST*dDeltaTimestampSeconds );
    const Eigen::Vector3d vecTranslationTotal( 0.5*m_vecLinearAccelerationFilteredLAST*dDeltaTimestampSeconds*dDeltaTimestampSeconds );

    //ds if the delta is acceptable
    if( CIMUInterpolator::dMaximumDeltaTimeSeconds > dDeltaTimestampSeconds )
    {
        //ds integrate imu input: overwrite rotation
        m_matTransformationLEFTLASTtoLEFTNOW.linear( ) = CMiniVisionToolbox::fromOrientationRodrigues( vecRotationTotal );

        //ds add acceleration
        m_matTransformationLEFTLASTtoLEFTNOW.translation( ) += vecTranslationTotal;
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
                     vecRotationTotal,
                     vecTranslationTotal );

    //ds update references
    m_dTimestampLASTSeconds          = dTimestampSeconds;
    m_vecVelocityAngularFilteredLAST = m_pCameraLEFT->m_matRotationIMUtoCAMERA*CIMUInterpolator::getAngularVelocityFiltered( p_pIMU->getAngularVelocity( ) );
}

void CTrackerStereoMotionModel::_trackLandmarks( const cv::Mat& p_matImageLEFT,
                                                 const cv::Mat& p_matImageRIGHT,
                                                 const Eigen::Isometry3d& p_matTransformationEstimateWORLDtoLEFT,
                                                 const Eigen::Isometry3d& p_matTransformationEstimateParallelWORLDtoLEFT,
                                                 const CLinearAccelerationIMU& p_vecLinearAcceleration,
                                                 const Eigen::Vector3d& p_vecRotationTotal,
                                                 const Eigen::Vector3d& p_vecTranslationTotal )
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
    m_cMatcher.resetVisibilityActiveLandmarks( );



    //ds initial transformation
    Eigen::Isometry3d matTransformationWORLDtoLEFT( p_matTransformationEstimateWORLDtoLEFT );

    try
    {
        //ds get the optimized pose
        matTransformationWORLDtoLEFT = m_cMatcher.getPoseOptimizedSTEREO( m_uFrameCount,
                                                                          matDisplayLEFT,
                                                                          matDisplayRIGHT,
                                                                          p_matImageLEFT,
                                                                          p_matImageRIGHT,
                                                                          p_matTransformationEstimateWORLDtoLEFT,
                                                                          p_vecRotationTotal,
                                                                          p_vecTranslationTotal,
                                                                          dMotionScaling );
    }
    catch( const CExceptionPoseOptimization& p_cException )
    {
        std::printf( "<CTrackerStereoMotionModel>(_trackLandmarks) pose optimization failed: '%s' trying parallel transform\n", p_cException.what( ) );
        try
        {
            //ds get the optimized pose on constant motion
            matTransformationWORLDtoLEFT = m_cMatcher.getPoseOptimizedSTEREO( m_uFrameCount,
                                                                              matDisplayLEFT,
                                                                              matDisplayRIGHT,
                                                                              p_matImageLEFT,
                                                                              p_matImageRIGHT,
                                                                              p_matTransformationEstimateParallelWORLDtoLEFT,
                                                                              p_vecRotationTotal,
                                                                              p_vecTranslationTotal,
                                                                              dMotionScaling );
        }
        catch( const CExceptionPoseOptimization& p_cException )
        {
            //ds stick to the last estimate and adopt orientation
            std::printf( "<CTrackerStereoMotionModel>(_trackLandmarks) parallel pose optimization failed: '%s'\n", p_cException.what( ) );
            matTransformationWORLDtoLEFT = m_matTransformationWORLDtoLEFTLAST;
            matTransformationWORLDtoLEFT.linear( ) = p_matTransformationEstimateWORLDtoLEFT.linear( );
            m_uWaitKeyTimeoutMS = 0;
        }
    }



    //ds respective camera transform
    Eigen::Isometry3d matTransformationLEFTtoWORLD( matTransformationWORLDtoLEFT.inverse( ) );

    //ds estimate acceleration in current WORLD frame (necessary to filter gravity)
    const Eigen::Isometry3d matTransformationIMUtoWORLD( matTransformationLEFTtoWORLD*m_pCameraLEFT->m_matTransformationIMUtoCAMERA );
    const CLinearAccelerationWORLD vecLinearAccelerationWORLD( matTransformationIMUtoWORLD.linear( )*p_vecLinearAcceleration );
    const CLinearAccelerationWORLD vecLinearAccelerationWORLDFiltered( CIMUInterpolator::getLinearAccelerationFiltered( vecLinearAccelerationWORLD ) );

    //ds update acceleration reference
    m_vecLinearAccelerationFilteredLAST = matTransformationWORLDtoLEFT.linear( )*vecLinearAccelerationWORLDFiltered;
    CLogger::CLogLinearAcceleration::addEntry( m_uFrameCount, vecLinearAccelerationWORLD, vecLinearAccelerationWORLDFiltered );

    //ds current translation
    m_vecPositionLAST    = m_vecPositionCurrent;
    m_vecPositionCurrent = matTransformationLEFTtoWORLD.translation( );
    CLogger::CLogTrajectory::addEntry( m_uFrameCount, m_vecPositionCurrent, Eigen::Quaterniond( matTransformationLEFTtoWORLD.linear( ) ) );



    //ds set visible landmarks (including landmarks already detected in the pose optimization)
    const std::shared_ptr< const std::vector< const CMeasurementLandmark* > > vecMeasurements( m_cMatcher.getMeasurementsEpipolar( matDisplayLEFT, matDisplayRIGHT, m_uFrameCount, p_matImageLEFT, p_matImageRIGHT, matTransformationWORLDtoLEFT, matTransformationLEFTtoWORLD, p_vecRotationTotal, dMotionScaling ) );

    //ds compute landmark lost since last (negative if we see more landmarks than before)
    const UIDLandmark uNumberOfVisibleLandmarks = vecMeasurements->size( );
    const int32_t iLandmarksLost                = m_uNumberofVisibleLandmarksLAST-uNumberOfVisibleLandmarks;

    //ds if we lose more than 75% landmarks in one frame
    if( 0.75 < static_cast< double >( iLandmarksLost )/m_uNumberofVisibleLandmarksLAST )
    {
        std::printf( "<CTrackerStereoMotionModel>(_trackLandmarks) lost track (landmarks lost: %i), total delta: %f (%f %f %f)\n", iLandmarksLost, m_vecVelocityAngularFilteredLAST.squaredNorm( ), m_vecVelocityAngularFilteredLAST.x( ), m_vecVelocityAngularFilteredLAST.y( ), m_vecVelocityAngularFilteredLAST.z( ) );
        //m_uWaitKeyTimeoutMS = 0;
    }

    //ds update reference
    m_uNumberofVisibleLandmarksLAST = uNumberOfVisibleLandmarks;

    //ds debug
    for( const CMeasurementLandmark* pMeasurement: *vecMeasurements )
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

    //ds accumulate orientation
    m_vecCameraOrientationAccumulated += p_vecRotationTotal;

    //ds position delta
    const Eigen::Vector3d vecDelta( m_vecPositionCurrent-m_vecPositionKeyFrameLAST );

    //ds get norm
    m_dTranslationDeltaSquaredNormCurrent = vecDelta.squaredNorm( );

    //ds add a keyframe if translational delta is sufficiently high
    if( m_dTranslationDeltaForKeyFrameMetersL2 < m_dTranslationDeltaSquaredNormCurrent           ||
        m_dAngleDeltaForOptimizationRadiansL2 < m_vecCameraOrientationAccumulated.squaredNorm( ) )
    {
        //ds register to matcher
        m_cMatcher.setKeyFrameToVisibleLandmarks( );

        //ds current "id"
        const UIDKeyFrame iIDKeyFrameCurrent = m_vecKeyFrames->size( );

        //ds create new frame
        m_vecKeyFrames->push_back( new CKeyFrame( iIDKeyFrameCurrent,
                                                  matTransformationLEFTtoWORLD,
                                                  p_vecLinearAcceleration.normalized( ),
                                                  *vecMeasurements,
                                                  m_uFrameCount,
                                                  _getLoopClosureKeyFrameFCFS( iIDKeyFrameCurrent, matTransformationLEFTtoWORLD, matDisplayLEFT ) ) );

        //ds check if optimization is required
        if( m_uIDDeltaKeyFrameForOptimization < iIDKeyFrameCurrent-m_uIDProcessedKeyFrameLAST )
        {
            //ds optimize the segment
            m_cOptimizer.optimizeTail( m_uIDProcessedKeyFrameLAST );
            //m_cOptimizer.optimizeContinuous( m_uIDProcessedKeyFrameLAST, iIDKeyFrameCurrent );

            assert( m_vecKeyFrames->back( )->bIsOptimized );

            //ds update transformations with optimized ones
            m_uIDProcessedKeyFrameLAST         = iIDKeyFrameCurrent;
            m_uIDProcessedLandmarkLAST         = m_vecKeyFrames->back( )->vecMeasurements.back( )->uID;
            matTransformationLEFTtoWORLD       = m_vecKeyFrames->back( )->matTransformationLEFTtoWORLD;
            m_vecPositionCurrent               = matTransformationLEFTtoWORLD.translation( );
            matTransformationWORLDtoLEFT       = matTransformationLEFTtoWORLD.inverse( );
            m_matTransformationWORLDtoLEFTLAST = matTransformationWORLDtoLEFT;
            //m_uWaitKeyTimeoutMS = 0;
        }

        //ds update references
        m_vecPositionKeyFrameLAST         = m_vecPositionCurrent;
        m_vecCameraOrientationAccumulated = Eigen::Vector3d::Zero( );

        //ds update scene in viewer
        m_prFrameLEFTtoWORLD = std::pair< bool, Eigen::Isometry3d >( true, matTransformationLEFTtoWORLD );
        m_bIsFrameAvailable  = true;
    }
    else
    {
        //ds update scene in viewer: no keyframe
        m_prFrameLEFTtoWORLD = std::pair< bool, Eigen::Isometry3d >( false, matTransformationLEFTtoWORLD );
        m_bIsFrameAvailable  = true;
    }

    //ds check if we have to detect new landmarks
    if( m_uVisibleLandmarksMinimum > m_uNumberofVisibleLandmarksLAST )
    {
        //ds clean the lower display
        cv::hconcat( matDisplayLEFTClean, matDisplayRIGHTClean, m_matDisplayLowerReference );

        //ds detect landmarks
        const std::shared_ptr< std::vector< CLandmark* > > vecNewLandmarks( _getNewLandmarks( m_uFrameCount, m_matDisplayLowerReference, p_matImageLEFT, p_matImageRIGHT, matTransformationWORLDtoLEFT, matTransformationLEFTtoWORLD, p_vecRotationTotal ) );

        //ds all visible in this frame
        m_uNumberofVisibleLandmarksLAST += vecNewLandmarks->size( );

        //ds add to permanent reference holder
        m_vecLandmarks->insert( m_vecLandmarks->end( ), vecNewLandmarks->begin( ), vecNewLandmarks->end( ) );

        //ds add this measurement point to the epipolar matcher
        m_cMatcher.addDetectionPoint( matTransformationLEFTtoWORLD, vecNewLandmarks );
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
            case CConfigurationOpenCV::KeyStroke::iBackspace:
            {
                if( 0 < m_uWaitKeyTimeoutMS )
                {
                    //ds switch to stepwise mode
                    m_uWaitKeyTimeoutMS = 0;
                    std::printf( "<CTrackerStereoMotionModel>(_trackLandmarks) switched to stepwise mode\n" );
                }
                else
                {
                    //ds switch to benchmark mode
                    m_uWaitKeyTimeoutMS = 1;
                    std::printf( "<CTrackerStereoMotionModel>(_trackLandmarks) switched back to benchmark mode\n" );
                }
                break;
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
    m_matTransformationLEFTLASTtoLEFTNOW = matTransformationWORLDtoLEFT*m_matTransformationWORLDtoLEFTLAST.inverse( );
    m_matTransformationWORLDtoLEFTLAST   = matTransformationWORLDtoLEFT;
}

const std::shared_ptr< std::vector< CLandmark* > > CTrackerStereoMotionModel::_getNewLandmarks( const UIDFrame& p_uFrame,
                                                                                     cv::Mat& p_matDisplay,
                                                                                     const cv::Mat& p_matImageLEFT,
                                                                                     const cv::Mat& p_matImageRIGHT,
                                                                                     const Eigen::Isometry3d& p_matTransformationWORLDtoLEFT,
                                                                                     const Eigen::Isometry3d& p_matTransformationLEFTtoWORLD,
                                                                                     const Eigen::Vector3d& p_vecRotation )
{
    //ds precompute extrinsics
    const MatrixProjection matProjectionWORLDtoLEFT( m_pCameraLEFT->m_matProjection*p_matTransformationWORLDtoLEFT.matrix( ) );

    //ds solution holder
    std::shared_ptr< std::vector< CLandmark* > > vecNewLandmarks( std::make_shared< std::vector< CLandmark* > >( ) );

    //ds detect new keypoints
    //const std::shared_ptr< std::vector< cv::KeyPoint > > vecKeyPoints( m_cDetector.detectKeyPointsTilewise( p_matImageLEFT, matMask ) );
    std::vector< cv::KeyPoint > vecKeyPoints;
    m_pDetector->detect( p_matImageLEFT, vecKeyPoints, m_cMatcher.getMaskActiveLandmarks( p_matTransformationWORLDtoLEFT, p_matDisplay ) );

    //ds compute descriptors for the keypoints
    CDescriptor matReferenceDescriptors;
    //m_pExtractor->compute( p_matImageLEFT, *vecKeyPoints, matReferenceDescriptors );
    m_pExtractor->compute( p_matImageLEFT, vecKeyPoints, matReferenceDescriptors );

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

    //std::printf( "<CTrackerStereoMotionModel>(_getNewLandmarks) added new landmarks: %lu/%lu\n", vecNewLandmarks->size( ), vecKeyPoints.size( ) );

    //ds return found landmarks
    return vecNewLandmarks;
}

const CKeyFrame* CTrackerStereoMotionModel::_getLoopClosureKeyFrame( const UIDKeyFrame& p_uID, const Eigen::Isometry3d& p_matTransformationLEFTtoWORLD, cv::Mat& p_matDisplayLEFT )
{
    return 0;
    //ds save to cloudfile
    //CCloudstreamer::saveLandmarksToCloudFile( p_uID, p_matTransformationLEFTtoWORLD, m_cMatcherEpipolar.getVisibleLandmarks( ) );

    //ds retrieve current point cloud
    const CDescriptorPointCloud* pCloudCurrent( CCloudstreamer::getCloud( p_uID, p_matTransformationLEFTtoWORLD, m_cMatcher.getVisibleOptimizedLandmarks( ) ) );

    //ds retrieve current kdtree
    //C67DTree* pTree( CCloudstreamer::getTree( p_uID, p_matTransformationLEFTtoWORLD, m_cMatcher.getVisibleOptimizedLandmarks( ) ) );

    //ds match buffers
    std::shared_ptr< const std::vector< CMatchCloud > > p_vecMatchesBest( 0 );
    UIDLandmark uBestMatches    = 0;
    UIDKeyFrame uIDKeyFrameBest = 0;

    //ds compare current cloud against previous ones to enable loop closure (skipping the keyframes added just before)
    for( UIDKeyFrame i = 0; i < p_uID-m_uLoopClosingKeyFrameDistance; ++i )
    {
        //ds get matches
        std::shared_ptr< const std::vector< CMatchCloud > > vecMatches( CCloudMatcher::getMatches( pCloudCurrent, m_vecClouds->at( i ) ) );
        const UIDLandmark uNumberOfMatches = vecMatches->size( );

        //int32_t iTreeMatches = pTree->getMatches( m_vecTrees->at( i ) );

        //std::printf( "<CTrackerStereoMotionModel>(_trackLandmarks) cloud [%02lu] > [%02lu] matches: %03lu\n", pCloudCurrent->uID, m_vecClouds->at( i )->uID, uNumberOfMatches );
        //std::printf( "<CTrackerStereoMotionModel>(_trackLandmarks) tree [%02lu] > [%02lu] matches: %03i\n", pTree->uID, m_vecTrees->at( i )->uID, iTreeMatches );

        //ds if we have a suffient amount of matches
        if( m_uMinimumNumberOfMatchesLoopClosure < uNumberOfMatches )
        {
            if( uBestMatches < uNumberOfMatches )
            {
                p_vecMatchesBest = vecMatches;
                uIDKeyFrameBest  = m_vecClouds->at( i )->uID;
                uBestMatches     = uNumberOfMatches;
            }
        }
    }

    //ds push cloud
    m_vecClouds->push_back( pCloudCurrent );
    //m_vecTrees->push_back( pTree );

    //ds if we got a match
    if( 0 < uBestMatches )
    {
        std::printf( "<CTrackerStereoMotionModel>(_trackLandmarks) found loop closure for keyframes: %06lu -> %06lu (points: %lu)\n", p_uID, uIDKeyFrameBest, uBestMatches );

        //ds draw matches
        for( const CMatchCloud& cMatch: *p_vecMatchesBest )
        {
            cv::circle( p_matDisplayLEFT, m_vecLandmarks->at( cMatch.uIDQuery )->getLastDetectionLEFT( ), 3, CColorCodeBGR( 255, 0, 255 ), -1 );
        }

        //m_uWaitKeyTimeoutMS = 0;

        //ds return reference
        return m_vecKeyFrames->at( uIDKeyFrameBest );
    }
    else
    {
        //ds return empty
        return 0;
    }
}

const CKeyFrame* CTrackerStereoMotionModel::_getLoopClosureKeyFrameFCFS( const UIDKeyFrame& p_uID, const Eigen::Isometry3d& p_matTransformationLEFTtoWORLD, cv::Mat& p_matDisplayLEFT )
{
    //ds retrieve current point cloud
    const CDescriptorPointCloud* pCloudCurrent( CCloudstreamer::getCloud( p_uID, p_matTransformationLEFTtoWORLD, m_cMatcher.getVisibleOptimizedLandmarks( ) ) );

    //ds compare current cloud against previous ones to enable loop closure (skipping the keyframes added just before)
    for( const CDescriptorPointCloud* pCloudReference: *m_vecClouds )
    {
        //ds if we get close to the current keyframe
        if( 2 > pCloudCurrent->uID-pCloudReference->uID )
        {
            break;
        }

        //ds get matches
        std::shared_ptr< const std::vector< CMatchCloud > > vecMatches( CCloudMatcher::getMatches( pCloudCurrent, pCloudReference ) );

        //ds if we have a suffient amount of matches
        if( m_uMinimumNumberOfMatchesLoopClosure < vecMatches->size( ) )
        {
            std::printf( "<CTrackerStereoMotionModel>(_getLoopClosureKeyFrameFCFS) found loop closure for keyframes: %06lu -> %06lu (points: %lu)\n", p_uID, pCloudReference->uID, vecMatches->size( ) );

            //ds draw matches
            for( const CMatchCloud& cMatch: *vecMatches )
            {
                cv::circle( p_matDisplayLEFT, m_vecLandmarks->at( cMatch.uIDQuery )->getLastDetectionLEFT( ), 3, CColorCodeBGR( 255, 0, 255 ), -1 );
            }

            //m_uWaitKeyTimeoutMS = 0;

            //ds transformation between this keyframe and the loop closure one (take current measurement as prior)
            Eigen::Isometry3d matTransformationToCLOSURE( pCloudReference->matTransformationLEFTtoWORLD.inverse( )*pCloudCurrent->matTransformationLEFTtoWORLD );

            //ds 1mm for convergence
            const double dErrorDeltaForConvergence = 0.001;
            double dErrorSquaredPrevious           = 0.0;
            const double dMaximumErrorForInlier    = 0.1;

            //ds run least-squares maximum 10 times
            for( uint8_t uLS = 0; uLS < 10; ++uLS )
            {
                //ds error
                double dErrorSquaredTotal = 0.0;
                uint8_t uOutliers         = 0;

                //ds LS setup
                Eigen::Matrix< double, 6, 6 > matH;
                Eigen::Matrix< double, 6, 1 > vecB;
                matH.setZero( );
                vecB.setZero( );

                //ds for all the points
                for( const CMatchCloud& cMatch: *vecMatches )
                {
                    //ds compute projection into closure
                    const CPoint3DCAMERA vecPointXYZCLOSURE( matTransformationToCLOSURE*cMatch.vecPointXYZCAMERAQuery );
                    assert( 0.0 < vecPointXYZCLOSURE.z( ) );

                    //ds compute error
                    const Eigen::Vector3d vecError( vecPointXYZCLOSURE-cMatch.vecPointXYZCAMERAMatch );

                    //ds update chi
                    const double dErrorSquared = vecError.transpose( )*vecError;

                    //ds check if outlier
                    double dWeight = 1.0;
                    if( dMaximumErrorForInlier < dErrorSquared )
                    {
                        dWeight = dMaximumErrorForInlier/dErrorSquared;
                        ++uOutliers;
                    }

                    dErrorSquaredTotal += dErrorSquared;

                    //ds get the jacobian of the transform part = [I 2*skew(T*modelPoint)]
                    Eigen::Matrix< double, 3, 6 > matJacobianTransform;
                    matJacobianTransform.setZero( );
                    matJacobianTransform.block<3,3>(0,0).setIdentity( );
                    matJacobianTransform.block<3,3>(0,3) = -2*CMiniVisionToolbox::getSkew( vecPointXYZCLOSURE );

                    //ds precompute transposed
                    const Eigen::Matrix< double, 6, 3 > matJacobianTransformTransposed( matJacobianTransform.transpose( ) );

                    //ds accumulate
                    matH += dWeight*matJacobianTransformTransposed*matJacobianTransform;
                    vecB += dWeight*matJacobianTransformTransposed*vecError;
                }

                //ds solve the system and update the estimate
                matTransformationToCLOSURE = CMiniVisionToolbox::vector2transform( matH.ldlt( ).solve( -vecB ) )*matTransformationToCLOSURE;

                //ds damp rotation
                const Eigen::Matrix3d matRotation        = matTransformationToCLOSURE.linear( );
                Eigen::Matrix3d matRotationSquared       = matRotation.transpose( )*matRotation;
                matRotationSquared.diagonal( ).array( ) -= 1;
                matTransformationToCLOSURE.linear( )    -= 0.5*matRotation*matRotationSquared;

                /*ds descent required
                if( dErrorSquaredPrevious < dErrorSquaredTotal )
                {
                    std::printf( "<CTrackerStereoMotionModel>(_getLoopClosureKeyFrameFCFS) unable to optimize - average error: %f (outliers: %u)\n", dErrorSquaredTotal/vecMatches->size( ), uOutliers );
                    break;
                }*/

                //ds check if converged
                if( dErrorDeltaForConvergence > std::fabs( dErrorSquaredPrevious-dErrorSquaredTotal ) )
                {
                    std::printf( "<CTrackerStereoMotionModel>(_getLoopClosureKeyFrameFCFS) converged in %u iterations - average error: %f (outliers: %u)\n", uLS, dErrorSquaredTotal/vecMatches->size( ), uOutliers );
                    break;
                }
                else
                {
                    dErrorSquaredPrevious = dErrorSquaredTotal;
                }
            }

            //ds add current cloud and return reference
            m_vecClouds->push_back( pCloudCurrent );
            return m_vecKeyFrames->at( pCloudReference->uID );
        }
    }

    //ds push cloud
    m_vecClouds->push_back( pCloudCurrent );
    return 0;
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
                           m_uNumberofVisibleLandmarksLAST, m_cMatcher.getNumberOfDetectionsPoseOptimization( ), m_cMatcher.getNumberOfDetectionsEpipolar( ), m_cMatcher.getNumberOfInvalidLandmarksTotal( ), m_vecLandmarks->size( ),
                           m_cMatcher.getNumberOfDetectionPointsActive( ), m_cMatcher.getNumberOfDetectionPointsTotal( ),
                           m_vecKeyFrames->size( ), m_vecKeyFrames->size( ), m_cOptimizer.getNumberOfSegmentOptimizations( ) );
            break;
        }
        case ePlaybackBenchmark:
        {
            std::snprintf( chBuffer, 1024, "[%13.2f|%05lu] BENCHMARK FPS: %4.1f | X: %5.1f Y: %5.1f Z: %5.1f DELTA: %4.2f | LANDMARKS VISIBLE: %3lu (%3lu,%3lu) INVALID: %3lu TOTAL: %4lu | DETECTIONS: %1lu(%2lu) | KEYFRAMES: %2lu(%2lu) | G2OPTIMIZATIONS: %02u",
                           m_dTimestampLASTSeconds, m_uFrameCount, m_dPreviousFrameRate,
                           m_vecPositionCurrent.x( ), m_vecPositionCurrent.y( ), m_vecPositionCurrent.z( ), m_dTranslationDeltaSquaredNormCurrent,
                           m_uNumberofVisibleLandmarksLAST, m_cMatcher.getNumberOfDetectionsPoseOptimization( ), m_cMatcher.getNumberOfDetectionsEpipolar( ), m_cMatcher.getNumberOfInvalidLandmarksTotal( ), m_vecLandmarks->size( ),
                           m_cMatcher.getNumberOfDetectionPointsActive( ), m_cMatcher.getNumberOfDetectionPointsTotal( ),
                           m_vecKeyFrames->size( ), m_vecKeyFrames->size( ), m_cOptimizer.getNumberOfSegmentOptimizations( ) );
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
