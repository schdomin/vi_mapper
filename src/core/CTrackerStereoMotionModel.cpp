#include "CTrackerStereoMotionModel.h"

#include <opencv/highgui.h>
#include <opencv2/features2d/features2d.hpp>

#include "configuration/CConfigurationCamera.h"
#include "configuration/CConfigurationOpenCV.h"
#include "exceptions/CExceptionPoseOptimization.h"
#include "exceptions/CExceptionNoMatchFound.h"

CTrackerStereoMotionModel::CTrackerStereoMotionModel( const EPlaybackMode& p_eMode,
                                                      const std::shared_ptr< CIMUInterpolator > p_pIMUInterpolator,
                                                      const uint32_t& p_uWaitKeyTimeoutMS ): m_uWaitKeyTimeoutMS( p_uWaitKeyTimeoutMS ),
                                                                           m_pCameraLEFT( std::make_shared< CPinholeCamera >( CConfigurationCamera::LEFT::cPinholeCamera ) ),
                                                                           m_pCameraRIGHT( std::make_shared< CPinholeCamera >( CConfigurationCamera::RIGHT::cPinholeCamera ) ),
                                                                           m_pCameraSTEREO( std::make_shared< CStereoCamera >( m_pCameraLEFT, m_pCameraRIGHT ) ),

                                                                           m_matTransformationWORLDtoLEFTLAST( p_pIMUInterpolator->getTransformationWORLDtoCAMERA( m_pCameraLEFT->m_matRotationIMUtoCAMERA ) ),
                                                                           m_matTransformationLEFTLASTtoLEFTNOW( Eigen::Matrix4d::Identity( ) ),
                                                                           m_vecVelocityAngularFilteredLAST( 0.0, 0.0, 0.0 ),
                                                                           m_vecLinearAccelerationFilteredLAST( 0.0, 0.0, 0.0 ),
                                                                           m_vecPositionKeyFrameLAST( m_matTransformationWORLDtoLEFTLAST.inverse( ).translation( ) ),
                                                                           m_vecCameraOrientationAccumulated( 0.0, 0.0, 0.0 ),
                                                                           m_vecPositionCurrent( m_vecPositionKeyFrameLAST ),
                                                                           m_vecPositionLAST( m_vecPositionCurrent ),

                                                                           //ds BRIEF (calibrated 2015-05-31)
                                                                           // m_uKeyPointSize( 7 ),
                                                                           m_pDetector( std::make_shared< cv::GoodFeaturesToTrackDetector >( 200, 0.01, 7.0, 7, true ) ),
                                                                           m_pExtractor( std::make_shared< cv::BriefDescriptorExtractor >( 64 ) ),
                                                                           m_pMatcher( std::make_shared< cv::BFMatcher >( cv::NORM_HAMMING ) ),
                                                                           m_dMatchingDistanceCutoffTriangulation( 100.0 ),
                                                                           m_dMatchingDistanceCutoffPoseOptimization( 50.0 ),
                                                                           m_dMatchingDistanceCutoffEpipolar( 50.0 ),
                                                                           m_uVisibleLandmarksMinimum( 100 ),

                                                                           m_pTriangulator( std::make_shared< CTriangulator >( m_pCameraSTEREO, m_pExtractor, m_pMatcher, m_dMatchingDistanceCutoffTriangulation ) ),
                                                                           m_cMatcher( m_pTriangulator, m_pDetector, m_dMinimumDepthMeters, m_dMaximumDepthMeters, m_dMatchingDistanceCutoffPoseOptimization, m_dMatchingDistanceCutoffEpipolar, m_uMaximumFailedSubsequentTrackingsPerLandmark ),

                                                                           m_vecLandmarks( std::make_shared< std::vector< CLandmark* > >( ) ),
                                                                           m_vecKeyFrames( std::make_shared< std::vector< CKeyFrame* > >( ) ),

                                                                           m_cGraphOptimizer( m_pCameraSTEREO, m_vecLandmarks, m_vecKeyFrames, m_matTransformationWORLDtoLEFTLAST.inverse( ) ),
                                                                           m_vecTranslationToG2o( m_vecPositionCurrent ),

                                                                           m_pIMU( p_pIMUInterpolator ),

                                                                           m_eMode( p_eMode )
{
    m_vecLandmarks->clear( );
    m_vecKeyFrames->clear( );
    m_vecRotations.clear( );

    //ds windows
    _initializeTranslationWindow( );

    //ds set opencv parallelization threads
    cv::setNumThreads( 4 );

    //ds initialize reference frames with black images
    m_matDisplayLowerReference = cv::Mat::zeros( m_pCameraSTEREO->m_uPixelHeight, 2*m_pCameraSTEREO->m_uPixelWidth, CV_8UC3 );

    //ds initialize the window
    cv::namedWindow( "vi_mapper [L|R]", cv::WINDOW_AUTOSIZE );

    CLogger::openBox( );
    std::printf( "[%06lu]<CTrackerStereoMotionModel>(CTrackerStereoMotionModel) <OpenCV> available CPUs: %i\n", m_uFrameCount, cv::getNumberOfCPUs( ) );
    std::printf( "[%06lu]<CTrackerStereoMotionModel>(CTrackerStereoMotionModel) <OpenCV> available threads: %i\n", m_uFrameCount, cv::getNumThreads( ) );
    std::printf( "[%06lu]<CTrackerStereoMotionModel>(CTrackerStereoMotionModel) feature detector: %s\n", m_uFrameCount, m_pDetector->name( ).c_str( ) );
    std::printf( "[%06lu]<CTrackerStereoMotionModel>(CTrackerStereoMotionModel) descriptor extractor: %s\n", m_uFrameCount, m_pExtractor->name( ).c_str( ) );
    std::printf( "[%06lu]<CTrackerStereoMotionModel>(CTrackerStereoMotionModel) descriptor matcher: %s\n", m_uFrameCount, m_pMatcher->name( ).c_str( ) );
    std::printf( "[%06lu]<CTrackerStereoMotionModel>(CTrackerStereoMotionModel) descriptor size: %i bytes\n", m_uFrameCount, m_pExtractor->descriptorSize( ) );
    std::printf( "[%06lu]<CTrackerStereoMotionModel>(CTrackerStereoMotionModel) <CIMUInterpolator> maximum timestamp delta: %f\n", m_uFrameCount, CIMUInterpolator::dMaximumDeltaTimeSeconds );
    std::printf( "[%06lu]<CTrackerStereoMotionModel>(CTrackerStereoMotionModel) <CIMUInterpolator> imprecision angular velocity: %f\n", m_uFrameCount, CIMUInterpolator::m_dImprecisionAngularVelocity );
    std::printf( "[%06lu]<CTrackerStereoMotionModel>(CTrackerStereoMotionModel) <CIMUInterpolator> imprecision linear acceleration: %f\n", m_uFrameCount, CIMUInterpolator::m_dImprecisionLinearAcceleration );
    std::printf( "[%06lu]<CTrackerStereoMotionModel>(CTrackerStereoMotionModel) <CIMUInterpolator> bias linear acceleration x/y/z: %4.2f/%4.2f/%4.2f\n", m_uFrameCount, CIMUInterpolator::m_vecBiasLinearAccelerationXYZ[0],
                                                                                                                                                  CIMUInterpolator::m_vecBiasLinearAccelerationXYZ[1],
                                                                                                                                                  CIMUInterpolator::m_vecBiasLinearAccelerationXYZ[2] );
    std::printf( "[%06lu]<CTrackerStereoMotionModel>(CTrackerStereoMotionModel) <Landmark> cap iterations: %u\n", m_uFrameCount, CLandmark::uCapIterations );
    std::printf( "[%06lu]<CTrackerStereoMotionModel>(CTrackerStereoMotionModel) <Landmark> convergence delta: %f\n", m_uFrameCount, CLandmark::dConvergenceDelta );
    std::printf( "[%06lu]<CTrackerStereoMotionModel>(CTrackerStereoMotionModel) <Landmark> maximum error L2 inlier: %f\n", m_uFrameCount, CLandmark::dKernelMaximumErrorSquaredPixels );
    std::printf( "[%06lu]<CTrackerStereoMotionModel>(CTrackerStereoMotionModel) <Landmark> maximum error L2 average: %f\n", m_uFrameCount, CLandmark::dMaximumErrorSquaredAveragePixels );
    std::printf( "[%06lu]<CTrackerStereoMotionModel>(CTrackerStereoMotionModel) key frame delta for optimization: %lu\n", m_uFrameCount, m_uIDDeltaKeyFrameForOptimization );
    std::printf( "[%06lu]<CTrackerStereoMotionModel>(CTrackerStereoMotionModel) minimum matches for loop closure: %lu\n", m_uFrameCount, m_uMinimumNumberOfMatchesLoopClosure );
    std::printf( "[%06lu]<CTrackerStereoMotionModel>(CTrackerStereoMotionModel) instance allocated\n", m_uFrameCount );
    CLogger::closeBox( );
}

CTrackerStereoMotionModel::~CTrackerStereoMotionModel( )
{
    //ds free all landmarks
    for( const CLandmark* pLandmark: *m_vecLandmarks )
    {
        //ds write final state to file before deleting
        //CLogger::CLogLandmarkFinal::addEntry( pLandmark );

        //ds save optimized landmarks to separate file
        if( pLandmark->bIsOptimal && 1 < pLandmark->uNumberOfKeyFramePresences )
        {
            //CLogger::CLogLandmarkFinalOptimized::addEntry( pLandmark );
        }

        delete pLandmark;
    }

    //ds free keyframes
    for( const CKeyFrame* pKeyFrame: *m_vecKeyFrames )
    {
        delete pKeyFrame;
    }

    /*ds close loggers
    CLogger::CLogLandmarkCreation::close( );
    CLogger::CLogLandmarkFinal::close( );
    CLogger::CLogLandmarkFinalOptimized::close( );
    CLogger::CLogTrajectory::close( );
    CLogger::CLogIMUInput::close( );*/

    std::printf( "[%06lu]<CTrackerStereoMotionModel>(~CTrackerStereoMotionModel) instance deallocated\n", m_uFrameCount );
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

    //ds parallel transformation with erased translation
    Eigen::Isometry3d matTransformationRotationOnlyLEFTLASTtoLEFTNOW( m_matTransformationLEFTLASTtoLEFTNOW );
    matTransformationRotationOnlyLEFTLASTtoLEFTNOW.translation( ) = Eigen::Vector3d::Zero( );

    //ds if the delta is acceptable
    if( CIMUInterpolator::dMaximumDeltaTimeSeconds > dDeltaTimestampSeconds )
    {
        //ds compute total rotation
        const Eigen::Vector3d vecRotationTotal( m_vecVelocityAngularFilteredLAST*dDeltaTimestampSeconds );
        const Eigen::Vector3d vecTranslationTotal( 0.5*m_vecLinearAccelerationFilteredLAST*dDeltaTimestampSeconds*dDeltaTimestampSeconds );

        //ds integrate imu input: overwrite rotation
        m_matTransformationLEFTLASTtoLEFTNOW.linear( ) = CMiniVisionToolbox::fromOrientationRodrigues( vecRotationTotal );

        //ds add acceleration
        m_matTransformationLEFTLASTtoLEFTNOW.translation( ) += vecTranslationTotal;

        //ds process images (fed with IMU prior pose)
        _trackLandmarks( matPreprocessedLEFT,
                         matPreprocessedRIGHT,
                         m_matTransformationLEFTLASTtoLEFTNOW*m_matTransformationWORLDtoLEFTLAST,
                         matTransformationRotationOnlyLEFTLASTtoLEFTNOW*m_matTransformationWORLDtoLEFTLAST,
                         p_pIMU->getLinearAcceleration( ),
                         p_pIMU->getAngularVelocity( ),
                         vecRotationTotal,
                         vecTranslationTotal,
                         dDeltaTimestampSeconds );
    }
    else
    {
        //ds compute reduced entities
        const Eigen::Vector3d vecRotationTotalDamped( m_vecVelocityAngularFilteredLAST*CIMUInterpolator::dMaximumDeltaTimeSeconds );
        const Eigen::Vector3d vecTranslationTotalDamped( Eigen::Vector3d::Zero( ) );

        //ds use full angular velocity
        std::printf( "[%06lu]<CTrackerStereoMotionModel>(receivevDataVI) using reduced IMU input, timestamp delta: %f\n", m_uFrameCount, dDeltaTimestampSeconds );

        //ds integrate imu input: overwrite rotation with limited IMU input
        m_matTransformationLEFTLASTtoLEFTNOW.linear( ) = CMiniVisionToolbox::fromOrientationRodrigues( vecRotationTotalDamped );

        //ds process images (fed with IMU prior pose: damped input)
        _trackLandmarks( matPreprocessedLEFT,
                         matPreprocessedRIGHT,
                         m_matTransformationLEFTLASTtoLEFTNOW*m_matTransformationWORLDtoLEFTLAST,
                         matTransformationRotationOnlyLEFTLASTtoLEFTNOW*m_matTransformationWORLDtoLEFTLAST,
                         p_pIMU->getLinearAcceleration( ),
                         p_pIMU->getAngularVelocity( ),
                         vecRotationTotalDamped,
                         vecTranslationTotalDamped,
                         dDeltaTimestampSeconds );
    }

    //ds update timestamp
    m_dTimestampLASTSeconds = dTimestampSeconds;
}

void CTrackerStereoMotionModel::finalize( )
{
    //ds if tracker GUI is still open - otherwise run the optimization right away
    if( !m_bIsShutdownRequested )
    {
        //ds inform
        std::printf( "[%06lu]<CTrackerStereoMotionModel>(finalize) press any key to perform final optimization\n", m_uFrameCount );

        //ds wait for user input
        cv::waitKey( 0 );
    }
    else
    {
        std::printf( "[%06lu]<CTrackerStereoMotionModel>(finalize) running final optimization\n", m_uFrameCount );
    }

    //ds if we have at least two keyframes
    if( 1 < m_vecKeyFrames->size( ) )
    {
        //ds landmark start id
        const std::vector< CLandmark* >::size_type uIDBeginLandmark = m_vecKeyFrames->at( std::max( static_cast< UIDKeyFrame >( 0 ), m_uIDProcessedKeyFrameLAST-1 ) )->vecMeasurements.front( )->uID;

        //ds newly closed keyframes
        std::vector< CKeyFrame* > vecClosedKeyFrames( 0 );

        //ds if minimum keyframes available
        if( 20 < m_vecKeyFrames->size( ) )
        {
            //ds try to find loop closures for the last 10 keyframes
            for( std::vector< CKeyFrame* >::const_iterator itKeyFrameForClosure = m_vecKeyFrames->end( )-20; itKeyFrameForClosure < m_vecKeyFrames->end( ); ++itKeyFrameForClosure )
            {
                //ds buffer key frame
                const CKeyFrame* pKeyFrameForClosure = *itKeyFrameForClosure;
                assert( 0 != pKeyFrameForClosure );

                //ds only check if we dont have closures for this key frame already
                if( pKeyFrameForClosure->vecLoopClosures.empty( ) )
                {
                    std::printf( "[%06lu]<CTrackerStereoMotionModel>(finalize) checking closures for keyframe: %06lu\n", m_uFrameCount, pKeyFrameForClosure->uID );

                    //ds stop time for loop closure checking
                    const double dStartTimeLoopClosingSeconds = CLogger::getTimeSeconds( );

                    //ds detect loop closures (practically unlimited radius)
                    const std::vector< const CKeyFrame::CMatchICP* > vecLoopClosures( _getLoopClosuresForKeyFrame( pKeyFrameForClosure->uID,
                                                                                                                   pKeyFrameForClosure->matTransformationLEFTtoWORLD,
                                                                                                                   pKeyFrameForClosure->vecCloud,
                                                                                                                   1e12,
                                                                                                                   10 ) );

                    //ds update time
                    m_dTotalLoopClosingDurationSeconds += ( CLogger::getTimeSeconds( )-dStartTimeLoopClosingSeconds );

                    std::printf( "[%06lu]<CTrackerStereoMotionModel>(finalize) found closures: %lu\n", m_uFrameCount, vecLoopClosures.size( ) );

                    //ds if we found closures
                    if( !vecLoopClosures.empty( ) )
                    {
                        //ds first allocate
                        CKeyFrame* pClosedKeyFrame = new CKeyFrame( pKeyFrameForClosure->uID,
                                                                    pKeyFrameForClosure->uFrameOfCreation,
                                                                    pKeyFrameForClosure->matTransformationLEFTtoWORLD,
                                                                    pKeyFrameForClosure->vecLinearAccelerationNormalized,
                                                                    pKeyFrameForClosure->vecMeasurements,
                                                                    pKeyFrameForClosure->vecCloud,
                                                                    vecLoopClosures,
                                                                    pKeyFrameForClosure->dInformationFactor );

                        assert( 0 != pClosedKeyFrame );

                        //ds create new frame
                        vecClosedKeyFrames.push_back( pClosedKeyFrame );

                        std::printf( "[%06lu]<CTrackerStereoMotionModel>(finalize) closed keyframe: %06lu\n", m_uFrameCount, pKeyFrameForClosure->uID );
                    }
                }
            }

            //ds check if we could close some frames and there was an optimization in the last 10 frames
            if( !vecClosedKeyFrames.empty( ) )
            {
                //ds substitute keyframes in container
                for( CKeyFrame* pKeyFrameSubstitute: vecClosedKeyFrames )
                {
                    //ds free reference
                    if( 0 != m_vecKeyFrames->at( pKeyFrameSubstitute->uID ) )
                    {
                        delete m_vecKeyFrames->at( pKeyFrameSubstitute->uID );

                        //ds substitute
                        m_vecKeyFrames->at( pKeyFrameSubstitute->uID ) = pKeyFrameSubstitute;
                    }
                }

                //ds check if we have to update closures in the graph
                if( m_uIDProcessedKeyFrameLAST > vecClosedKeyFrames.front( )->uID )
                {
                    std::printf( "[%06lu]<CTrackerStereoMotionModel>(finalize) updating already optimized keyframes \n", m_uFrameCount );

                    //ds update the already added closures
                    m_cGraphOptimizer.updateLoopClosuresFromKeyFrame( vecClosedKeyFrames.front( )->uID, m_uIDProcessedKeyFrameLAST, m_vecTranslationToG2o );
                }

                //ds lock initial trajectory
                //m_cGraphOptimizer.lockTrajectory( 0, 100 );
            }
        }

        //ds last optmization
        m_cGraphOptimizer.optimizeContinuous( m_uFrameCount, m_uIDProcessedKeyFrameLAST, uIDBeginLandmark, m_vecTranslationToG2o, 1 );

        //ds save final graph
        m_cGraphOptimizer.saveFinalGraph( m_uFrameCount, m_vecTranslationToG2o );
    }
    else
    {
        std::printf( "[%06lu]<CTrackerStereoMotionModelKITTI>(finalize) skipped - not enough keyframes\n", m_uFrameCount );
    }

    //ds trigger shutdown
    m_bIsShutdownRequested = true;
}

void CTrackerStereoMotionModel::_trackLandmarks( const cv::Mat& p_matImageLEFT,
                                                 const cv::Mat& p_matImageRIGHT,
                                                 const Eigen::Isometry3d& p_matTransformationEstimateWORLDtoLEFT,
                                                 const Eigen::Isometry3d& p_matTransformationEstimateParallelWORLDtoLEFT,
                                                 const CLinearAccelerationIMU& p_vecLinearAcceleration,
                                                 const CAngularVelocityIMU& p_vecAngularVelocity,
                                                 const Eigen::Vector3d& p_vecRotationTotal,
                                                 const Eigen::Vector3d& p_vecTranslationTotal,
                                                 const double& p_dDeltaTimeSeconds )
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

    //ds compute motion scaling (capped)
    const double dMotionScaling = std::min( 1.0+100*( p_vecRotationTotal.squaredNorm( )+p_vecTranslationTotal.squaredNorm( ) ), 2.0 );

    //ds refresh landmark states
    m_cMatcher.resetVisibilityActiveLandmarks( );

    //ds initial transformation
    Eigen::Isometry3d matTransformationWORLDtoLEFT( p_matTransformationEstimateWORLDtoLEFT );

    try
    {
        //ds get the optimized pose
        matTransformationWORLDtoLEFT = m_cMatcher.getPoseOptimizedSTEREOUV( m_uFrameCount,
                                                                          matDisplayLEFT,
                                                                          matDisplayRIGHT,
                                                                          p_matImageLEFT,
                                                                          p_matImageRIGHT,
                                                                          p_matTransformationEstimateWORLDtoLEFT,
                                                                          m_matTransformationWORLDtoLEFTLAST,
                                                                          p_vecRotationTotal,
                                                                          p_vecTranslationTotal,
                                                                          dMotionScaling );

        //ds if still here it went okay - reduce instability count if necessary
        if( 0 < m_uCountInstability )
        {
            --m_uCountInstability;
        }
    }
    catch( const CExceptionPoseOptimization& p_cException )
    {
        //std::printf( "<CTrackerStereoMotionModel>(_trackLandmarks) pose optimization failed [RAW PRIOR]: '%s'\n", p_cException.what( ) );
        try
        {
            //ds get the optimized pose on constant motion
            matTransformationWORLDtoLEFT = m_cMatcher.getPoseOptimizedSTEREOUV( m_uFrameCount,
                                                                              matDisplayLEFT,
                                                                              matDisplayRIGHT,
                                                                              p_matImageLEFT,
                                                                              p_matImageRIGHT,
                                                                              m_matTransformationWORLDtoLEFTLAST,
                                                                              m_matTransformationWORLDtoLEFTLAST,
                                                                              p_vecRotationTotal,
                                                                              p_vecTranslationTotal,
                                                                              dMotionScaling );
        }
        catch( const CExceptionPoseOptimization& p_cException )
        {
            //ds if not capped already
            if( 20 > m_uCountInstability )
            {
                m_uCountInstability += 5;
            }

            std::printf( "[%06lu]<CTrackerStereoMotionModel>(_trackLandmarks) pose optimization failed [DAMPED PRIOR]: '%s' - running on damped IMU only\n", m_uFrameCount, p_cException.what( ) );

            //ds compute damped rotation
            Eigen::Vector3d vecRotationYZ( p_vecRotationTotal );
            vecRotationYZ.x( ) = 0.0;
            matTransformationWORLDtoLEFT = CMiniVisionToolbox::fromOrientationRodrigues( vecRotationYZ )*m_matTransformationWORLDtoLEFTLAST;

            /*try
            {
                //ds get the optimized pose on last detection transformation
                matTransformationWORLDtoLEFT = m_cMatcher.getPoseOptimizedSTEREOUVfromLAST( m_uFrameCount,
                                                                                            matDisplayLEFT,
                                                                                            matDisplayRIGHT,
                                                                                            p_matImageLEFT,
                                                                                            p_matImageRIGHT,
                                                                                            p_matTransformationEstimateWORLDtoLEFT,
                                                                                            m_matTransformationWORLDtoLEFTLAST,
                                                                                            p_vecRotationTotal,
                                                                                            p_vecTranslationTotal,
                                                                                            dMotionScaling );
            }
            catch( const CExceptionPoseOptimization& p_cException )
            {
                //ds if not capped already
                if( 20 > m_uCountInstability )
                {
                    m_uCountInstability += 5;
                }

                //ds compute damped rotation
                Eigen::Vector3d vecRotationYZ( p_vecRotationTotal );
                vecRotationYZ.x( ) = 0.0;

                //ds stick to the IMU rotation estimate
                std::printf( "[%06lu]<CTrackerStereoMotionModel>(_trackLandmarks) pose optimization failed [LAST DETECTION]: '%s' - running on damped IMU\n", m_uFrameCount, p_cException.what( ) );
                matTransformationWORLDtoLEFT = CMiniVisionToolbox::fromOrientationRodrigues( vecRotationYZ )*m_matTransformationWORLDtoLEFTLAST;
                //m_uWaitKeyTimeoutMS = 0;
            }*/
        }
    }

    //ds get current measurements (including landmarks already detected in the pose optimization)
    const std::shared_ptr< const std::vector< const CMeasurementLandmark* > > vecMeasurements = m_cMatcher.getMeasurementsEpipolar( m_uFrameCount, p_matImageLEFT, p_matImageRIGHT, matTransformationWORLDtoLEFT, matTransformationWORLDtoLEFT.inverse( ), dMotionScaling, matDisplayLEFT, matDisplayRIGHT );

    //ds refine pose AGAIN on all measurements
    //matTransformationWORLDtoLEFT = m_cMatcher.getPoseRefinedOnVisibleLandmarks( matTransformationWORLDtoLEFT );
    Eigen::Isometry3d matTransformationLEFTtoWORLD( matTransformationWORLDtoLEFT.inverse( ) );

    //ds compute landmark lost since last (negative if we see more landmarks than before)
    const UIDLandmark uNumberOfVisibleLandmarks = vecMeasurements->size( );
    const int32_t iLandmarksLost                = m_uNumberofVisibleLandmarksLAST-uNumberOfVisibleLandmarks;

    //ds if we lose more than 75% landmarks in one frame
    if( 0.75 < static_cast< double >( iLandmarksLost )/m_uNumberofVisibleLandmarksLAST )
    {
        //ds if not capped already
        if( 20 > m_uCountInstability )
        {
            m_uCountInstability += 5;
        }

        std::printf( "[%06lu]<CTrackerStereoMotionModel>(_trackLandmarks) lost track (landmarks visible: %3lu lost: %3i), total delta: %f (%f %f %f), motion scaling: %f\n", m_uFrameCount, uNumberOfVisibleLandmarks, iLandmarksLost, m_vecVelocityAngularFilteredLAST.squaredNorm( ), m_vecVelocityAngularFilteredLAST.x( ), m_vecVelocityAngularFilteredLAST.y( ), m_vecVelocityAngularFilteredLAST.z( ), dMotionScaling );
        //m_uWaitKeyTimeoutMS = 0;
    }



    //ds estimate acceleration in current WORLD frame (necessary to filter gravity)
    const Eigen::Matrix3d matRotationIMUtoWORLD( matTransformationLEFTtoWORLD.linear( )*m_pCameraLEFT->m_matRotationIMUtoCAMERA );
    const CLinearAccelerationWORLD vecLinearAccelerationWORLD( matRotationIMUtoWORLD*p_vecLinearAcceleration );
    const CLinearAccelerationWORLD vecLinearAccelerationWORLDFiltered( CIMUInterpolator::getLinearAccelerationFiltered( vecLinearAccelerationWORLD ) );

    //ds get angular velocity filtered
    const CAngularVelocityIMU vecAngularVelocityFiltered( CIMUInterpolator::getAngularVelocityFiltered( p_vecAngularVelocity ) );

    //ds update IMU input references
    m_vecLinearAccelerationFilteredLAST = matTransformationWORLDtoLEFT.linear( )*vecLinearAccelerationWORLDFiltered;
    m_vecVelocityAngularFilteredLAST    = m_pCameraLEFT->m_matRotationIMUtoCAMERA*vecAngularVelocityFiltered;

    //ds current translation
    m_vecPositionLAST    = m_vecPositionCurrent;
    m_vecPositionCurrent = matTransformationLEFTtoWORLD.translation( );

    //ds update reference
    m_uNumberofVisibleLandmarksLAST = uNumberOfVisibleLandmarks;

    //ds display measurements
    for( const CMeasurementLandmark* pMeasurement: *vecMeasurements )
    {
        //ds compute green brightness based on depth (further away -> darker)
        const uint8_t uGreenValue = 255-pMeasurement->vecPointXYZLEFT.z( )/100.0*255;
        cv::circle( matDisplayLEFT, pMeasurement->ptUVLEFT, 2, CColorCodeBGR( 0, uGreenValue, 0 ), -1 );
        cv::circle( matDisplayRIGHT, pMeasurement->ptUVRIGHT, 2, CColorCodeBGR( 0, uGreenValue, 0 ), -1 );

        //ds get 3d position in current camera frame
        const CPoint3DCAMERA vecXYZLEFT( matTransformationWORLDtoLEFT*pMeasurement->vecPointXYZWORLDOptimized );

        //ds also draw reprojections
        cv::circle( matDisplayLEFT, m_pCameraLEFT->getProjection( vecXYZLEFT ), 6, CColorCodeBGR( 0, uGreenValue, 0 ), 1 );
        cv::circle( matDisplayRIGHT, m_pCameraRIGHT->getProjection( vecXYZLEFT ), 6, CColorCodeBGR( 0, uGreenValue, 0 ), 1 );
    }

    //ds update scene in viewer
    m_prFrameLEFTtoWORLD = std::pair< bool, Eigen::Isometry3d >( false, matTransformationLEFTtoWORLD );

    //ds accumulate orientation
    m_vecCameraOrientationAccumulated += p_vecRotationTotal;

    //ds position delta
    m_dTranslationDeltaSquaredNormCurrent = ( m_vecPositionCurrent-m_vecPositionKeyFrameLAST ).squaredNorm( );

    //ds translation history
    m_vecTranslationDeltas.push_back( m_vecPositionCurrent-m_vecPositionLAST );
    m_vecRotations.push_back( p_vecRotationTotal );

    //ds current element count
    const std::vector< Eigen::Vector3d >::size_type uMaximumIndex = m_vecTranslationDeltas.size( )-1;
    const std::vector< Eigen::Vector3d >::size_type uMinimumIndex = uMaximumIndex-m_uIMULogbackSize+1;

    //ds remove first and add recent
    m_vecGradientXYZ -= m_vecTranslationDeltas[uMinimumIndex];
    m_vecGradientXYZ += m_vecTranslationDeltas[uMaximumIndex];

    //ds add a keyframe if valid
    if( m_dTranslationDeltaForKeyFrameMetersL2 < m_dTranslationDeltaSquaredNormCurrent       ||
        m_dAngleDeltaForKeyFrameRadiansL2 < m_vecCameraOrientationAccumulated.squaredNorm( ) ||
        m_uFrameDifferenceForKeyFrame < m_uFrameCount-m_uFrameKeyFrameLAST                   )
    {
        //ds compute cloud for current keyframe (also optimizes landmarks!)
        const std::shared_ptr< const std::vector< CDescriptorVectorPoint3DWORLD > > vecCloud = m_cMatcher.getCloudForVisibleOptimizedLandmarks( m_uFrameCount );

        //ds if the number of points in the cloud is sufficient
        if( m_uMinimumLandmarksForKeyFrame < vecCloud->size( ) )
        {
            //ds register keyframing to matcher
            m_cMatcher.setKeyFrameToVisibleLandmarks( );

            //ds current "id"
            const UIDKeyFrame uIDKeyFrameCurrent = m_vecKeyFrames->size( );

            //ds stop time for loop closure checking
            const double dStartTimeLoopClosingSeconds = CLogger::getTimeSeconds( );

            //ds detect loop closures
            const std::vector< const CKeyFrame::CMatchICP* > vecLoopClosures( _getLoopClosuresForKeyFrame( uIDKeyFrameCurrent, matTransformationLEFTtoWORLD, vecCloud, m_dLoopClosingRadiusSquaredMeters, m_uMinimumNumberOfMatchesLoopClosure ) );

            //ds update time
            m_dTotalLoopClosingDurationSeconds += ( CLogger::getTimeSeconds( )-dStartTimeLoopClosingSeconds );

            //ds if we found closures
            if( !vecLoopClosures.empty( ) )
            {
                ++m_uLoopClosingKeyFramesInQueue;
            }

            //ds no keyframes before
            assert( 9 < m_vecRotations.size( ) );

            //ds get average rotation over the last n frames
            double dSumRotationAverage = 0.0;
            for( std::vector< Eigen::Vector3d >::size_type u = m_vecRotations.size( )-10; u < m_vecRotations.size( ); ++u )
            {
                dSumRotationAverage += m_vecRotations[u].squaredNorm( );
            }
            dSumRotationAverage /= 10;
            const double dInformationFactor = 1.0/( 1.0+1000*dSumRotationAverage );

            //ds create new frame
            m_vecKeyFrames->push_back( new CKeyFrame( uIDKeyFrameCurrent,
                                                      m_uFrameCount,
                                                      matTransformationLEFTtoWORLD,
                                                      p_vecLinearAcceleration.normalized( ),
                                                      *vecMeasurements,
                                                      vecCloud,
                                                      vecLoopClosures,
                                                      dInformationFactor ) );

            //ds check if we are not in a critical situation before triggering an optimization
            if( m_dMaximumMotionScalingForOptimization > ( dMotionScaling+m_dMotionScalingLAST )/2.0 && 0 == m_uCountInstability )
            {
                //ds check if optimization is required (based on key frame id or loop closing) TODO beautify this case
                if( m_uIDDeltaKeyFrameForOptimization < uIDKeyFrameCurrent-m_uIDProcessedKeyFrameLAST                                                                              ||
                   ( m_uLoopClosingKeyFrameWaitingQueue < m_uLoopClosingKeyFramesInQueue && m_uIDDeltaKeyFrameForOptimization < uIDKeyFrameCurrent-m_uIDLoopClosureOptimizedLAST ) )
                {
                    //ds compute first used landmarks in the keyframe chunk TODO: sometimes invalid ID's returned - FIXIT
                    const std::vector< CLandmark* >::size_type uIDBeginLandmark = std::min( m_vecKeyFrames->at( m_uIDProcessedKeyFrameLAST )->vecMeasurements.front( )->uID, m_vecLandmarks->back( )->uID );

                    //ds optimize the segment (in global frame)
                    //m_cOptimizer.optimizeTail( m_uIDProcessedKeyFrameLAST );
                    //m_cOptimizer.optimizeTailLoopClosuresOnly( m_uIDProcessedKeyFrameLAST );
                    m_cGraphOptimizer.optimizeContinuous( m_uFrameCount, m_uIDProcessedKeyFrameLAST, uIDBeginLandmark, m_vecTranslationToG2o, m_uLoopClosingKeyFramesInQueue );

                    //ds has to work
                    assert( m_vecKeyFrames->back( )->bIsOptimized );

                    //ds compute transformation induced through optimization
                    //const Eigen::Isometry3d matTransformationLEFTtoLEFTOptimized = m_vecKeyFrames->back( )->matTransformationLEFTtoWORLD.inverse( )*matTransformationLEFTtoWORLD;

                    //ds adjust angular velocity and acceleration
                    //m_vecVelocityAngularFilteredLAST    += CMiniVisionToolbox::toOrientationRodrigues( matTransformationLEFTtoLEFTOptimized.linear( ) )/( p_dDeltaTimeSeconds );
                    //m_vecLinearAccelerationFilteredLAST += matTransformationLEFTtoLEFTOptimized.translation( )/( 0.5*p_dDeltaTimeSeconds*p_dDeltaTimeSeconds );

                    //ds if loop closure triggered
                    if( 0 < m_uLoopClosingKeyFramesInQueue )
                    {
                        m_uIDLoopClosureOptimizedLAST = uIDKeyFrameCurrent;
                    }

                    //ds update transformations with optimized ones (mock identity transform from LAST as we also move the landmarks)
                    m_uLoopClosingKeyFramesInQueue     = 0;
                    m_uIDProcessedKeyFrameLAST         = uIDKeyFrameCurrent+1; //ds +1 for continuous optimization
                    matTransformationLEFTtoWORLD       = m_vecKeyFrames->back( )->matTransformationLEFTtoWORLD;
                    m_vecPositionCurrent               = matTransformationLEFTtoWORLD.translation( );
                    matTransformationWORLDtoLEFT       = matTransformationLEFTtoWORLD.inverse( );
                    m_matTransformationWORLDtoLEFTLAST = matTransformationWORLDtoLEFT;

                    //ds update linear acceleration estimate on new transform (rotation is not filtered)
                    const Eigen::Isometry3d matTransformationIMUtoWORLDOptimized( matTransformationLEFTtoWORLD*m_pCameraLEFT->m_matTransformationIMUtoCAMERA );
                    m_vecLinearAccelerationFilteredLAST = matTransformationWORLDtoLEFT.linear( )*CIMUInterpolator::getLinearAccelerationFiltered( matTransformationIMUtoWORLDOptimized.linear( )*p_vecLinearAcceleration );

                    /*ds if we are in a continuous situation and not too recent
                    if( 125.0 < m_vecGradientXYZ.squaredNorm( ) )
                    {
                        //ds if not close to the previous world frame
                        if( 2000.0 < ( m_vecPositionCurrent-m_vecTranslationToG2o ).squaredNorm( ) )
                        {
                            //std::printf( "[%06lu]<CTrackerStereoMotionModel>(_trackLandmarks) steady mode entered, absolute delta [%06lu-%06li]: %f\n", m_uFrameCount, uMaximumIndex-m_uIMULogbackSize, static_cast< int64_t >( uMinimumIndex-m_uIMULogbackSize ), m_vecGradientXYZ.squaredNorm( ) );

                            //ds update local frame
                            _updateWORLDFrame( m_vecPositionCurrent );

                            //ds update g2o distance
                            m_vecTranslationToG2o += m_vecPositionCurrent;

                            //ds apply reset position
                            matTransformationLEFTtoWORLD.translation( ) = Eigen::Vector3d::Zero( );
                            matTransformationWORLDtoLEFT                = matTransformationLEFTtoWORLD.inverse( );
                            m_matTransformationWORLDtoLEFTLAST          = matTransformationWORLDtoLEFT;
                            m_vecPositionCurrent                        = matTransformationLEFTtoWORLD.translation( );
                            m_vecPositionLAST                           = m_vecPositionCurrent;

                            //ds clear active measurements for the landmarks - otherwise we might get inconsistencies with the inner landmark optimization
                            m_cMatcher.clearActiveLandmarksMeasurements( matTransformationWORLDtoLEFT );
                        }

                        //ds always reinitialize translation window
                        _initializeTranslationWindow( );
                    }*/
                }
            }

            //ds update references
            m_vecPositionKeyFrameLAST         = m_vecPositionCurrent;
            m_vecCameraOrientationAccumulated = Eigen::Vector3d::Zero( );
            m_uFrameKeyFrameLAST              = m_uFrameCount;

            //ds update scene in viewer with keyframe transformation
            m_prFrameLEFTtoWORLD = std::pair< bool, Eigen::Isometry3d >( true, matTransformationLEFTtoWORLD );
        }
        /*else
        {
            std::printf( "<CTrackerStereoMotionModel>(_trackLandmarks) not enough points for keyframing: %lu\n", vecCloud->size( ) );
        }*/
    }

    //ds frame available for viewer
    m_bIsFrameAvailable = true;

    //ds check if we have to detect new landmarks
    if( m_uVisibleLandmarksMinimum > m_uNumberofVisibleLandmarksLAST || m_uMaximumNumberOfFramesWithoutDetection < m_uNumberOfFramesWithoutDetection )
    {
        //ds clean the lower display (to show detection details)
        cv::hconcat( matDisplayLEFTClean, matDisplayRIGHTClean, m_matDisplayLowerReference );

        //ds detect new landmarks
        _addNewLandmarks( p_matImageLEFT, p_matImageRIGHT, matTransformationWORLDtoLEFT, matTransformationLEFTtoWORLD, m_matDisplayLowerReference );

        //ds reset counter
        m_uNumberOfFramesWithoutDetection = 0;
    }
    else
    {
        ++m_uNumberOfFramesWithoutDetection;
    }

    //ds build display mat
    cv::Mat matDisplayUpper = cv::Mat( m_pCameraSTEREO->m_uPixelHeight, 2*m_pCameraSTEREO->m_uPixelWidth, CV_8UC3 );
    cv::hconcat( matDisplayLEFT, matDisplayRIGHT, matDisplayUpper );
    _drawInfoBox( matDisplayUpper, dMotionScaling );
    cv::Mat matDisplayComplete = cv::Mat( 2*m_pCameraSTEREO->m_uPixelHeight, 2*m_pCameraSTEREO->m_uPixelWidth, CV_8UC3 );
    cv::vconcat( matDisplayUpper, m_matDisplayLowerReference, matDisplayComplete );

    //ds check if unstable
    if( 0 < m_uCountInstability )
    {
        //ds draw red frame around display
        cv::rectangle( matDisplayComplete, cv::Point2i( 0, 17 ), cv::Point2i( 2*m_pCameraSTEREO->m_uPixelWidth, m_pCameraSTEREO->m_uPixelHeight ), CColorCodeBGR( 0, 0, 255 ), m_uCountInstability );
    }

    //ds display
    cv::imshow( "vi_mapper [L|R]", matDisplayComplete );

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
                    m_eMode = ePlaybackStepwise;
                    std::printf( "[%06lu]<CTrackerStereoMotionModel>(_trackLandmarks) switched to stepwise mode\n", m_uFrameCount );
                }
                else
                {
                    //ds switch to benchmark mode
                    m_uWaitKeyTimeoutMS = 1;
                    m_eMode = ePlaybackBenchmark;
                    std::printf( "[%06lu]<CTrackerStereoMotionModel>(_trackLandmarks) switched back to benchmark mode\n", m_uFrameCount );
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
    m_dDistanceTraveledMeters           += m_vecTranslationDeltas.back( ).norm( );
    m_dMotionScalingLAST                 = dMotionScaling;

    //ds log final status (after potential optimization)
    //CLogger::CLogIMUInput::addEntry( m_uFrameCount, vecLinearAccelerationWORLD, vecLinearAccelerationWORLDFiltered, p_vecAngularVelocity, vecAngularVelocityFiltered );
    //CLogger::CLogTrajectory::addEntry( m_uFrameCount, m_vecPositionCurrent, Eigen::Quaterniond( matTransformationLEFTtoWORLD.linear( ) ) );
}

void CTrackerStereoMotionModel::_addNewLandmarks( const cv::Mat& p_matImageLEFT,
                                                  const cv::Mat& p_matImageRIGHT,
                                                  const Eigen::Isometry3d& p_matTransformationWORLDtoLEFT,
                                                  const Eigen::Isometry3d& p_matTransformationLEFTtoWORLD,
                                                  cv::Mat& p_matDisplaySTEREO )
{
    //ds precompute intrinsics
    const MatrixProjection matProjectionWORLDtoLEFT( m_pCameraLEFT->m_matProjection*p_matTransformationWORLDtoLEFT.matrix( ) );
    const MatrixProjection matProjectionWORLDtoRIGHT( m_pCameraRIGHT->m_matProjection*p_matTransformationWORLDtoLEFT.matrix( ) );

    //ds solution holder
    std::shared_ptr< std::vector< CLandmark* > > vecNewLandmarks( std::make_shared< std::vector< CLandmark* > >( ) );

    //ds key points buffer
    std::vector< cv::KeyPoint > vecKeyPoints;

    //const std::shared_ptr< std::vector< cv::KeyPoint > > vecKeyPoints( m_cDetector.detectKeyPointsTilewise( p_matImageLEFT, matMask ) );
    m_pDetector->detect( p_matImageLEFT, vecKeyPoints, m_cMatcher.getMaskActiveLandmarks( p_matTransformationWORLDtoLEFT, p_matDisplaySTEREO ) );

    //ds compute descriptors for the keypoints
    CDescriptors matReferenceDescriptors;
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
            const CMatchTriangulation cMatchRIGHT( m_pTriangulator->getPointTriangulatedCompactInRIGHT( p_matImageRIGHT, cKeyPointLEFT, matDescriptorLEFT ) );
            const CPoint3DCAMERA vecPointTriangulatedLEFT( cMatchRIGHT.vecPointXYZCAMERA );
            const CDescriptor matDescriptorRIGHT( cMatchRIGHT.matDescriptorCAMERA );

            //ds check depth
            const double dDepthMeters( vecPointTriangulatedLEFT.z( ) );

            //ds check if point is in front of camera an not more than a defined distance away
            if( m_dMinimumDepthMeters < dDepthMeters && m_dMaximumDepthMeters > dDepthMeters )
            {
                //ds landmark right
                const cv::Point2f ptLandmarkRIGHT( cMatchRIGHT.ptUVCAMERA );

                //ds allocate a new landmark and add the current position
                CLandmark* pLandmark( new CLandmark( m_uAvailableLandmarkID,
                                                     matDescriptorLEFT,
                                                     cMatchRIGHT.matDescriptorCAMERA,
                                                     cKeyPointLEFT.size,
                                                     ptLandmarkLEFT,
                                                     ptLandmarkRIGHT,
                                                     vecPointTriangulatedLEFT,
                                                     p_matTransformationLEFTtoWORLD,
                                                     p_matTransformationWORLDtoLEFT,
                                                     m_pCameraLEFT->m_matProjection,
                                                     m_pCameraRIGHT->m_matProjection,
                                                     matProjectionWORLDtoLEFT,
                                                     matProjectionWORLDtoRIGHT,
                                                     m_uFrameCount ) );

                //ds log creation
                //CLogger::CLogLandmarkCreation::addEntry( m_uFrameCount, pLandmark, dDepthMeters, ptLandmarkLEFT, ptLandmarkRIGHT );

                //ds add to newly detected
                vecNewLandmarks->push_back( pLandmark );

                //ds next landmark id
                ++m_uAvailableLandmarkID;

                //ds draw detected point
                cv::line( p_matDisplaySTEREO, ptLandmarkLEFT, cv::Point2f( ptLandmarkLEFT.x+m_pCameraSTEREO->m_uPixelWidth, ptLandmarkLEFT.y ), CColorCodeBGR( 175, 175, 175 ) );
                cv::circle( p_matDisplaySTEREO, ptLandmarkLEFT, 2, CColorCodeBGR( 0, 255, 0 ), -1 );
                //cv::circle( p_matDisplay, ptLandmarkLEFT, cLandmark->dKeyPointSize, CColorCodeBGR( 255, 0, 0 ), 1 );
                cv::putText( p_matDisplaySTEREO, std::to_string( pLandmark->uID ) , cv::Point2d( ptLandmarkLEFT.x+pLandmark->dKeyPointSize, ptLandmarkLEFT.y+pLandmark->dKeyPointSize ), cv::FONT_HERSHEY_PLAIN, 0.5, CColorCodeBGR( 0, 0, 255 ) );

                //ds draw reprojection of triangulation
                cv::circle( p_matDisplaySTEREO, cv::Point2d( ptLandmarkRIGHT.x+m_pCameraSTEREO->m_uPixelWidth, ptLandmarkRIGHT.y ), 2, CColorCodeBGR( 255, 0, 0 ), -1 );
            }
            else
            {
                cv::circle( p_matDisplaySTEREO, ptLandmarkLEFT, 2, CColorCodeBGR( 0, 255, 255 ), -1 );
                //cv::circle( p_matDisplay, ptLandmarkLEFT, cKeyPoint.size, CColorCodeBGR( 0, 0, 255 ) );

                //std::printf( "<CTrackerStereoMotionModel>(_addNewLandmarks) could not find match for keypoint (invalid depth: %f m)\n", vecPointTriangulatedLEFT(2) );
            }
        }
        catch( const CExceptionNoMatchFound& p_cException )
        {
            cv::circle( p_matDisplaySTEREO, ptLandmarkLEFT, 2, CColorCodeBGR( 0, 0, 255 ), -1 );
            //cv::circle( p_matDisplay, ptLandmarkLEFT, cKeyPoint.size, CColorCodeBGR( 0, 0, 255 ) );

            //std::printf( "<CTrackerStereoMotionModel>(_addNewLandmarks) could not find match for keypoint (%s)\n", p_cException.what( ) );
        }
    }

    //ds if we couldnt find new landmarks
    if( vecNewLandmarks->empty( ) )
    {
        //std::printf( "<CTrackerStereoMotionModel>(_getNewLandmarks) unable to detect new landmarks\n" );
    }
    else
    {
        //ds all visible in this frame
        m_uNumberofVisibleLandmarksLAST += vecNewLandmarks->size( );

        //ds add to permanent reference holder (this will copy the landmark references)
        m_vecLandmarks->insert( m_vecLandmarks->end( ), vecNewLandmarks->begin( ), vecNewLandmarks->end( ) );

        //ds add this measurement point to the epipolar matcher (which will remove references from its detection point -> does not affect the landmarks main vector)
        m_cMatcher.addDetectionPoint( p_matTransformationLEFTtoWORLD, vecNewLandmarks );

        //std::printf( "<CTrackerStereoMotionModel>(_getNewLandmarks) added new landmarks: %lu/%lu\n", vecNewLandmarks->size( ), vecKeyPoints.size( ) );
    }
}

const std::vector< const CKeyFrame::CMatchICP* > CTrackerStereoMotionModel::_getLoopClosuresForKeyFrame( const UIDKeyFrame& p_uID,
                                                                                                         const Eigen::Isometry3d& p_matTransformationLEFTtoWORLD,
                                                                                                         const std::shared_ptr< const std::vector< CDescriptorVectorPoint3DWORLD > > p_vecCloudQuery,
                                                                                                         const double& p_dSearchRadiusMeters,
                                                                                                         const std::vector< CMatchCloud >::size_type& p_uMinimumNumberOfMatchesLoopClosure )
{
    //ds count attempts
    //uint32_t uOptimizationAttempts = 0;

    //ds potential keyframes for loop closing
    std::vector< const CKeyFrame* > vecPotentialClosureKeyFrames;

    //ds check all keyframes for distance
    for( const CKeyFrame* pKeyFrameReference: *m_vecKeyFrames )
    {
        //ds break if near to current id (forward closing)
        if( m_uMinimumLoopClosingKeyFrameDistance > p_uID-pKeyFrameReference->uID )
        {
            break;
        }

        //ds if the distance is acceptable
        if( p_dSearchRadiusMeters > ( pKeyFrameReference->matTransformationLEFTtoWORLD.translation( )-p_matTransformationLEFTtoWORLD.translation( ) ).squaredNorm( ) )
        {
            //ds add the keyframe to the loop closing pool
            vecPotentialClosureKeyFrames.push_back( pKeyFrameReference );
        }
    }

    //ds solution vector
    std::vector< const CKeyFrame::CMatchICP* > vecLoopClosures( 0 );

    //std::printf( "<CTrackerStereoMotionModel>(_getLoopClosureKeyFrameFCFS) checking for closure in potential keyframes: %lu\n", vecPotentialClosureKeyFrames.size( ) );

    //ds compare current cloud against previous ones to enable loop closure (skipping the keyframes added just before)
    for( const CKeyFrame* pKeyFrameReference: vecPotentialClosureKeyFrames )
    {
        //ds get matches
        std::shared_ptr< const std::vector< CMatchCloud > > vecMatches( pKeyFrameReference->getMatches( p_vecCloudQuery ) );

        //ds if we have a suffient amount of matches
        if( p_uMinimumNumberOfMatchesLoopClosure < vecMatches->size( ) )
        {
            //std::printf( "<CTrackerStereoMotionModel>(_getLoopClosureKeyFrameFCFS) found closure: [%06lu] > [%06lu] matches: %lu\n", p_uID, pKeyFrameReference->uID, vecMatches->size( ) );

            //ds transformation between this keyframe and the loop closure one (take current measurement as prior)
            Eigen::Isometry3d matTransformationToClosure( pKeyFrameReference->matTransformationLEFTtoWORLD.inverse( )*p_matTransformationLEFTtoWORLD );
            matTransformationToClosure.translation( ) = Eigen::Vector3d::Zero( );
            //const Eigen::Isometry3d matTransformationToClosureInitial( matTransformationToClosure );

            //ds 1mm for convergence
            const double dErrorDeltaForConvergence      = 1e-5;
            double dErrorSquaredTotalPrevious           = 0.0;
            const double dMaximumErrorForInlier         = 0.25; //0.25
            const double dMaximumErrorAverageForClosure = 0.2; //0.1

            //ds LS setup
            Eigen::Matrix< double, 6, 6 > matH;
            Eigen::Matrix< double, 6, 1 > vecB;
            Eigen::Matrix3d matOmega( Eigen::Matrix3d::Identity( ) );

            //std::printf( "<CTrackerStereoMotionModel>(_getLoopClosureKeyFrameFCFS) t: %4.1f %4.1f %4.1f > ", matTransformationToClosure.translation( ).x( ), matTransformationToClosure.translation( ).y( ), matTransformationToClosure.translation( ).z( ) );

            //ds run least-squares maximum 100 times
            for( uint8_t uLS = 0; uLS < 100; ++uLS )
            {
                //ds error
                double dErrorSquaredTotal = 0.0;
                uint32_t uInliers         = 0;

                //ds initialize setup
                matH.setZero( );
                vecB.setZero( );

                //ds for all the points
                for( const CMatchCloud& cMatch: *vecMatches )
                {
                    //ds compute projection into closure
                    const CPoint3DCAMERA vecPointXYZQuery( matTransformationToClosure*cMatch.cPointQuery.vecPointXYZCAMERA );
                    if( 0.0 < vecPointXYZQuery.z( ) )
                    {
                        assert( 0.0 < cMatch.cPointReference.vecPointXYZCAMERA.z( ) );

                        //ds adjust omega to inverse depth value (the further away the point, the less weight)
                        matOmega(2,2) = 1.0/( cMatch.cPointReference.vecPointXYZCAMERA.z( ) );

                        //ds compute error
                        const Eigen::Vector3d vecError( vecPointXYZQuery-cMatch.cPointReference.vecPointXYZCAMERA );

                        //ds update chi
                        const double dErrorSquared = vecError.transpose( )*matOmega*vecError;

                        //ds check if outlier
                        double dWeight = 1.0;
                        if( dMaximumErrorForInlier < dErrorSquared )
                        {
                            dWeight = dMaximumErrorForInlier/dErrorSquared;
                        }
                        else
                        {
                            ++uInliers;
                        }
                        dErrorSquaredTotal += dWeight*dErrorSquared;

                        //ds get the jacobian of the transform part = [I 2*skew(T*modelPoint)]
                        Eigen::Matrix< double, 3, 6 > matJacobianTransform;
                        matJacobianTransform.setZero( );
                        matJacobianTransform.block<3,3>(0,0).setIdentity( );
                        matJacobianTransform.block<3,3>(0,3) = -2*CMiniVisionToolbox::getSkew( vecPointXYZQuery );

                        //ds precompute transposed
                        const Eigen::Matrix< double, 6, 3 > matJacobianTransformTransposed( matJacobianTransform.transpose( ) );

                        //ds accumulate
                        matH += dWeight*matJacobianTransformTransposed*matOmega*matJacobianTransform;
                        vecB += dWeight*matJacobianTransformTransposed*matOmega*vecError;
                    }
                }

                //ds solve the system and update the estimate
                matTransformationToClosure = CMiniVisionToolbox::getTransformationFromVector( matH.ldlt( ).solve( -vecB ) )*matTransformationToClosure;

                //ds enforce rotation symmetry
                const Eigen::Matrix3d matRotation        = matTransformationToClosure.linear( );
                Eigen::Matrix3d matRotationSquared       = matRotation.transpose( )*matRotation;
                matRotationSquared.diagonal( ).array( ) -= 1;
                matTransformationToClosure.linear( )    -= 0.5*matRotation*matRotationSquared;

                //ds check if converged (no descent required)
                if( dErrorDeltaForConvergence > std::fabs( dErrorSquaredTotalPrevious-dErrorSquaredTotal ) )
                {
                    //ds compute average error
                    const double dErrorAverage = dErrorSquaredTotal/vecMatches->size( );

                    //ds if the solution is acceptable
                    if( dMaximumErrorAverageForClosure > dErrorAverage && p_uMinimumNumberOfMatchesLoopClosure < uInliers )
                    {
                        //std::printf( "%4.1f %4.1f %4.1f | ", matTransformationToClosure.translation( ).x( ), matTransformationToClosure.translation( ).y( ), matTransformationToClosure.translation( ).z( ) );
                        //std::printf( "system converged in %2u iterations, average error: %5.3f (inliers: %2u) - MATCH\n", uLS, dErrorAverage, uInliers );
                        //std::printf( "<CTrackerStereoMotionModel>(_getLoopClosuresKeyFrameFCFS) found closure: [%06lu] > [%06lu] (matches: %3lu, iterations: %2u, average error: %5.3f, inliers: %2u)\n",
                        //             p_uID, pKeyFrameReference->uID, vecMatches->size( ), uLS, dErrorAverage, uInliers );
                        vecLoopClosures.push_back( new CKeyFrame::CMatchICP( pKeyFrameReference, matTransformationToClosure, vecMatches ) );
                        break;
                    }
                    else
                    {
                        //ds keep looping through keyframes
                        //std::printf( "%4.1f %4.1f %4.1f | ", matTransformationToClosure.translation( ).x( ), matTransformationToClosure.translation( ).y( ), matTransformationToClosure.translation( ).z( ) );
                        //std::printf( "system converged in %2u iterations, average error: %5.3f (inliers: %2u) - discarded\n", uLS, dErrorAverage, uInliers );
                        break;
                    }
                }
                else
                {
                    dErrorSquaredTotalPrevious = dErrorSquaredTotal;
                }

                //ds if not converged
                if( 99 == uLS )
                {
                    //std::printf( "system did not converge\n" );
                }
            }
        }
    }

    //ds return found closures
    return vecLoopClosures;
}

void CTrackerStereoMotionModel::_updateWORLDFrame( const Eigen::Vector3d& p_vecTranslationWORLD )
{
    std::printf( "[%06lu]<CTrackerStereoMotionModel>(_updateWORLDFrame) refreshing WORLD frame - ", m_uFrameCount );

    //ds set initial frame in g2o
    //m_cGraphOptimizer.updateSTART( p_vecTranslationWORLD );

    //ds move all keyframes
    for( CKeyFrame* pKeyFrame: *m_vecKeyFrames )
    {
        pKeyFrame->matTransformationLEFTtoWORLD.translation( ) -= p_vecTranslationWORLD;

        //ds update in g2o
        //m_cGraphOptimizer.updateEstimate( pKeyFrame );
    }

    //ds move all landmarks
    for( CLandmark* pLandmark: *m_vecLandmarks )
    {
        pLandmark->vecPointXYZOptimized -= p_vecTranslationWORLD;

        //ds update in g2o
        //m_cGraphOptimizer.updateEstimate( pLandmark );
    }

    //ds done
    std::printf( "complete!\n" );
}

void CTrackerStereoMotionModel::_initializeTranslationWindow( )
{
    //ds reinitialize translation window
    m_vecTranslationDeltas.clear( );
    for( std::vector< Eigen::Vector3d >::size_type u = 0; u < m_uIMULogbackSize; ++u )
    {
        m_vecTranslationDeltas.push_back( Eigen::Vector3d::Zero( ) );
    }
    m_vecGradientXYZ = Eigen::Vector3d::Zero( );
}

void CTrackerStereoMotionModel::_shutDown( )
{
    m_bIsShutdownRequested = true;
    std::printf( "[%06lu]<CTrackerStereoMotionModel>(_shutDown) termination requested, <CTrackerStereoMotionModel> disabled\n", m_uFrameCount );
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

void CTrackerStereoMotionModel::_drawInfoBox( cv::Mat& p_matDisplay, const double& p_dMotionScaling ) const
{
    char chBuffer[1024];

    switch( m_eMode )
    {
        case ePlaybackStepwise:
        {
            std::snprintf( chBuffer, 1024, "[%13.2f|%05lu] STEPWISE | X: %5.1f Y: %5.1f Z: %5.1f DELTA L2: %4.2f MOTION: %4.2f | LANDMARKS V: %3lu (%3lu,%3lu,%3lu) F: %4lu I: %4lu TOTAL: %4lu | DETECTIONS: %2lu(%3lu) | KFs: %3lu | OPTs: %02u",
                           m_dTimestampLASTSeconds, m_uFrameCount,
                           m_vecPositionCurrent.x( ), m_vecPositionCurrent.y( ), m_vecPositionCurrent.z( ), m_dTranslationDeltaSquaredNormCurrent, p_dMotionScaling,
                           m_uNumberofVisibleLandmarksLAST, m_cMatcher.getNumberOfDetectionsPoseOptimizationDirect( ), m_cMatcher.getNumberOfDetectionsPoseOptimizationDetection( ), m_cMatcher.getNumberOfDetectionsEpipolar( ), m_cMatcher.getNumberOfFailedLandmarkOptimizations( ), m_cMatcher.getNumberOfInvalidLandmarksTotal( ), m_vecLandmarks->size( ),
                           m_cMatcher.getNumberOfDetectionPointsActive( ), m_cMatcher.getNumberOfDetectionPointsTotal( ),
                           m_vecKeyFrames->size( ), m_cGraphOptimizer.getNumberOfOptimizations( ) );
            break;
        }
        case ePlaybackBenchmark:
        {
            std::snprintf( chBuffer, 1024, "[%13.2f|%05lu] BENCHMARK FPS: %4.1f | X: %5.1f Y: %5.1f Z: %5.1f DELTA L2: %4.2f SCALING: %4.2f | LANDMARKS V: %3lu (%3lu,%3lu,%3lu) F: %4lu I: %4lu TOTAL: %4lu | DETECTIONS: %2lu(%3lu) | KFs: %3lu | OPTs: %02u",
                           m_dTimestampLASTSeconds, m_uFrameCount, m_dPreviousFrameRate,
                           m_vecPositionCurrent.x( ), m_vecPositionCurrent.y( ), m_vecPositionCurrent.z( ), m_dTranslationDeltaSquaredNormCurrent, p_dMotionScaling,
                           m_uNumberofVisibleLandmarksLAST, m_cMatcher.getNumberOfDetectionsPoseOptimizationDirect( ), m_cMatcher.getNumberOfDetectionsPoseOptimizationDetection( ), m_cMatcher.getNumberOfDetectionsEpipolar( ), m_cMatcher.getNumberOfFailedLandmarkOptimizations( ), m_cMatcher.getNumberOfInvalidLandmarksTotal( ), m_vecLandmarks->size( ),
                           m_cMatcher.getNumberOfDetectionPointsActive( ), m_cMatcher.getNumberOfDetectionPointsTotal( ),
                           m_vecKeyFrames->size( ), m_cGraphOptimizer.getNumberOfOptimizations( ) );
            break;
        }
        default:
        {
            std::printf( "[%06lu]<CTrackerStereoMotionModel>(_drawInfoBox) unsupported playback mode, no info box displayed\n", m_uFrameCount );
            break;
        }
    }


    p_matDisplay( cv::Rect( 0, 0, 2*m_pCameraSTEREO->m_uPixelWidth, 17 ) ).setTo( CColorCodeBGR( 0, 0, 0 ) );
    cv::putText( p_matDisplay, chBuffer , cv::Point2i( 2, 12 ), cv::FONT_HERSHEY_PLAIN, 0.8, CColorCodeBGR( 0, 0, 255 ) );
}
