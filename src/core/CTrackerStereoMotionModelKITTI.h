#ifndef CTRACKERSTEREOMOTIONMODELKITTI_H
#define CTRACKERSTEREOMOTIONMODELKITTI_H

#include "txt_io/imu_message.h"
#include "txt_io/pinhole_image_message.h"
#include "txt_io/pose_message.h"

#include "vision/CStereoCamera.h"
#include "CTriangulator.h"
#include "CFundamentalMatcher.h"
#include "types/CKeyFrame.h"
#include "optimization/Cg2oOptimizer.h"
#include "utility/CIMUInterpolator.h"

class CTrackerStereoMotionModelKITTI
{

//ds ctor/dtor
public:

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    CTrackerStereoMotionModelKITTI( const EPlaybackMode& p_eMode,
                                    const uint32_t& p_uWaitKeyTimeoutMS = 1 );
    ~CTrackerStereoMotionModelKITTI( );

//ds members
private:

    //ds vision setup
    uint32_t m_uWaitKeyTimeoutMS;
    const std::shared_ptr< CPinholeCamera > m_pCameraLEFT;
    const std::shared_ptr< CPinholeCamera > m_pCameraRIGHT;
    const std::shared_ptr< CStereoCamera > m_pCameraSTEREO;

    //ds reference information
    UIDFrame m_uFrameCount = 0;
    Eigen::Isometry3d m_matTransformationWORLDtoLEFTLAST;
    Eigen::Isometry3d m_matTransformationLEFTLASTtoLEFTNOW;
    double m_dTimestampLASTSeconds                      = -0.1;
    CPoint3DWORLD m_vecPositionKeyFrameLAST;
    Eigen::Vector3d m_vecCameraOrientationAccumulated   = {0.0, 0.0, 0.0};
    const double m_dTranslationDeltaForKeyFrameMetersL2 = 1.0;
    const double m_dAngleDeltaForKeyFrameRadiansL2      = 0.025;
    const UIDFrame m_uFrameDifferenceForKeyFrame        = 100;
    UIDFrame m_uFrameKeyFrameLAST                       = 0;
    CPoint3DWORLD m_vecPositionCurrent;
    CPoint3DWORLD m_vecPositionLAST;
    double m_dTranslationDeltaSquaredNormCurrent = 0.0;

    //ds feature related
    //const uint32_t m_uKeyPointSize;
    const std::shared_ptr< cv::FeatureDetector > m_pDetector;
    const std::shared_ptr< cv::DescriptorExtractor > m_pExtractor;
    const std::shared_ptr< cv::DescriptorMatcher > m_pMatcher;
    const double m_dMatchingDistanceCutoffTriangulation;
    const double m_dMatchingDistanceCutoffPoseOptimization;
    const double m_dMatchingDistanceCutoffEpipolar;

    const uint8_t m_uMaximumFailedSubsequentTrackingsPerLandmark = 5;
    const uint8_t m_uVisibleLandmarksMinimum;
    const double m_dMinimumDepthMeters = 0.05;
    const double m_dMaximumDepthMeters = 1000.0;
    const UIDFrame m_uMaximumNumberOfFramesWithoutDetection = 10; //20;
    UIDFrame m_uNumberOfFramesWithoutDetection              = 0;

    std::shared_ptr< CTriangulator > m_pTriangulator;
    CFundamentalMatcher m_cMatcher;

    //ds tracking (we use the ID counter instead of accessing the vector size every time for speed)
    UIDLandmark m_uAvailableLandmarkID          = 0;
    UIDLandmark m_uNumberofVisibleLandmarksLAST = 0;
    std::shared_ptr< std::vector< CLandmark* > > m_vecLandmarks;
    //std::shared_ptr< std::array< CLandmark*, 8388608 > > m_arrLandmarks;
    uint8_t m_uCountInstability                         = 0;

    //ds g2o optimization
    std::shared_ptr< std::vector< CKeyFrame* > > m_vecKeyFrames;
    const UIDLandmark m_uMinimumLandmarksForKeyFrame    = 25; //50
    UIDKeyFrame m_uIDProcessedKeyFrameLAST              = 0;
    const UIDKeyFrame m_uIDDeltaKeyFrameForOptimization = 40; //20
    Cg2oOptimizer m_cGraphOptimizer;
    Eigen::Vector3d m_vecTranslationToG2o;

    //ds loop closing
    const UIDKeyFrame m_uMinimumLoopClosingKeyFrameDistance = 50; //20
    const UIDLandmark m_uMinimumNumberOfMatchesLoopClosure  = 20; //25
    const std::vector< CKeyFrame* >::size_type m_uLoopClosingKeyFrameWaitingQueue = 1;
    std::vector< CKeyFrame* >::size_type m_uLoopClosingKeyFramesInQueue = 0;
    std::vector< CKeyFrame::CMatchICP* >::size_type m_uLoopClosuresAfterburn = 0;
    UIDKeyFrame m_uIDLoopClosureOptimizedLAST               = 0;
    const double m_dLoopClosingRadiusSquaredMeters          = 10000.0;
    //uint32_t m_uOptimizationsWithoutLoopClosure             = 0;
    //double m_dOptimizationsWithoutLoopClosureDistanceMeters = 0.0;

    //ds robocentric world frame refreshing
    std::vector< Eigen::Vector3d > m_vecTranslationDeltas;
    const std::vector< Eigen::Vector3d >::size_type m_uIMULogbackSize = 200;
    Eigen::Vector3d m_vecGradientXYZ;
    bool m_bAvailable = true;

    //ds control
    const EPlaybackMode m_eMode;
    bool m_bIsShutdownRequested = false;

    //ds info display
    cv::Mat m_matDisplayLowerReference;
    bool m_bIsFrameAvailable = false;
    std::pair< bool, Eigen::Isometry3d > m_prFrameLEFTtoWORLD;
    uint32_t m_uFramesCurrentCycle = 0;
    double m_dPreviousFrameRate    = 0.0;
    double m_dPreviousFrameTime    = 0.0;
    double m_dDistanceTraveledMeters = 0.0;
    double m_dTotalLoopClosingDurationSeconds = 0.0;
    double m_dTotalDurationSecondsTrackingOdometry = 0.0;
    double m_dTotalDurationSecondsTrackingEpipolar = 0.0;

//ds accessors
public:

    void receivevDataVI( const std::shared_ptr< txt_io::PinholeImageMessage > p_pImageLEFT,
                         const std::shared_ptr< txt_io::PinholeImageMessage > p_pImageRIGHT );

    const UIDFrame getFrameCount( ) const { return m_uFrameCount; }
    const bool isShutdownRequested( ) const { return m_bIsShutdownRequested; }
    const std::shared_ptr< std::vector< CLandmark* > > getLandmarksHandle( ) const { return m_vecLandmarks; }
    const std::shared_ptr< std::vector< CKeyFrame* > > getKeyFramesHandle( ) const { return m_vecKeyFrames; }
    const double getLoopClosingRadius( ) const { return std::sqrt( m_dLoopClosingRadiusSquaredMeters ); }
    const bool isFrameAvailable( ) const { return m_bIsFrameAvailable; }
    const std::pair< bool, Eigen::Isometry3d > getFrameLEFTtoWORLD( ){ m_bIsFrameAvailable = false; return m_prFrameLEFTtoWORLD; }
    void finalize( );
    void sanitizeFiletree( ){ m_cGraphOptimizer.clearFiles( ); }
    const double getDistanceTraveled( ) const { return m_dDistanceTraveledMeters; }
    const double getTotalDurationOptimizationSeconds( ) const { return m_cGraphOptimizer.getTotalOptimizationDurationSeconds( ); }
    const double getTotalDurationLoopClosingSeconds( ) const { return m_dTotalLoopClosingDurationSeconds; }
    const double getTotalDurationSecondsTrackingOdometry( ) const { return m_dTotalDurationSecondsTrackingOdometry; }
    const double getTotalDurationSecondsTrackingEpipolar( ) const { return m_dTotalDurationSecondsTrackingEpipolar; }

//ds helpers
private:

    void _trackLandmarks( const cv::Mat& p_matImageLEFT,
                          const cv::Mat& p_matImageRIGHT,
                          const Eigen::Isometry3d& p_matTransformationEstimateWORLDtoLEFT,
                          const Eigen::Isometry3d& p_matTransformationEstimateParallelWORLDtoLEFT,
                          const CLinearAccelerationIMU& p_vecLinearAcceleration,
                          const CAngularVelocityIMU& p_vecAngularVelocity,
                          const Eigen::Vector3d& p_vecRotationTotal,
                          const Eigen::Vector3d& p_vecTranslationTotal,
                          const double& p_dDeltaTimeSeconds );

    void _addNewLandmarks( const cv::Mat& p_matImageLEFT,
                           const cv::Mat& p_matImageRIGHT,
                           const Eigen::Isometry3d& p_matTransformationWORLDtoLEFT,
                           const Eigen::Isometry3d& p_matTransformationLEFTtoWORLD,
                           cv::Mat& p_matDisplaySTEREO );

    //ds loop closing
    const std::vector< const CKeyFrame::CMatchICP* > _getLoopClosuresForKeyFrame( const UIDKeyFrame& p_uID,
                                                                                  const Eigen::Isometry3d& p_matTransformationLEFTtoWORLD,
                                                                                  const std::shared_ptr< const std::vector< CDescriptorVectorPoint3DWORLD > > p_vecCloudQuery,
                                                                                  const double& p_dSearchRadiusMeters,
                                                                                  const std::vector< CMatchCloud >::size_type& p_uMinimumNumberOfMatchesLoopClosure );

    //ds reference frame update
    void _updateWORLDFrame( const Eigen::Vector3d& p_vecTranslationWORLD );

    //ds translation window to detect steady states
    void _initializeTranslationWindow( );

    //ds control
    void _shutDown( );
    void _updateFrameRateForInfoBox( const uint32_t& p_uFrameProbeRange = 10 );
    void _drawInfoBox( cv::Mat& p_matDisplay, const double& p_dMotionScaling ) const;

};

#endif //#define CTRACKERSTEREOMOTIONMODELKITTI_H
