#ifndef CTRACKERSTEREOMOTIONMODEL_H
#define CTRACKERSTEREOMOTIONMODEL_H

#include "txt_io/imu_message.h"
#include "txt_io/pinhole_image_message.h"
#include "txt_io/pose_message.h"

#include "vision/CStereoCamera.h"
#include "CTriangulator.h"
#include "CFundamentalMatcher.h"
#include "utility/CCloudMatcher.h"
#include "utility/CCloudStreamer.h"
#include "types/CKeyFrame.h"
#include "optimization/Cg2oOptimizer.h"

class CTrackerStereoMotionModel
{

//ds ctor/dtor
public:

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    CTrackerStereoMotionModel( const EPlaybackMode& p_eMode,
                               const uint32_t& p_uWaitKeyTimeoutMS = 1 );
    ~CTrackerStereoMotionModel( );

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
    CAngularVelocityLEFT m_vecVelocityAngularFilteredLAST;
    CLinearAccelerationLEFT m_vecLinearAccelerationFilteredLAST;
    double m_dTimestampLASTSeconds = 0.0;
    CPoint3DWORLD m_vecPositionKeyFrameLAST;
    Eigen::Vector3d m_vecCameraOrientationAccumulated;
    const double m_dTranslationDeltaForKeyFrameMetersL2;
    const double m_dAngleDeltaForOptimizationRadiansL2;
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

    const uint8_t m_uMaximumFailedSubsequentTrackingsPerLandmark;
    const uint8_t m_uVisibleLandmarksMinimum;
    cv::Mat m_matDisplayLowerReference;
    const double m_dMinimumDepthMeters;
    const double m_dMaximumDepthMeters;

    std::shared_ptr< CTriangulator > m_pTriangulator;
    CFundamentalMatcher m_cMatcher;

    //ds tracking (we use the ID counter instead of accessing the vector size every time for speed)
    UIDLandmark m_uAvailableLandmarkID          = 0;
    UIDLandmark m_uNumberofVisibleLandmarksLAST = 0;
    std::shared_ptr< std::vector< CLandmark* > > m_vecLandmarks;

    //ds g2o data
    std::shared_ptr< std::vector< CKeyFrame* > > m_vecKeyFrames;
    UIDKeyFrame m_uIDProcessedKeyFrameLAST = 0;
    UIDKeyFrame m_uIDLoopClosureLAST       = 0;
    const UIDKeyFrame m_uIDDeltaKeyFrameForOptimization;
    const UIDKeyFrame m_uIDDeltaLoopClosureForOptimization;
    Cg2oOptimizer m_cOptimizer;

    //ds loop closing
    const UIDKeyFrame m_uLoopClosingKeyFrameDistance;
    const UIDLandmark m_uMinimumNumberOfMatchesLoopClosure;

    //ds control
    const EPlaybackMode m_eMode;
    bool m_bIsShutdownRequested = false;

    //ds info display
    bool m_bIsFrameAvailable       = false;
    std::pair< bool, Eigen::Isometry3d > m_prFrameLEFTtoWORLD;
    uint32_t m_uFramesCurrentCycle        = 0;
    double m_dPreviousFrameRate           = 0.0;
    double m_dPreviousFrameTime           = 0.0;

//ds accessors
public:

    void receivevDataVI( const std::shared_ptr< txt_io::PinholeImageMessage > p_pImageLEFT,
                         const std::shared_ptr< txt_io::PinholeImageMessage > p_pImageRIGHT,
                         const std::shared_ptr< txt_io::CIMUMessage > p_pIMU );

    const UIDFrame getFrameCount( ) const { return m_uFrameCount; }
    const bool isShutdownRequested( ) const { return m_bIsShutdownRequested; }
    const std::shared_ptr< std::vector< CLandmark* > > getLandmarksHandle( ) const { return m_vecLandmarks; }
    const std::shared_ptr< std::vector< CKeyFrame* > > getKeyFramesHandle( ) const { return m_vecKeyFrames; }
    const bool isFrameAvailable( ) const { return m_bIsFrameAvailable; }
    const std::pair< bool, Eigen::Isometry3d > getFrameLEFTtoWORLD( ){ m_bIsFrameAvailable = false; return m_prFrameLEFTtoWORLD; }
    void finalize( );
    void sanitizeFiletree( ){ m_cOptimizer.clearFiles( ); }

//ds helpers
private:

    void _trackLandmarks( const cv::Mat& p_matImageLEFT,
                          const cv::Mat& p_matImageRIGHT,
                          const Eigen::Isometry3d& p_matTransformationEstimateWORLDtoLEFT,
                          const Eigen::Isometry3d& p_matTransformationEstimateParallelWORLDtoLEFT,
                          const CLinearAccelerationIMU& p_vecLinearAcceleration,
                          const Eigen::Vector3d& p_vecRotationTotal,
                          const Eigen::Vector3d& p_vecTranslationTotal,
                          const double& p_dDeltaTimeSeconds );

    const std::shared_ptr< std::vector< CLandmark* > > _getNewLandmarks( const UIDFrame& p_uFrame,
                                                      cv::Mat& p_matDisplay,
                                                      const cv::Mat& p_matImageLEFT,
                                                      const cv::Mat& p_matImageRIGHT,
                                                      const Eigen::Isometry3d& p_matTransformationWORLDtoLEFT,
                                                      const Eigen::Isometry3d& p_matTransformationLEFTtoWORLD,
                                                      const Eigen::Vector3d& p_vecRotation );

    //ds loop closing
    const CKeyFrame::CMatchICP* _getLoopClosureKeyFrameFCFS( const UIDKeyFrame& p_uID, const Eigen::Isometry3d& p_matTransformationLEFTtoWORLD, const std::shared_ptr< const std::vector< CDescriptorVectorPoint3DWORLD > > p_vecCloudQuery );

    //ds sliding window keyframe detection
    const bool _containsSlidingWindowKeyFrame( ) const;

    //ds control
    void _shutDown( );
    void _updateFrameRateForInfoBox( const uint32_t& p_uFrameProbeRange = 10 );
    void _drawInfoBox( cv::Mat& p_matDisplay ) const;

};

#endif //#define CTRACKERSTEREOMOTIONMODEL_H
