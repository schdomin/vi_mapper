#if 0
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
    std::shared_ptr< std::vector< const CDescriptorPointCloud* > > m_vecClouds;

    //ds g2o data
    std::shared_ptr< std::vector< CKeyFrame* > > m_vecKeyFrames;
    UIDKeyFrame m_uIDProcessedKeyFrameLAST              = 0;
    UIDFrame m_uIDFrameOfKeyFrameLAST                   = 0;
    const UIDKeyFrame m_uIDDeltaKeyFrameForOptimization = 9;
    Cg2oOptimizer m_cOptimizer;

    //ds sliding window, minimum size of 10 (TODO simplify/secure configuration)
    uint8_t m_uFrameToWindowRatio                 = 10; //ds for ratio=1 every frame is added to the window (PERFORMANCE)
    UIDFrame m_uMinimumFrameDeltaForKeyFrame      = 10;
    const UIDKeyFrame m_uBufferSizeSlidingWindow  = 8192;
    std::pair< CPoint3DWORLD, CKeyFrame* >* m_arrFramesSlidingWindow;
    uint8_t m_uIndexSlidingWindow                 = 9;
    const double m_dDeltaSlidingWindowRelative    = 0.05;
    const double m_dDeltaSlidingWindowAbsolute    = 0.1;

    //ds loop closing
    const UIDKeyFrame m_uLoopClosingKeyFrameDistance       = 5;
    const UIDLandmark m_uMinimumNumberOfMatchesLoopClosure = 25;
    std::shared_ptr< std::vector< const C67DTree* > > m_vecTrees;

    //ds control
    const EPlaybackMode m_eMode;
    bool m_bIsShutdownRequested = false;

    //ds info display
    bool m_bIsFrameAvailable       = false;
    std::pair< bool, Eigen::Isometry3d > m_prFrameLEFTtoWORLD;
    bool m_bIsFrameAvailableSlidingWindow = false;
    uint32_t m_uFramesCurrentCycle        = 0;
    double m_dPreviousFrameRate           = 0.0;
    double m_dPreviousFrameTime           = 0.0;

//ds accessors
public:

    void receivevDataVI( const std::shared_ptr< txt_io::PinholeImageMessage > p_pImageLEFT,
                         const std::shared_ptr< txt_io::PinholeImageMessage > p_pImageRIGHT,
                         const std::shared_ptr< txt_io::CIMUMessage > p_pIMU );

    const uint64_t getFrameCount( ) const { return m_uFrameCount; }
    const bool isShutdownRequested( ) const { return m_bIsShutdownRequested; }
    const std::shared_ptr< std::vector< CLandmark* > > getLandmarksHandle( ) const { return m_vecLandmarks; }
    const std::shared_ptr< std::vector< CKeyFrame* > > getKeyFramesHandle( ) const { return m_vecKeyFrames; }
    const bool isFrameAvailable( ) const { return m_bIsFrameAvailable; }
    const bool isFrameAvailableSlidingWindow( ) const { return m_bIsFrameAvailableSlidingWindow; }
    const std::pair< bool, Eigen::Isometry3d > getFrameLEFTtoWORLD( ){ m_bIsFrameAvailable = false; return m_prFrameLEFTtoWORLD; }
    const UIDFrame getFrameLEFTtoWORLDSlidingWindow( ){ m_bIsFrameAvailableSlidingWindow = false; return m_uIDFrameOfKeyFrameLAST; }

    //ds postprocessing
    void saveUVDepthOrDisparity( const std::string& p_strOutfile ) const
    {
        //CBridgeG2O::saveUVDepthOrDisparity( p_strOutfile, *m_pCameraSTEREO, *m_vecLandmarks, m_vecKeyFrames );
    }
    void saveXYZ( const std::string& p_strOutfile ) const
    {
        //CBridgeG2O::saveXYZ( p_strOutfile, *m_pCameraSTEREO, *m_vecLandmarks, m_vecKeyFrames );
    }
    void saveUVDepth( const std::string& p_strOutfile ) const
    {
        //CBridgeG2O::saveUVDepth( p_strOutfile, *m_pCameraSTEREO, *m_vecLandmarks, m_vecKeyFrames );
    }
    void saveUVDisparity( const std::string& p_strOutfile ) const
    {
        //CBridgeG2O::saveUVDisparity( p_strOutfile, *m_pCameraSTEREO, *m_vecLandmarks, m_vecKeyFrames );
    }
    void saveCOMBO( const std::string& p_strOutfile ) const
    {
        //CBridgeG2O::saveCOMBO( p_strOutfile, *m_pCameraSTEREO, *m_vecLandmarks, m_vecKeyFrames );
    }

//ds helpers
private:

    void _trackLandmarks( const cv::Mat& p_matImageLEFT,
                          const cv::Mat& p_matImageRIGHT,
                          const Eigen::Isometry3d& p_matTransformationEstimateWORLDtoLEFT,
                          const Eigen::Isometry3d& p_matTransformationEstimateParallelWORLDtoLEFT,
                          const CLinearAccelerationIMU& p_vecLinearAcceleration,
                          const Eigen::Vector3d& p_vecRotationTotal );

    const std::shared_ptr< std::vector< CLandmark* > > _getNewLandmarks( const UIDFrame& p_uFrame,
                                                      cv::Mat& p_matDisplay,
                                                      const cv::Mat& p_matImageLEFT,
                                                      const cv::Mat& p_matImageRIGHT,
                                                      const Eigen::Isometry3d& p_matTransformationWORLDtoLEFT,
                                                      const Eigen::Isometry3d& p_matTransformationLEFTtoWORLD,
                                                      const Eigen::Vector3d& p_vecRotation );

    //ds loop closing
    const CKeyFrame* _getLoopClosureKeyFrame( const int64_t& p_uID, const Eigen::Isometry3d& p_matTransformationLEFTtoWORLD, cv::Mat& p_matDisplayLEFT );

    //ds sliding window keyframe detection
    const bool _containsSlidingWindowKeyFrame( ) const;

    //ds control
    void _shutDown( );
    void _updateFrameRateForInfoBox( const uint32_t& p_uFrameProbeRange = 10 );
    void _drawInfoBox( cv::Mat& p_matDisplay ) const;

};

#endif //#define CTRACKERSTEREOMOTIONMODEL_H
