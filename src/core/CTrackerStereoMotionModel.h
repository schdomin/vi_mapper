#ifndef CTRACKERSTEREOMOTIONMODEL_H
#define CTRACKERSTEREOMOTIONMODEL_H

#include <memory>

#include "txt_io/imu_message.h"
#include "txt_io/pinhole_image_message.h"
#include "txt_io/pose_message.h"

#include "vision/CStereoCamera.h"
#include "optimization/CBridgeG2O.h"
#include "CTriangulator.h"
#include "CDetectorMonoTilewise.h"
#include "CMatcherEpipolar.h"

class CTrackerStereoMotionModel
{

//ds ctor/dtor
public:

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    CTrackerStereoMotionModel( const EPlaybackMode& p_eMode,
                    const uint32_t& p_uWaitKeyTimeout = 1 );
    ~CTrackerStereoMotionModel( );

//ds members
private:

    //ds vision setup
    const uint32_t m_uWaitKeyTimeout;
    const std::shared_ptr< CPinholeCamera > m_pCameraLEFT;
    const std::shared_ptr< CPinholeCamera > m_pCameraRIGHT;
    const std::shared_ptr< CStereoCamera > m_pCameraSTEREO;

    //ds reference information
    uint64_t m_uFrameCount;
    Eigen::Isometry3d m_matTransformationWORLDtoLEFTLAST;
    Eigen::Isometry3d m_matTransformationLEFTLASTtoLEFTNOW;
    Eigen::Isometry3d m_matTransformationMotionWORLDtoIMU;
    Eigen::Isometry3d m_matTransformationIMULAST;
    CAngularVelocityInCameraFrame m_vecVelocityAngularFilteredLAST;
    const double m_dMaximumDeltaTimestampSeconds;
    double m_dTimestampLASTSeconds;
    CPoint3DInWorldFrame m_vecTranslationLastKeyFrame;
    const double m_dTranslationDeltaForKeyFrameMetersSquaredNorm;
    CPoint3DInWorldFrame m_vecPositionCurrent;
    double m_dTranslationDeltaSquaredNormCurrent;

    //ds feature related
    //const uint32_t m_uKeyPointSize;
    const std::shared_ptr< cv::FeatureDetector > m_pDetector;
    const std::shared_ptr< cv::DescriptorExtractor > m_pExtractor;
    const std::shared_ptr< cv::DescriptorMatcher > m_pMatcher;
    const double m_dMatchingDistanceCutoffTriangulation;
    const double m_dMatchingDistanceCutoffPoseOptimization;
    const double m_dMatchingDistanceCutoffTracking;

    const uint8_t m_uMaximumFailedSubsequentTrackingsPerLandmark;
    const uint8_t m_uVisibleLandmarksMinimum;
    cv::Mat m_matDisplayLowerReference;
    const double m_dMinimumDepthMeters;
    const double m_dMaximumDepthMeters;

    std::shared_ptr< CTriangulator > m_pTriangulator;
    CMatcherEpipolar m_cMatcherEpipolar;

    //ds tracking (we use the ID counter instead of accessing the vector size every time for speed)
    UIDLandmark m_uAvailableLandmarkID;
    std::shared_ptr< std::vector< CLandmark* > > m_vecLandmarks;
    uint64_t m_uNumberofLastVisibleLandmarks;

    //ds g2o data
    std::vector< CKeyFrame > m_vecKeyFrames;

    //ds control
    const EPlaybackMode m_eMode;
    bool m_bIsShutdownRequested;

    //ds info display
    cv::Mat m_matTrajectoryXY;
    const uint32_t m_uOffsetTrajectoryU;
    const uint32_t m_uOffsetTrajectoryV;
    uint64_t m_uTimingToken;
    uint32_t m_uFramesCurrentCycle;
    double m_dPreviousFrameRate;

    //ds debug logging
    std::FILE* m_pFileLandmarkCreation;
    std::FILE* m_pFileLandmarkFinal;
    std::FILE* m_pFileTrajectory;

//ds accessors
public:

    void receivevDataVI( const std::shared_ptr< txt_io::PinholeImageMessage > p_pImageLEFT,
                         const std::shared_ptr< txt_io::PinholeImageMessage > p_pImageRIGHT,
                         const txt_io::CIMUMessage& p_cIMU );

    void receivevDataVI( const std::shared_ptr< txt_io::PinholeImageMessage > p_pImageLEFT,
                         const std::shared_ptr< txt_io::PinholeImageMessage > p_pImageRIGHT,
                         const txt_io::CIMUMessage& p_cIMU,
                         const Eigen::Isometry3d& p_matTransformationIMU );

    const uint64_t getFrameCount( ) const { return m_uFrameCount; }
    const bool isShutdownRequested( ) const { return m_bIsShutdownRequested; }

    //ds postprocessing
    void saveUVDepthOrDisparity( const std::string& p_strOutfile ) const
    {
        CBridgeG2O::saveUVDepthOrDisparity( p_strOutfile, *m_pCameraSTEREO, *m_vecLandmarks, m_vecKeyFrames );
    }
    void saveXYZ( const std::string& p_strOutfile ) const
    {
        CBridgeG2O::saveXYZ( p_strOutfile, *m_pCameraSTEREO, *m_vecLandmarks, m_vecKeyFrames );
    }
    void saveUVDepth( const std::string& p_strOutfile ) const
    {
        CBridgeG2O::saveUVDepth( p_strOutfile, *m_pCameraSTEREO, *m_vecLandmarks, m_vecKeyFrames );
    }
    void saveUVDisparity( const std::string& p_strOutfile ) const
    {
        CBridgeG2O::saveUVDisparity( p_strOutfile, *m_pCameraSTEREO, *m_vecLandmarks, m_vecKeyFrames );
    }
    void saveCOMBO( const std::string& p_strOutfile ) const
    {
        CBridgeG2O::saveCOMBO( p_strOutfile, *m_pCameraSTEREO, *m_vecLandmarks, m_vecKeyFrames );
    }

//ds helpers
private:

    void _trackLandmarks( const cv::Mat& p_matImageLEFT,
                          const cv::Mat& p_matImageRIGHT,
                          const Eigen::Isometry3d& p_matTransformationEstimateWORLDtoLEFT,
                          const Eigen::Isometry3d& p_matTransformationEstimateDampedWORLDtoLEFT,
                          const CAngularVelocityInIMUFrame& p_vecAngularVelocity,
                          const CLinearAccelerationInIMUFrame& p_vecLinearAcceleration );

    const std::shared_ptr< std::vector< CLandmark* > > _getNewLandmarks( const uint64_t& p_uFrame,
                                                      cv::Mat& p_matDisplay,
                                                      const cv::Mat& p_matImageLEFT,
                                                      const cv::Mat& p_matImageRIGHT,
                                                      const Eigen::Isometry3d& p_matTransformationLEFTtoWORLD,
                                                      const cv::Mat& p_matMask );

    //ds control
    void _shutDown( );
    void _updateFrameRateForInfoBox( const uint32_t& p_uFrameProbeRange = 10 );
    void _drawInfoBox( cv::Mat& p_matDisplay ) const;
};

#endif //#define CTRACKERSTEREOMOTIONMODEL_H
