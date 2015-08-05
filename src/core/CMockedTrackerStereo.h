#ifndef CMOCKEDTRACKERSTEREO_H
#define CMOCKEDTRACKERSTEREO_H

#include <memory>

#include "txt_io/imu_message.h"
#include "txt_io/pinhole_image_message.h"
#include "txt_io/pose_message.h"

#include "vision/CMockedStereoCamera.h"
#include "optimization/CBridgeG2O.h"
#include "CMockedMatcherEpipolar.h"

class CMockedTrackerStereo
{

//ds ctor/dtor
public:

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    CMockedTrackerStereo( const EPlaybackMode& p_eMode,
                    const std::string& p_strLandmarksMocked,
                    const uint32_t& p_uWaitKeyTimeout = 1 );
    ~CMockedTrackerStereo( );

//ds members
private:

    //ds mocking
    const std::shared_ptr< std::vector< CMockedLandmark > > m_vecLandmarksMocked;

    //ds vision setup
    const uint32_t m_uWaitKeyTimeout;
    const std::shared_ptr< CPinholeCamera > m_pCameraLEFT;
    const std::shared_ptr< CPinholeCamera > m_pCameraRIGHT;
    const std::shared_ptr< CMockedStereoCamera > m_pCameraSTEREO;

    //ds reference information
    uint64_t m_uFrameCount;
    CPoint3DInWorldFrame m_vecTranslationKeyFrameLAST;
    double m_dTranslationDeltaForMAPMeters;

    const uint8_t m_uMaximumFailedSubsequentTrackingsPerLandmark;
    const uint8_t m_uVisibleLandmarksMinimum;
    cv::Mat m_matDisplayLowerReference;
    const double m_dMinimumDepthMeters;
    const double m_dMaximumDepthMeters;

    //const CDetectorMonoTilewise m_cDetector;
    CMockedMatcherEpipolar m_cMatcherEpipolar;

    //ds tracking (we use the ID counter instead of accessing the vector size every time for speed)
    UIDLandmark m_uAvailableLandmarkID;
    std::shared_ptr< std::vector< CLandmark* > > m_vecLandmarks;
    uint64_t m_uNumberofLastVisibleLandmarks;

    //ds g2o data
    std::vector< CKeyFrame > m_vecKeyFrames;
    UIDKeyFrame m_uIDProcessedKeyFrameLAST            = 0;
    const UIDKeyFrame m_uIDDeltaKeyFrameForProcessing = 9;
    uint32_t m_uOptimizationsG2O                      = 0;

    //ds control
    const EPlaybackMode m_eMode;
    bool m_bIsShutdownRequested;

    //ds info display
    cv::Mat m_matTrajectoryXY;
    cv::Point2d m_ptPositionXY;
    const uint32_t m_uOffsetTrajectoryU = 180;
    const uint32_t m_uOffsetTrajectoryV = 360;
    double m_dFrameTimeSecondsLAST      = 0.0;
    uint32_t m_uFramesCurrentCycle      = 0;
    double m_dPreviousFrameRate         = 0.0;

//ds accessors
public:

    void receivevDataVIWithPose( const std::shared_ptr< txt_io::PinholeImageMessage > p_pImageLEFT,
                                 const std::shared_ptr< txt_io::PinholeImageMessage > p_pImageRIGHT,
                                 const txt_io::CIMUMessage& p_cIMU,
                                 const std::shared_ptr< txt_io::CPoseMessage > p_cPose );

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
                          const Eigen::Isometry3d& p_matTransformationLEFTtoWORLD,
                          const CLinearAccelerationIMU& p_vecLinearAcceleration );

    const std::shared_ptr< std::vector< CLandmark* > > _getNewLandmarks( const uint64_t& p_uFrame,
                                                                                     cv::Mat& p_matDisplay,
                                                                                     cv::Mat& p_matDisplayTrajectory,
                                                                                     const cv::Point2d& p_ptPositionXY,
                                                                                     const Eigen::Isometry3d& p_matTransformationLEFTtoWORLD );

    //ds control
    void _shutDown( );
    void _updateFrameRateForInfoBox( const uint32_t& p_uFrameProbeRange = 10 );
    void _drawInfoBox( cv::Mat& p_matDisplay ) const;
};

#endif //#define CMOCKEDTRACKERSTEREO_H
