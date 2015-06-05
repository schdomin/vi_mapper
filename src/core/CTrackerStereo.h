#ifndef CTRACKERSTEREO_H
#define CTRACKERSTEREO_H

#include <memory>

#include "txt_io/imu_message.h"
#include "txt_io/pinhole_image_message.h"
#include "txt_io/pose_message.h"

#include "utility/CStereoCamera.h"
#include "utility/CBridgeG2O.h"
#include "CTriangulator.h"
#include "CDetectorMonoTilewise.h"
#include "CMatcherEpipolar.h"

class CTrackerStereo
{

//ds ctor/dtor
public:

    CTrackerStereo( const uint32_t& p_uFrequencyPlaybackHz,
                    const EPlaybackMode& p_eMode,
                    const uint32_t& p_uWaitKeyTimeout = 1 );
    ~CTrackerStereo( );

//ds members
private:

    //ds vision setup
    const uint32_t m_uWaitKeyTimeout;
    const std::shared_ptr< CPinholeCamera > m_pCameraLEFT;
    const std::shared_ptr< CPinholeCamera > m_pCameraRIGHT;
    const std::shared_ptr< CStereoCamera > m_pStereoCamera;

    //ds reference information
    uint64_t m_uFrameCount;
    CPoint3DInWorldFrame m_vecTranslationLast;
    double m_dTranslationDeltaForMAPMeters;

    //ds feature related
    const uint32_t m_uKeyPointSize;
    std::shared_ptr< cv::FeatureDetector > m_pDetector;
    std::shared_ptr< cv::DescriptorExtractor > m_pExtractor;
    std::shared_ptr< cv::DescriptorMatcher > m_pMatcher;
    const float m_fMatchingDistanceCutoffTriangulation;
    const float m_fMatchingDistanceCutoffTracking;

    const uint8_t m_uMaximumFailedSubsequentTrackingsPerLandmark;
    const uint8_t m_uVisibleLandmarksMinimum;
    cv::Mat m_matTrajectoryXY;
    cv::Mat m_matTrajectoryZ;
    cv::Mat m_matDisplayLowerReference;
    const double m_dMinimumDepthMeters;
    const double m_dMaximumDepthMeters;

    //const CDetectorMonoTilewise m_cDetector;
    std::shared_ptr< CTriangulator > m_pTriangulator;
    CMatcherEpipolar m_cMatcherEpipolar;

    //ds tracking
    UIDLandmark m_uAvailableLandmarkID;
    std::shared_ptr< std::vector< CLandmark* > > m_vecLandmarks;
    uint64_t m_uNumberofLastVisibleLandmarks;

    //ds g2o data
    std::vector< std::pair< Eigen::Isometry3d, std::shared_ptr< std::vector< CLandmarkMeasurement* > > > > m_vecLogMeasurementPoints;

    //ds control
    const EPlaybackMode m_eMode;
    bool m_bIsShutdownRequested;
    double m_dFrequencyPlaybackHz;
    uint32_t m_uFrequencyPlaybackDeltaHz;
    int32_t m_iPlaybackSpeedupCounter;
    cv::RNG_MT19937 m_cRandomGenerator;

    //ds info display
    uint64_t m_uTimingToken;
    uint32_t m_uFramesCurrentCycle;
    double m_dPreviousFrameRate;

//ds accessors
public:

    void receivevDataVIWithPose( std::shared_ptr< txt_io::PinholeImageMessage >& p_pImageLeft,
                                 std::shared_ptr< txt_io::PinholeImageMessage >& p_pImageRight,
                                 const txt_io::CIMUMessage& p_cIMU,
                                 const std::shared_ptr< txt_io::CPoseMessage >& p_cPose );

    const uint64_t getFrameCount( ) const { return m_uFrameCount; }
    const bool isShutdownRequested( ) const { return m_bIsShutdownRequested; }
    const uint32_t getPlaybackFrequencyHz( ) const { return std::lround( m_dFrequencyPlaybackHz ); }

    //ds postprocessing
    void savesolveAndOptimizeG2O( const std::string& p_strOutfile ) const
    {
        CBridgeG2O::savesolveAndOptimizeG2O( p_strOutfile, *m_pStereoCamera, *m_vecLandmarks, m_vecLogMeasurementPoints );
    }

//ds helpers
private:

    void _trackLandmarks( const cv::Mat& p_matImageLEFT,
                          const cv::Mat& p_matImageRIGHT,
                          const Eigen::Isometry3d& p_matTransformationLEFTtoWORLD,
                          const Eigen::Vector3d p_vecAngularVelocity,
                          const Eigen::Vector3d p_vecLinearAcceleration );

    const std::shared_ptr< std::vector< CLandmark* > > _getNewLandmarksTriangulated( cv::Mat& p_matDisplay,
                                                      const cv::Mat& p_matImageLEFT,
                                                      const cv::Mat& p_matImageRIGHT,
                                                      const Eigen::Isometry3d& p_matTransformationLEFTtoWORLD,
                                                      const std::shared_ptr< std::vector< CLandmarkMeasurement* > > p_vecVisibleLandmarks );

    //ds control
    void _shutDown( );
    void _speedUp( );
    void _slowDown( );
    void _updateFrameRateForInfoBox( const uint32_t& p_uFrameProbeRange = 10 );
    void _drawInfoBox( cv::Mat& p_matDisplay ) const;
};

#endif //#define CTRACKERSTEREO_H
