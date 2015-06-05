#ifndef CEPIPOLARDETECTORBRIEF_H
#define CEPIPOLARDETECTORBRIEF_H

#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <chrono>
#include <ctime>
#include <memory>

#include "txt_io/imu_message.h"
#include "txt_io/pinhole_image_message.h"
#include "txt_io/pose_message.h"

#include "utility/CStereoCamera.h"

class CEpipolarDetectorBRIEF
{

//ds ctor/dtor
public:

    CEpipolarDetectorBRIEF( const uint32_t& p_uImageRows,
                            const uint32_t& p_uImageCols,
                            const bool& p_bDisplayImages,
                            const uint32_t p_uFrequencyPlaybackHz );
    ~CEpipolarDetectorBRIEF( );

//ds members
private:

    //ds vision setup
    const uint32_t m_uImageRows;
    const uint32_t m_uImageCols;

    //ds reference information
    uint64_t m_uFrameCount;
    Eigen::Isometry3d m_matPreviousTransformationLeft;
    CPoint3DInWorldFrame m_vecTranslationLast;
    double m_dTranslationDeltaForMAPMeters;

    //ds feature related
    cv::BriefDescriptorExtractor m_cExtractorBRIEF;
    cv::FlannBasedMatcher m_cMatcherBRIEF;
    const uint32_t m_uKeyPointSizeLimit;
    const int32_t m_iSearchUMin;
    const int32_t m_iSearchUMax;
    const int32_t m_iSearchVMin;
    const int32_t m_iSearchVMax;
    const cv::Rect m_cSearchROI;
    const uint32_t m_uNumberOfTilesBase;
    const float m_fMatchingDistanceCutoffTracking;
    const float m_fMatchingDistanceCutoffTriangulation;
    const uint32_t m_uLimitLandmarksPerScan;
    const uint8_t m_uMaximumNonMatches;
    const uint8_t m_uVisibleLandmarksMinimum;
    cv::Mat m_matTrajectoryXY;
    cv::Mat m_matTrajectoryZ;
    cv::Mat m_matDisplayLowerReference;
    const double m_dMaximumDepthMeters;

    //ds tracking
    UIDLandmark m_uAvailableLandmarkID;
    std::vector< CLandmarkInWorldFrame > m_vecLandmarks;
    std::vector< std::pair< Eigen::Isometry3d, std::vector< CLandmark > > > m_vecActiveMeasurementPoints;
    std::vector< std::pair< Eigen::Isometry3d, std::vector< CLandmarkMeasurement > > > m_vecLogMeasurementPoints;

    //ds control
    bool m_bIsShutdownRequested;
    double m_dFrequencyPlaybackHz;
    uint32_t m_uFrequencyPlaybackDeltaHz;
    int32_t m_iPlaybackSpeedupCounter;
    cv::RNG_MT19937 m_cRandomGenerator;

    //ds fps display
    std::chrono::time_point< std::chrono::system_clock > m_tmStart;
    uint32_t m_uFramesCurrentCycle;
    double m_dPreviousFrameRate;

    //ds cameras
    const CPinholeCamera m_cCameraLEFT;
    const CPinholeCamera m_cCameraRIGHT;
    const CStereoCamera m_cStereoCamera;

//ds accessors
public:

    void receivevDataVIWithPose( std::shared_ptr< txt_io::PinholeImageMessage >& p_cImageLeft,
                                 std::shared_ptr< txt_io::PinholeImageMessage >& p_cImageRight,
                                 const txt_io::CIMUMessage& p_cIMU,
                                 const std::shared_ptr< txt_io::CPoseMessage >& p_cPose );

    const uint64_t getFrameCount( ) const { return m_uFrameCount; }
    const bool isShutdownRequested( ) const { return m_bIsShutdownRequested; }
    const uint32_t getPlaybackFrequencyHz( ) const { return std::lround( m_dFrequencyPlaybackHz ); }

    //ds postprocessing
    void solveAndOptimizeG2O( const std::string& p_strOutfile ) const;

//ds helpers
private:

    void _trackLandmarksAuto( cv::Mat& p_matImageLEFT, const cv::Mat& p_matImageRIGHT, const Eigen::Isometry3d& p_matCurrentTransformation );
    void _trackLandmarksManual( cv::Mat& p_matImageLEFT, const cv::Mat& p_matImageRIGHT, const Eigen::Isometry3d& p_matCurrentTransformation );

    //ds epipolar lines
    const std::vector< CLandmarkMeasurement > _getVisibleLandmarksOnEpipolarLineEssential( const Eigen::Isometry3d& p_matCurrentTransformation, cv::Mat& p_matDisplay, cv::Mat& p_matImage, const int32_t& p_uLineLength );

    const std::vector< CLandmark > _getLandmarksGFTT( cv::Mat& p_matDisplay, const cv::Mat& p_matImageLEFT, const cv::Mat& p_matImageRIGHT, const uint32_t& p_uTileNumberBase, const Eigen::Isometry3d& p_matTransformation );

    const CPoint3DInCameraFrame _getPointTriangulated( const cv::Mat& p_matImageRight,
                                                       const cv::KeyPoint& p_cKeyPoint,
                                                       const CDescriptor& p_matReferenceDescriptor,
                                                       const cv::DescriptorExtractor& p_cExtractor,
                                                       const cv::DescriptorMatcher& p_cMatcher,
                                                       const double& p_dMatchingDistanceCutoff ) const;

    CPoint2DInCameraFrameHomogenized _getMatchSampleUBRIEF( cv::Mat& p_matDisplay,
                                              const cv::Mat& p_matImage,
                                              const int32_t& p_iUMinimum,
                                              const int32_t& p_iDeltaU,
                                              const Eigen::Vector3d& p_vecCoefficients,
                                              const cv::Mat& p_matReferenceDescriptor,
                                              const cv::DescriptorExtractor& p_cExtractor,
                                              const cv::DescriptorMatcher& p_cMatcher,
                                              const float& p_fKeyPointSize ) const;

    CPoint2DInCameraFrameHomogenized _getMatchSampleVBRIEF( cv::Mat& p_matDisplay,
                                              const cv::Mat& p_matImage,
                                              const int32_t& p_iVMinimum,
                                              const int32_t& p_iDeltaV,
                                              const Eigen::Vector3d& p_vecCoefficients,
                                              const cv::Mat& p_matReferenceDescriptor,
                                              const cv::DescriptorExtractor& p_cExtractor,
                                              const cv::DescriptorMatcher& p_cMatcher,
                                              const float& p_fKeyPointSize ) const;

    CPoint2DInCameraFrameHomogenized _getMatchBRIEF( cv::Mat& p_matDisplay,
                                       const cv::Mat& p_matImage,
                                       std::vector< cv::KeyPoint >& p_vecPoolKeyPoints,
                                       const cv::Mat& p_matReferenceDescriptor,
                                       const cv::DescriptorExtractor& p_cExtractor,
                                       const cv::DescriptorMatcher& p_cMatcher ) const;

    //ds control
    void _shutDown( );
    void _speedUp( );
    void _slowDown( );
    void _updateFrameRateDisplay( const uint32_t& p_uFrameProbeRange );
};

#endif //#define CEPIPOLARDETECTOR_H
