#ifndef CEPIPOLARDETECTORBRIEF_H_
#define CEPIPOLARDETECTORBRIEF_H_

#include <opencv/cv.h>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <memory>
#include <chrono>
#include <ctime>
#include <fstream>

#include "txt_io/imu_message.h"
#include "txt_io/pinhole_image_message.h"
#include "txt_io/pose_message.h"

#include "configuration/CConfigurationOpenCV.h"
#include "utility/CWrapperOpenCV.h"
#include "utility/CMiniVisionToolbox.h"
#include "utility/CStereoCamera.h"
#include "exceptions/CExceptionNoMatchFound.h"

//ds readability
typedef Eigen::Vector3d CPoint2DNormalized;
typedef Eigen::Vector2d CPoint2DInCameraFrame;
typedef Eigen::Vector3d CPoint3DInCameraFrame;
typedef Eigen::Vector3d CPoint3DInWorldFrame;
typedef cv::Scalar      CColorCode;
typedef cv::Mat         CDescriptor;

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
    cv::Mat m_matReferenceFrameLeft;
    cv::Mat m_matReferenceFrameRight;
    uint64_t m_uFrameCount;
    Eigen::Isometry3d m_matPreviousTransformationLeft;

    //ds tracking points
    std::vector< std::pair< std::vector< std::tuple< uint64_t, cv::KeyPoint, CDescriptor, CPoint2DNormalized, CPoint2DNormalized, uint8_t > >, Eigen::Isometry3d > > m_vecScanPoints;

    //ds feature related
    cv::BriefDescriptorExtractor m_cExtractorBRIEF;
    cv::FlannBasedMatcher m_cFLANNMatcher;
    const uint32_t m_uKeyPointSizeLimit;
    const int32_t m_iSearchUMin;
    const int32_t m_iSearchUMax;
    const int32_t m_iSearchVMin;
    const int32_t m_iSearchVMax;
    const cv::Rect m_cSearchROI;
    const float m_fMatchingDistanceCutoff;
    const uint32_t m_uLimitFeaturesPerScan;
    const uint8_t m_uMaximumNonMatches;
    const uint8_t m_uActiveLandmarksMinimum;
    cv::Mat m_matTrajectoryXY;
    cv::Mat m_matTrajectoryZ;
    cv::Mat m_matDisplayLowerReference;

    //ds recording
    uint64_t m_uAvailableLandmarkID;
    std::vector< std::pair< uint64_t, CPoint3DInWorldFrame > > m_vecLandmarks;
    std::vector< std::pair< Eigen::Isometry3d, std::vector< std::pair< uint64_t, CPoint2DNormalized > > > > m_vecPosesWithLandmarks;

    //ds control
    bool m_bDisplayImages;
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
    void dumpVerticesWithLandmarks( const std::string& p_strOutfile ) const;
    void solveAndOptimizeG2O( const std::string& p_strOutfile ) const;

//ds helpers
private:

    void _localize( const cv::Mat& p_matImageLeft, const cv::Mat& p_matImageRight, const Eigen::Isometry3d& p_matCurrentTransformation );

    //ds epipolar lines
    uint64_t _matchProjectedEpipolarLineEssential( const Eigen::Isometry3d& p_matCurrentTransformation, cv::Mat& p_matDisplay, cv::Mat& p_matImage, const int32_t& p_uLineLength );

    std::vector< std::tuple< uint64_t, cv::KeyPoint, CDescriptor, CPoint2DNormalized, CPoint2DNormalized, uint8_t > > _getLandmarksGFTT( cv::Mat& p_matDisplay, const cv::Mat& p_matImage, const uint32_t& p_uTileNumberBase );

    CPoint2DNormalized _getMatchSampleUBRIEF( cv::Mat& p_matDisplay,
                                              const cv::Mat& p_matImage,
                                              const int32_t& p_iUMinimum,
                                              const int32_t& p_iDeltaU,
                                              const Eigen::Vector3d& p_vecCoefficients,
                                              const cv::Mat& p_matReferenceDescriptor ) const;
    CPoint2DNormalized _getMatchSampleVBRIEF( cv::Mat& p_matDisplay,
                                              const cv::Mat& p_matImage,
                                              const int32_t& p_iVMinimum,
                                              const int32_t& p_iDeltaV,
                                              const Eigen::Vector3d& p_vecCoefficients,
                                              const cv::Mat& p_matReferenceDescriptor ) const;

    CPoint2DNormalized _getMatchBRIEF( cv::Mat& p_matDisplay,
                                       const cv::Mat& p_matImage,
                                       std::vector< cv::KeyPoint >& p_vecPoolKeyPoints,
                                       const cv::Mat& p_matReferenceDescriptor ) const;

    //ds control
    void _shutDown( );
    void _speedUp( );
    void _slowDown( );
    void _updateFrameRateDisplay( const uint32_t& p_uFrameProbeRange );
};

#endif //#define CEPIPOLARDETECTOR_H_
