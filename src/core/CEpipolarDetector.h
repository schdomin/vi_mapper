#ifndef CEPIPOLARDETECTOR_H_
#define CEPIPOLARDETECTOR_H_

#include <opencv/cv.h>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <memory>
#include <chrono>
#include <ctime>

#include "txt_io/imu_message.h"
#include "txt_io/pinhole_image_message.h"
#include "txt_io/pose_message.h"
#include "configuration/CConfigurationOpenCV.h"
#include "utility/CWrapperOpenCV.h"
#include "utility/CMiniVisionToolbox.h"
#include "utility/CStereoCamera.h"

//ds readability
typedef Eigen::Vector3d CPoint2DNormalized;
typedef Eigen::Vector2d CPoint2DInCameraFrame;
typedef Eigen::Vector3d CPoint3DInCameraFrame;
typedef cv::Scalar      CColorCode;
typedef cv::Mat         CDescriptor;

class CEpipolarDetector
{

//ds ctor/dtor
public:

    CEpipolarDetector( const uint32_t& p_uImageRows,
                     const uint32_t& p_uImageCols,
                     const bool& p_bDisplayImages,
                     const uint32_t p_uFrequencyPlaybackHz );
    ~CEpipolarDetector( );

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

    //ds slam points
    std::vector< std::pair< std::vector< std::tuple< cv::KeyPoint, CDescriptor, CPoint2DNormalized, CPoint2DNormalized > >, Eigen::Isometry3d > > m_vecScanPoints;

    //ds feature related
    cv::BriefDescriptorExtractor m_cExtractorBRIEF;
    cv::SurfFeatureDetector m_cDetectorSURF;
    cv::SurfDescriptorExtractor m_cExtractorSURF;
    cv::FlannBasedMatcher m_cFLANNMatcher;
    const uint32_t m_uKeyPointSizeLimit;
    const double m_dMatchingDistanceCutoff;
    const uint32_t m_uFeaturesCap;
    cv::Mat m_matTrajectoryXY;
    cv::Mat m_matTrajectoryZ;
    cv::Mat m_matDisplayLowerReference;

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

    const uint64_t getFrameCount( ){ return m_uFrameCount; }
    const bool isShutdownRequested( ){ return m_bIsShutdownRequested; }
    const uint32_t getPlaybackFrequencyHz( ){ return std::lround( m_dFrequencyPlaybackHz ); }

//ds helpers
private:

    void _localizeAutoBRIEF( const cv::Mat& p_matImageLeft, const cv::Mat& p_matImageRight, const Eigen::Isometry3d& p_matCurrentTransformation );
    void _localizeAutoSURF( const cv::Mat& p_matImageLeft, const cv::Mat& p_matImageRight, const Eigen::Isometry3d& p_matCurrentTransformation );

    //ds epipolar lines
    void _drawProjectedEpipolarLineEssentialBRIEF( const Eigen::Isometry3d& p_matCurrentTransformation, cv::Mat& p_matDisplay, cv::Mat& p_matImage, const int32_t& p_uLineLength );
    void _drawProjectedEpipolarLineEssentialSURF( const Eigen::Isometry3d& p_matCurrentTransformation, cv::Mat& p_matDisplay, cv::Mat& p_matImage, const int32_t& p_uLineLength );

    //ds triangulation methods
    void _triangulatePointSURF( const cv::Mat& p_matImageLeft, const cv::Mat& p_matImageRight, cv::Mat& p_matDisplayUpper, cv::Mat& p_matDisplayUpperTemporary, const Eigen::Isometry3d p_matCurrentTransformation );

    //ds control
    void _shutDown( );
    void _speedUp( );
    void _slowDown( );
    void _updateFrameRateDisplay( const uint32_t& p_uFrameProbeRange );
};

#endif //#define CEPIPOLARDETECTOR_H_
