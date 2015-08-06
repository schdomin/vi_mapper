#ifndef CSTEREODETECTOR_H_
#define CSTEREODETECTOR_H_

#include <opencv/cv.h>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <memory>

#include "txt_io/imu_message.h"
#include "txt_io/pinhole_image_message.h"
#include "txt_io/pose_message.h"
#include "configuration/CConfigurationOpenCV.h"
#include "utility/CWrapperOpenCV.h"
#include "utility/CMiniVisionToolbox.h"
#include "utility/CStereoCamera.h"

//ds readability
typedef Eigen::Vector2d CPoint2DInCameraFrame;
typedef Eigen::Vector3d CPoint3DCAMERA;
typedef cv::Scalar      CColorCodeBGR;
typedef cv::Mat         CDescriptor;

class CStereoDetector
{

//ds ctor/dtor
public:

    CStereoDetector( const uint32_t& p_uImageRows,
                     const uint32_t& p_uImageCols,
                     const bool& p_bDisplayImages,
                     const uint32_t p_uFrequencyPlaybackHz );
    ~CStereoDetector( );

//ds members
private:

    //ds vision setup
    const uint32_t m_uImageRows;
    const uint32_t m_uImageCols;
    const cv::Size m_prSize;
    cv::Mat m_arrMapsLEFT[2];
    cv::Mat m_arrMapsRIGHT[2];
    const Eigen::Matrix3d m_matIntrinsicLEFT;
    const Eigen::Matrix3d m_matIntrinsicRIGHT;
    const Eigen::Matrix< double, 3, 4 > m_matProjectionLEFT;
    const Eigen::Matrix< double, 3, 4 > m_matProjectionRIGHT;
    const Eigen::Matrix3d m_matMLEFT;
    const Eigen::Vector3d m_vecTLEFT;

    //ds reference information
    cv::Mat m_matReferenceFrameLeft;
    cv::Mat m_matReferenceFrameRight;
    uint64_t m_uFrameCount;
    Eigen::Isometry3d m_matPreviousTransformationLeft;

    //ds slam points
    std::vector< std::tuple< CPoint2DInCameraFrame, CPoint3DCAMERA, cv::KeyPoint, CDescriptor, Eigen::Isometry3d, CColorCodeBGR, CPoint2DInCameraFrame > > m_vecReferencePoints;

    //ds feature related
    cv::SurfFeatureDetector m_cDetectorSURF;
    cv::SurfDescriptorExtractor m_cExtractorSURF;
    cv::FlannBasedMatcher m_cDescriptorMatcher;
    const double m_dMatchingDistanceCutoff;
    const uint32_t m_uFeaturesCap;
    cv::Mat m_matTrajectoryXY;
    cv::Mat m_matTrajectoryZ;
    const uint32_t m_uDescriptorRadius;
    const uint32_t m_uDescriptorCenterPixelOffset;
    const cv::Rect m_rectROI;
    cv::Mat m_matDisplayLowerReference;

    //ds control
    bool m_bDisplayImages;
    bool m_bIsShutdownRequested;
    double m_dFrequencyPlaybackHz;
    uint32_t m_uFrequencyPlaybackDeltaHz;
    int32_t m_iPlaybackSpeedupCounter;
    cv::RNG_MT19937 m_cRandomGenerator;

    //ds cameras
    const CPinholeCamera m_cCameraLEFT;
    const CPinholeCamera m_cCameraRIGHT;
    const CStereoCamera m_cStereoCamera;

    //ds user input
    static cv::Point2i m_ptMouseClick;
    static bool m_bRightClicked;

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

    void _localizeManual( const cv::Mat& p_matImageLeft, const cv::Mat& p_matImageRight, const Eigen::Isometry3d& p_matCurrentTransformation );
    void _localizeAuto( const cv::Mat& p_matImageLeft, const cv::Mat& p_matImageRight, const Eigen::Isometry3d& p_matCurrentTransformation );

    //ds epipolar lines
    void _drawProjectedEpipolarLineEssential( const Eigen::Isometry3d& p_matCurrentTransformation, cv::Mat& p_matDisplay, cv::Mat& p_matImage, const double& p_dLineLength );

    //ds triangulation methods
    void _triangulatePointSURF( const cv::Mat& p_matImageLeft, const cv::Mat& p_matImageRight, cv::Mat& p_matDisplayUpper, cv::Mat& p_matDisplayUpperTemporary, const Eigen::Isometry3d p_matCurrentTransformation );

    //ds control
    void _shutDown( );
    void _speedUp( );
    void _slowDown( );
    static void _catchMouseClick( int p_iEventType, int p_iX, int p_iY, int p_iFlags, void* p_hUserdata );
};

#endif //#define CSTEREODETECTOR_H_
