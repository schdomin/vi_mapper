#ifndef CDETECTORTESTSTEREODEPTH_H
#define CDETECTORTESTSTEREODEPTH_H

#include <configuration/Types.h>
#include <core/CDetectorMonoTilewise.h>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/features2d.hpp>

#include "txt_io/imu_message.h"
#include "txt_io/pinhole_image_message.h"
#include "txt_io/pose_message.h"
#include "configuration/CConfigurationOpenCV.h"
#include "utility/CWrapperOpenCV.h"
#include "utility/CMiniVisionToolbox.h"
#include "utility/CStereoCamera.h"
#include "CTriangulator.h"

class CDetectorTestStereoDepth
{

//ds ctor/dtor
public:

    CDetectorTestStereoDepth( const uint32_t& p_uImageRows,
                     const uint32_t& p_uImageCols,
                     const bool& p_bDisplayImages,
                     const uint32_t p_uFrequencyPlaybackHz );
    ~CDetectorTestStereoDepth( );

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

    //ds feature related
    std::shared_ptr< cv::BriefDescriptorExtractor > m_pExtractorBRIEF;
    cv::SurfFeatureDetector m_cDetectorSURF;
    cv::SurfDescriptorExtractor m_cExtractorSURF;
    std::shared_ptr< cv::FlannBasedMatcher > m_pMatcherBRIEF;
    cv::FlannBasedMatcher m_cMatcherSURF;
    const float m_fMatchingDistanceCutoffBRIEF;
    const float m_fMatchingDistanceCutoffSURF;
    const uint32_t m_uFeaturesCap;
    const uint32_t m_uKeyPointSizeLimit;
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
    const std::shared_ptr< CPinholeCamera > m_pCameraLEFT;
    const std::shared_ptr< CPinholeCamera > m_pCameraRIGHT;
    const std::shared_ptr< CStereoCamera > m_pStereoCamera;

    //ds detection
    const CDetectorMonoTilewise m_cDetectorMonoGFTT;

    //ds triangulation
    const CTriangulator m_cTriangulator;

    //ds user input
    static cv::Point2i m_ptMouseClick;
    static bool m_bRightClicked;

//ds accessors
public:

    void receivevDataVIWithPose( const std::shared_ptr< txt_io::PinholeImageMessage >& p_cImageLeft,
                                 const std::shared_ptr< txt_io::PinholeImageMessage >& p_cImageRight,
                                 const txt_io::CIMUMessage& p_cIMU,
                                 const std::shared_ptr< txt_io::CPoseMessage >& p_cPose );

    const uint64_t getFrameCount( ){ return m_uFrameCount; }
    const bool isShutdownRequested( ){ return m_bIsShutdownRequested; }
    const uint32_t getPlaybackFrequencyHz( ){ return std::lround( m_dFrequencyPlaybackHz ); }

//ds helpers
private:

    void _localizeManual( const cv::Mat& p_matImageLeft, const cv::Mat& p_matImageRight, const Eigen::Isometry3d& p_matCurrentTransformation );

    //ds control
    void _shutDown( );
    void _speedUp( );
    void _slowDown( );
    static void _catchMouseClick( int p_iEventType, int p_iX, int p_iY, int p_iFlags, void* p_hUserdata );
};

#endif //#define CDETECTORTESTSTEREODEPTH_H
