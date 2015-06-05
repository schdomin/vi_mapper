#ifndef CMATCHEREPIPOLAR_H
#define CMATCHEREPIPOLAR_H

#include <memory>

#include "CTriangulator.h"
#include "utility/CPinholeCamera.h"
#include "types/CLandmark.h"

class CMatcherEpipolar
{

//ds ctor/dtor
public:

    CMatcherEpipolar( const std::shared_ptr< CTriangulator > p_pTriangulator,
                      const float& p_fMatchingDistanceCutoff,
                      const uint8_t& p_uMaximumFailedSubsequentTrackingsPerLandmark );
    ~CMatcherEpipolar( );

//ds members
private:

    //ds triangulation
    std::shared_ptr< CTriangulator > m_pTriangulator;

    //ds cameras (not necessarily used in stereo here)
    const std::shared_ptr< CPinholeCamera > m_pCameraLEFT;
    const std::shared_ptr< CPinholeCamera > m_pCameraRIGHT;

    //ds matching
    const std::shared_ptr< cv::DescriptorExtractor > m_pExtractor;
    const std::shared_ptr< cv::DescriptorMatcher > m_pMatcher;
    const float m_fMatchingDistanceCutoff;
    const float m_fMatchingDistanceCutoffOriginal;
    const uint32_t m_uKeyPointSize;

    std::vector< std::pair< Eigen::Isometry3d, std::vector< CLandmark* > > > m_vecMeasurementPoints;

    //ds internal
    const int32_t m_iSearchUMin;
    const int32_t m_iSearchUMax;
    const int32_t m_iSearchVMin;
    const int32_t m_iSearchVMax;
    const cv::Rect m_cSearchROI;
    const uint8_t m_uMaximumFailedSubsequentTrackingsPerLandmark;
    const uint8_t m_uRecursionLimit;

//ds api
public:

    void addMeasurementPoint( const Eigen::Isometry3d& p_matTransformationLEFTtoWORLD, const std::shared_ptr< std::vector< CLandmark* > > p_vecLandmarks )
    {
        m_vecMeasurementPoints.push_back( std::pair< Eigen::Isometry3d, std::vector< CLandmark* > >( p_matTransformationLEFTtoWORLD, *p_vecLandmarks ) );
    }

    /*const std::shared_ptr< std::vector< CLandmark* > > getVisibleLandmarksEssential( cv::Mat& p_matDisplay,
                                                                                                 const Eigen::Isometry3d& p_matCurrentTransformation,
                                                                                                 cv::Mat& p_matImage,
                                                                                                 const int32_t& p_iHalfLineLengthBase,
                                                                                                 std::shared_ptr< std::vector< std::pair< Eigen::Isometry3d, std::vector< CLandmark* > > > >& p_vecDetectionPoints ) const;

    const std::shared_ptr< std::vector< CLandmark* > > getVisibleLandmarksEssential( cv::Mat& p_matDisplay,
                                                                                     const Eigen::Isometry3d& p_matCurrentTransformation,
                                                                                     cv::Mat& p_matImage,
                                                                                     const int32_t& p_iHalfLineLengthBase,
                                                                                     const Eigen::Isometry3d p_matTransformationOnDetection,
                                                                                     const std::vector< CLandmark* >& p_vecLandmarks ) const;

    const std::shared_ptr< std::vector< CLandmark* > > getVisibleLandmarksEssential( cv::Mat& p_matDisplay,
                                                                                     const cv::Mat& p_matImage,
                                                                                     const std::shared_ptr< std::vector< CLandmark* > > p_vecLandmarks,
                                                                                     const Eigen::Matrix3d& p_matEssential,
                                                                                     const int32_t& p_iHalfLineLengthBase ) const;*/

    const std::shared_ptr< std::vector< CLandmarkMeasurement* > > getVisibleLandmarksEssential( cv::Mat& p_matDisplayLEFT,
                                                                                     cv::Mat& p_matDisplayRIGHT,
                                                                                     const cv::Mat& p_matImageLEFT,
                                                                                     const cv::Mat& p_matImageRIGHT,
                                                                                     const Eigen::Isometry3d& p_matTransformationLEFTToWorldNow,
                                                                                     const int32_t& p_iHalfLineLengthBase );

private:

    /*const std::pair< cv::Point2f, CDescriptor >  _getMatchSampleU( cv::Mat& p_matDisplay,
                                                       const cv::Mat& p_matImage,
                                                       const int32_t& p_iUMinimum,
                                                       const int32_t& p_iDeltaU,
                                                       const Eigen::Vector3d& p_vecCoefficients,
                                                       const cv::Mat& p_matReferenceDescriptor,
                                                       const float& p_fKeyPointSize ) const;

    const std::pair< cv::Point2f, CDescriptor >  _getMatchSampleV( cv::Mat& p_matDisplay,
                                                       const cv::Mat& p_matImage,
                                                       const int32_t& p_iVMinimum,
                                                       const int32_t& p_iDeltaV,
                                                       const Eigen::Vector3d& p_vecCoefficients,
                                                       const cv::Mat& p_matReferenceDescriptor,
                                                       const float& p_fKeyPointSize ) const;*/

    const CMatchTracking _getMatchSampleRecursiveU( cv::Mat& p_matDisplay,
                                                       const cv::Mat& p_matImage,
                                                       const int32_t& p_iUMinimum,
                                                       const int32_t& p_iDeltaU,
                                                       const Eigen::Vector3d& p_vecCoefficients,
                                                       const cv::Mat& p_matReferenceDescriptor,
                                                       const cv::Mat& p_matOriginalDescriptor,
                                                       const uint8_t& p_uRecursionDepth ) const;

    const CMatchTracking _getMatchSampleRecursiveV( cv::Mat& p_matDisplay,
                                                       const cv::Mat& p_matImage,
                                                       const int32_t& p_iVMinimum,
                                                       const int32_t& p_iDeltaV,
                                                       const Eigen::Vector3d& p_vecCoefficients,
                                                       const cv::Mat& p_matReferenceDescriptor,
                                                       const cv::Mat& p_matOriginalDescriptor,
                                                       const uint8_t& p_uRecursionDepth ) const;

    const CMatchTracking _getMatch( const cv::Mat& p_matImage,
                                                std::vector< cv::KeyPoint >& p_vecPoolKeyPoints,
                                                const cv::Mat& p_matReferenceDescriptor,
                                                const cv::Mat& p_matOriginalDescriptor ) const;

    inline const double _getCurveX( const Eigen::Vector3d& p_vecCoefficients, const double& p_dY ) const;
    inline const double _getCurveY( const Eigen::Vector3d& p_vecCoefficients, const double& p_dX ) const;
    inline const int32_t _getCurveU( const Eigen::Vector3d& p_vecCoefficients, const int32_t& p_uV ) const;
    inline const int32_t _getCurveV( const Eigen::Vector3d& p_vecCoefficients, const int32_t& p_uU ) const;
    inline const int32_t _getCurveU( const Eigen::Vector3d& p_vecCoefficients, const double& p_uV ) const;
    inline const int32_t _getCurveV( const Eigen::Vector3d& p_vecCoefficients, const double& p_uU ) const;

};

#endif //#define CMATCHEREPIPOLAR_H
