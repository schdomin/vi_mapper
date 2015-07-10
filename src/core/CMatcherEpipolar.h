#ifndef CMATCHEREPIPOLAR_H
#define CMATCHEREPIPOLAR_H

#include <memory>

#include "CTriangulator.h"
#include "types/CLandmark.h"
#include "optimization/CPositSolver.h"
#include "optimization/CPositSolverProjection.h"
#include "optimization/CPositSolverStereo.h"

class CMatcherEpipolar
{

//ds private structs
private:

    struct CKeyFrame
    {
        const UIDMeasurementPoint uID;
        const Eigen::Isometry3d matTransformationLEFTtoWORLD;
        const std::shared_ptr< std::vector< CLandmark* > > vecLandmarks;

        CKeyFrame( const UIDMeasurementPoint& p_uID,
                           const Eigen::Isometry3d& p_matTransformationLEFTtoWORLD,
                           const std::shared_ptr< std::vector< CLandmark* > > p_vecLandmarks ): uID( p_uID ),
                                                                                                matTransformationLEFTtoWORLD( p_matTransformationLEFTtoWORLD ),
                                                                                                vecLandmarks( p_vecLandmarks )
        {
            //ds nothing to do
        }
        ~CKeyFrame( )
        {
            //ds nothing to do
        }
    };

//ds ctor/dtor
public:

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    CMatcherEpipolar( const std::shared_ptr< CTriangulator > p_pTriangulator,
                      const std::shared_ptr< cv::FeatureDetector > p_pDetectorSingle,
                      const double& p_dMinimumDepthMeters,
                      const double& p_dMaximumDepthMeters,
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
    const std::shared_ptr< CStereoCamera > m_pCameraSTEREO;

    //ds matching
    const std::shared_ptr< cv::FeatureDetector > m_pDetector;
    const std::shared_ptr< cv::DescriptorExtractor > m_pExtractor;
    const std::shared_ptr< cv::DescriptorMatcher > m_pMatcher;
    const double m_dMinimumDepthMeters;
    const double m_dMaximumDepthMeters;
    const double m_dMatchingDistanceCutoff;
    const double m_dMatchingDistanceCutoffOriginal;

    //ds measurement point storage (we use the ID counter instead of accessing the vector size every time for speed)
    UIDMeasurementPoint m_uAvailableKeyFrameID;
    std::vector< CKeyFrame > m_vecKeyFramesActive;

    //ds internal
    const int32_t m_iSearchUMin;
    const int32_t m_iSearchUMax;
    const int32_t m_iSearchVMin;
    const int32_t m_iSearchVMax;
    const cv::Rect m_cSearchROI;
    const cv::Rect m_cSearchROIPoseOptimization;
    const uint8_t m_uMaximumFailedSubsequentTrackingsPerLandmark;
    const uint8_t m_uRecursionLimitEpipolarLines;
    const uint8_t m_uMinimumPointsForPoseOptimization;

    //ds debug logging
    //gtools::CPositSolver m_cSolverPose;
    gtools::CPositSolverProjection m_cSolverPoseProjection;
    gtools::CPositSolverStereo m_cSolverPoseSTEREO;
    std::FILE* m_pFileOdometryError;
    std::FILE* m_pFileEpipolarDetection;

//ds api
public:

    void addKeyFrame( const Eigen::Isometry3d& p_matTransformationLEFTtoWORLD, const std::shared_ptr< std::vector< CLandmark* > > p_vecLandmarks )
    {
        m_vecKeyFramesActive.push_back( CKeyFrame( m_uAvailableKeyFrameID, p_matTransformationLEFTtoWORLD, p_vecLandmarks ) );

        ++m_uAvailableKeyFrameID;
    }

    const Eigen::Isometry3d getPoseOptimizedLEFT( const uint64_t p_uFrame,
                                              cv::Mat& p_matDisplayLEFT,
                                              const cv::Mat& p_matImageLEFT,
                                              const Eigen::Isometry3d& p_matTransformationEstimateWORLDtoLEFT );

    const Eigen::Isometry3d getPoseOptimizedSTEREO( const uint64_t p_uFrame,
                                                    cv::Mat& p_matDisplayLEFT,
                                                    cv::Mat& p_matDisplayRIGHT,
                                                    const cv::Mat& p_matImageLEFT,
                                                    const cv::Mat& p_matImageRIGHT,
                                                    const Eigen::Isometry3d& p_matTransformationEstimateWORLDtoLEFT );

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

    const std::shared_ptr< std::vector< const CMeasurementLandmark* > > getVisibleLandmarksEssential( const uint64_t p_uFrame,
                                                                                                      cv::Mat& p_matDisplayLEFT,
                                                                                                      cv::Mat& p_matDisplayRIGHT,
                                                                                                      const cv::Mat& p_matImageLEFT,
                                                                                                      const cv::Mat& p_matImageRIGHT,
                                                                                                      const Eigen::Isometry3d& p_matTransformationLEFTToWorldNow,
                                                                                                      const int32_t& p_iHalfLineLengthBase,
                                                                                                      cv::Mat& p_matDisplayTrajectory );

    const std::vector< CKeyFrame >::size_type getNumberOfActiveMeasurementPoints( ) const
    {
        return m_vecKeyFramesActive.size( );
    }

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

    const CMatchTracking* _getMatchSampleRecursiveU( cv::Mat& p_matDisplay,
                                                       const cv::Mat& p_matImage,
                                                       const int32_t& p_iUMinimum,
                                                       const int32_t& p_iDeltaU,
                                                       const Eigen::Vector3d& p_vecCoefficients,
                                                       const CDescriptor& p_matReferenceDescriptor,
                                                       const CDescriptor& p_matOriginalDescriptor,
                                                       const double& p_dKeyPointSize,
                                                       const uint8_t& p_uRecursionDepth ) const;

    const CMatchTracking* _getMatchSampleRecursiveV( cv::Mat& p_matDisplay,
                                                       const cv::Mat& p_matImage,
                                                       const int32_t& p_iVMinimum,
                                                       const int32_t& p_iDeltaV,
                                                       const Eigen::Vector3d& p_vecCoefficients,
                                                       const CDescriptor& p_matReferenceDescriptor,
                                                       const CDescriptor& p_matOriginalDescriptor,
                                                       const double& p_dKeyPointSize,
                                                       const uint8_t& p_uRecursionDepth ) const;

    const CMatchTracking* _getMatchSampleRecursiveU( const cv::Mat& p_matImage,
                                                     const int32_t& p_iUMinimum,
                                                     const int32_t& p_iDeltaU,
                                                     const Eigen::Vector3d& p_vecCoefficients,
                                                     const CDescriptor& p_matReferenceDescriptor,
                                                     const CDescriptor& p_matOriginalDescriptor,
                                                     const double& p_dKeyPointSize,
                                                     const uint8_t& p_uRecursionDepth ) const;

    const CMatchTracking* _getMatchSampleRecursiveV( const cv::Mat& p_matImage,
                                                     const int32_t& p_iVMinimum,
                                                     const int32_t& p_iDeltaV,
                                                     const Eigen::Vector3d& p_vecCoefficients,
                                                     const CDescriptor& p_matReferenceDescriptor,
                                                     const CDescriptor& p_matOriginalDescriptor,
                                                     const double& p_dKeyPointSize,
                                                     const uint8_t& p_uRecursionDepth ) const;

    const CMatchTracking* _getMatch( const cv::Mat& p_matImage,
                                     std::vector< cv::KeyPoint >& p_vecPoolKeyPoints,
                                     const CDescriptor& p_matDescriptorReference,
                                     const CDescriptor& p_matDescriptorOriginal ) const;

    inline const double _getCurveX( const Eigen::Vector3d& p_vecCoefficients, const double& p_dY ) const;
    inline const double _getCurveY( const Eigen::Vector3d& p_vecCoefficients, const double& p_dX ) const;
    inline const int32_t _getCurveU( const Eigen::Vector3d& p_vecCoefficients, const int32_t& p_uV ) const;
    inline const int32_t _getCurveV( const Eigen::Vector3d& p_vecCoefficients, const int32_t& p_uU ) const;
    inline const int32_t _getCurveU( const Eigen::Vector3d& p_vecCoefficients, const double& p_uV ) const;
    inline const int32_t _getCurveV( const Eigen::Vector3d& p_vecCoefficients, const double& p_uU ) const;

};

#endif //#define CMATCHEREPIPOLAR_H
