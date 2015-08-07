#ifndef CMATCHEREPIPOLAR_H
#define CMATCHEREPIPOLAR_H

#include "CTriangulator.h"
#include "types/CLandmark.h"
#include "optimization/CPositSolver.h"
#include "optimization/CPositSolverProjection.h"
#include "optimization/CPositSolverStereo.h"

class CMatcherEpipolar
{

//ds eigen memory alignment
public:

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

//ds custom types
private:

    enum EImageSource
    {
        eImageLEFT,
        eImageRIGHT
    };

    struct CDetectionPoint
    {
        const UIDDetectionPoint uID;
        const Eigen::Isometry3d matTransformationLEFTtoWORLD;
        const std::shared_ptr< std::vector< CLandmark* > > vecLandmarks;

        CDetectionPoint( const UIDDetectionPoint& p_uID,
                         const Eigen::Isometry3d& p_matTransformationLEFTtoWORLD,
                         const std::shared_ptr< std::vector< CLandmark* > > p_vecLandmarks ): uID( p_uID ),
                                                                                              matTransformationLEFTtoWORLD( p_matTransformationLEFTtoWORLD ),
                                                                                              vecLandmarks( p_vecLandmarks )
        {
            //ds nothing to do
        }
        ~CDetectionPoint( )
        {
            //ds nothing to do
        }
    };

    struct CMatchPoseOptimizationLEFT
    {
        CLandmark* pLandmark;
        const cv::KeyPoint cKeyPoint;
        const CDescriptor matDescriptor;

        CMatchPoseOptimizationLEFT( CLandmark* p_pLandmark, const cv::KeyPoint& p_cKeyPoint, const CDescriptor& p_matDescriptor ): pLandmark( p_pLandmark ), cKeyPoint( p_cKeyPoint ), matDescriptor( p_matDescriptor )
        {
            //ds nothing to do
        }
        ~CMatchPoseOptimizationLEFT( )
        {
            //ds nothing to do
        }
    };

    struct CMatchPoseOptimizationSTEREO
    {
        CLandmark* pLandmark;
        const CPoint3DCAMERA vecPointXYZLEFT;
        const cv::Point2f ptUVLEFT;
        const cv::Point2f ptUVRIGHT;
        const CDescriptor matDescriptorLEFT;
        const CDescriptor matDescriptorRIGHT;

        CMatchPoseOptimizationSTEREO( CLandmark* p_pLandmark,
                                      const CPoint3DCAMERA& p_vecPointXYZLEFT,
                                      const cv::Point2f& p_ptUVLEFT,
                                      const cv::Point2f& p_ptUVRIGHT,
                                      const CDescriptor& p_matDescriptorLEFT,
                                      const CDescriptor& p_matDescriptorRIGHT ): pLandmark( p_pLandmark ),
                                                                                 vecPointXYZLEFT( p_vecPointXYZLEFT ),
                                                                                 ptUVLEFT( p_ptUVLEFT ),
                                                                                 ptUVRIGHT( p_ptUVRIGHT ),
                                                                                 matDescriptorLEFT( p_matDescriptorLEFT ),
                                                                                 matDescriptorRIGHT( p_matDescriptorRIGHT )
        {
            //ds nothing to do
        }
        ~CMatchPoseOptimizationSTEREO( )
        {
            //ds nothing to do
        }
    };

//ds ctor/dtor
public:

    CMatcherEpipolar( const std::shared_ptr< CTriangulator > p_pTriangulator,
                      const std::shared_ptr< cv::FeatureDetector > p_pDetectorSingle,
                      const double& p_dMinimumDepthMeters,
                      const double& p_dMaximumDepthMeters,
                      const double& p_dMatchingDistanceCutoffPoseOptimization,
                      const double& p_dMatchingDistanceCutoffEssential,
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
    const double m_dMatchingDistanceCutoffPoseOptimization;
    const double m_dMatchingDistanceCutoffTrackingEssential;
    const double m_dMatchingDistanceCutoffOriginal;

    //ds measurement point storage (we use the ID counter instead of accessing the vector size every time for speed)
    UIDDetectionPoint m_uAvailableDetectionPointID;
    std::vector< CDetectionPoint > m_vecDetectionPointsActive;

    //ds internal
    const int32_t m_iSearchUMin;
    const int32_t m_iSearchUMax;
    const int32_t m_iSearchVMin;
    const int32_t m_iSearchVMax;
    const cv::Rect m_cSearchROI;
    const uint8_t m_uMaximumFailedSubsequentTrackingsPerLandmark;
    const uint8_t m_uRecursionLimitEpipolarLines      = 3;
    const uint8_t m_uRecursionStepSize                = 1;
    UIDLandmark m_uNumberOfInvalidLandmarksTotal      = 0;
    UIDLandmark m_uNumberOfDetectionsPoseOptimization = 0;
    UIDLandmark m_uNumberOfDetectionsEpipolar         = 0;

    //ds posit solving
    const uint8_t m_uSearchBlockSizePoseOptimization   = 15;
    const uint8_t m_uMinimumPointsForPoseOptimization  = 50;
    const uint8_t m_uMinimumInliersForPoseOptimization = 10;
    const uint8_t m_uCapIterationsPoseOptimization     = 10;
    const double m_dConvergenceDeltaPoseOptimization   = 1e-5;

    //ds debug logging
    //gtools::CPositSolver m_cSolverPose;
    gtools::CPositSolverProjection m_cSolverPoseProjection;
    gtools::CPositSolverStereo m_cSolverPoseSTEREO;

//ds api
public:

    //ds add current detected landmarks to the matcher
    void addDetectionPoint( const Eigen::Isometry3d& p_matTransformationLEFTtoWORLD, const std::shared_ptr< std::vector< CLandmark* > > p_vecLandmarks );

    //ds routine that resets the visibility of all active landmarks
    void resetVisibilityActiveLandmarks( );

    //ds register keyframing on currently visible landmarks
    void setKeyFrameToVisibleLandmarks( );

    //ds returns a handle to all currently visible landmarks
    const std::shared_ptr< const std::vector< CLandmark* > > getVisibleOptimizedLandmarks( ) const;

    const Eigen::Isometry3d getPoseOptimizedLEFT( const uint64_t p_uFrame,
                                              cv::Mat& p_matDisplayLEFT,
                                              const cv::Mat& p_matImageLEFT,
                                              const cv::Mat& p_matImageRIGHT,
                                              const Eigen::Isometry3d& p_matTransformationEstimateWORLDtoLEFT,
                                              const Eigen::Vector3d& p_vecCameraOrientation );

    const Eigen::Isometry3d getPoseOptimizedSTEREO( const uint64_t p_uFrame,
                                                    cv::Mat& p_matDisplayLEFT,
                                                    cv::Mat& p_matDisplayRIGHT,
                                                    const cv::Mat& p_matImageLEFT,
                                                    const cv::Mat& p_matImageRIGHT,
                                                    const Eigen::Isometry3d& p_matTransformationEstimateWORLDtoLEFT,
                                                    const Eigen::Vector3d& p_vecCameraOrientation,
                                                    const double& p_dMotionScaling );

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

    const std::shared_ptr< std::vector< const CMeasurementLandmark* > > getVisibleLandmarksEssentialOptimized( const uint64_t p_uFrame,
                                                                                                      cv::Mat& p_matDisplayLEFT,
                                                                                                      cv::Mat& p_matDisplayRIGHT,
                                                                                                      const cv::Mat& p_matImageLEFT,
                                                                                                      const cv::Mat& p_matImageRIGHT,
                                                                                                      const Eigen::Isometry3d& p_matTransformationLEFTToWorldNow,
                                                                                                      const Eigen::Vector3d& p_vecCameraOrientation,
                                                                                                      const int32_t& p_iHalfLineLengthBase,
                                                                                                      cv::Mat& p_matDisplayTrajectory );

    const std::shared_ptr< std::vector< const CMeasurementLandmark* > > getVisibleLandmarksEssential( cv::Mat& p_matDisplayLEFT,
                                                                                                      cv::Mat& p_matDisplayRIGHT,
                                                                                                      const uint64_t p_uFrame,
                                                                                                      const cv::Mat& p_matImageLEFT,
                                                                                                      const cv::Mat& p_matImageRIGHT,
                                                                                                      const Eigen::Isometry3d& p_matTransformationLEFTToWorldNow,
                                                                                                      const Eigen::Vector3d& p_vecCameraOrientation,
                                                                                                      const int32_t& p_iHalfLineLengthBase );

    const std::shared_ptr< std::vector< const CMeasurementLandmark* > > getVisibleLandmarksMocked( cv::Mat& p_matDisplayLEFT,
                                                                                                   cv::Mat& p_matDisplayRIGHT,
                                                                                                   const uint64_t p_uFrame,
                                                                                                   const cv::Mat& p_matImageLEFT,
                                                                                                   const cv::Mat& p_matImageRIGHT,
                                                                                                   const Eigen::Isometry3d& p_matTransformationLEFTToWorldNow );

    const std::shared_ptr< std::vector< const CMeasurementLandmark* > > getVisibleLandmarksFundamental( cv::Mat& p_matDisplayLEFT,
                                                                                                        cv::Mat& p_matDisplayRIGHT,
                                                                                                        const uint64_t p_uFrame,
                                                                                                        const cv::Mat& p_matImageLEFT,
                                                                                                        const cv::Mat& p_matImageRIGHT,
                                                                                                        const Eigen::Isometry3d& p_matTransformationLEFTToWorldNow,
                                                                                                        const Eigen::Vector3d& p_vecCameraOrientation,
                                                                                                        const double& p_dMotionScaling );

    const std::vector< CDetectionPoint >::size_type getNumberOfDetectionPointsActive( ) const{ return m_vecDetectionPointsActive.size( ); }

    const UIDDetectionPoint getNumberOfDetectionPointsTotal( ) const { return m_uAvailableDetectionPointID; }

    const UIDLandmark getNumberOfInvalidLandmarksTotal( ) const { return m_uNumberOfInvalidLandmarksTotal; }
    const UIDLandmark getNumberOfDetectionsPoseOptimization( ) const { return m_uNumberOfDetectionsPoseOptimization; }
    const UIDLandmark getNumberOfDetectionsEpipolar( ) const { return m_uNumberOfDetectionsEpipolar; }

private:

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

    const std::shared_ptr< CMatchTracking > _getMatchSampleRecursiveFundamentalU( cv::Mat& p_matDisplay,
                                                                const cv::Mat& p_matImage,
                                                                const double& p_dUMinimum,
                                                                const uint32_t& p_uDeltaU,
                                                                const Eigen::Vector3d& p_vecCoefficients,
                                                                const CDescriptor& p_matReferenceDescriptor,
                                                                const CDescriptor& p_matOriginalDescriptor,
                                                                const double& p_dKeyPointSize,
                                                                const uint8_t& p_uRecursionDepth ) const;

    const std::shared_ptr< CMatchTracking > _getMatchSampleRecursiveFundamentalV( cv::Mat& p_matDisplay,
                                                                const cv::Mat& p_matImage,
                                                                const double& p_dVMinimum,
                                                                const uint32_t& p_uDeltaV,
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

    const CMatchTracking* _getMatchCStyle( const cv::Mat& p_matImage,
                                     std::vector< cv::KeyPoint >& p_vecPoolKeyPoints,
                                     const CDescriptor& p_matDescriptorReference,
                                     const CDescriptor& p_matDescriptorOriginal ) const;

    const std::shared_ptr< CMatchTracking > _getMatch( const cv::Mat& p_matImage,
                                     std::vector< cv::KeyPoint >& p_vecPoolKeyPoints,
                                     const CDescriptor& p_matDescriptorReference,
                                     const CDescriptor& p_matDescriptorOriginal ) const;

    void _addMeasurementToLandmarkLEFT( const uint64_t p_uFrame,
                                   CLandmark* p_pLandmark,
                                   const cv::Mat& p_matImageRIGHT,
                                   const cv::KeyPoint& p_cKeyPoint,
                                   const CDescriptor& p_matDescriptorNew,
                                   const Eigen::Isometry3d& p_matTransformationLEFTtoWORLD,
                                   const Eigen::Vector3d& p_vecCameraOrientation,
                                   const MatrixProjection& p_matProjectionWORLDtoLEFT );
    void _addMeasurementToLandmarkSTEREO( const uint64_t p_uFrame,
                                    CMatchPoseOptimizationSTEREO& p_cMatchSTEREO,
                                    const Eigen::Isometry3d& p_matTransformationLEFTtoWORLD,
                                    const Eigen::Vector3d& p_vecCameraOrientation,
                                    const MatrixProjection& p_matProjectionWORLDtoLEFT );

    inline const double _getCurveEssentialX( const Eigen::Vector3d& p_vecCoefficients, const double& p_dY ) const;
    inline const double _getCurveEssentialY( const Eigen::Vector3d& p_vecCoefficients, const double& p_dX ) const;
    inline const int32_t _getCurveEssentialU( const Eigen::Vector3d& p_vecCoefficients, const int32_t& p_uV ) const;
    inline const int32_t _getCurveEssentialV( const Eigen::Vector3d& p_vecCoefficients, const int32_t& p_uU ) const;
    inline const int32_t _getCurveEssentialU( const Eigen::Vector3d& p_vecCoefficients, const double& p_uV ) const;
    inline const int32_t _getCurveEssentialV( const Eigen::Vector3d& p_vecCoefficients, const double& p_uU ) const;

    inline const double _getCurveFundamentalU( const Eigen::Vector3d& p_vecCoefficients, const double& p_dV ) const;
    inline const double _getCurveFundamentalV( const Eigen::Vector3d& p_vecCoefficients, const double& p_dU ) const;

};

#endif //#define CMATCHEREPIPOLAR_H
