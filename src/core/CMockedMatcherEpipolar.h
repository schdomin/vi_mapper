#ifndef CMOCKEDMATCHEREPIPOLAR_H
#define CMOCKEDMATCHEREPIPOLAR_H

#include <memory>

#include "vision/CMockedStereoCamera.h"
#include "types/CLandmark.h"
#include "optimization/CPositSolverProjection.h"
#include "optimization/CPositSolverStereo.h"

class CMockedMatcherEpipolar
{

//ds private structs
private:

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

//ds ctor/dtor
public:

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    CMockedMatcherEpipolar( const std::shared_ptr< CPinholeCamera > p_pCameraLEFT,
                            const std::shared_ptr< CPinholeCamera > p_pCameraRIGHT,
                            const std::shared_ptr< CMockedStereoCamera > p_pCameraSTEREO,
                            const uint8_t& p_uMaximumFailedSubsequentTrackingsPerLandmark );
    ~CMockedMatcherEpipolar( );

//ds members
private:

    //ds cameras (not necessarily used in stereo here)
    const std::shared_ptr< CPinholeCamera > m_pCameraLEFT;
    const std::shared_ptr< CPinholeCamera > m_pCameraRIGHT;
    const std::shared_ptr< CMockedStereoCamera > m_pCameraSTEREO;

    //ds measurement point storage (we use the ID counter instead of accessing the vector size every time for speed)
    UIDDetectionPoint m_uAvailableMeasurementPointID;
    std::vector< CDetectionPoint > m_vecDetectionPointsActive;

    //ds internal
    const uint8_t m_uMaximumFailedSubsequentTrackingsPerLandmark;
    const uint8_t m_uMinimumPointsForPoseOptimization  = 5;
    const uint8_t m_uMinimumInliersForPoseOptimization = 2;
    const uint8_t m_uCapIterationsPoseOptimization     = 10;
    const double m_dConvergenceDeltaPoseOptimization   = 1e-5;

    //ds debug logging
    gtools::CPositSolverProjection m_cSolverPose;
    gtools::CPositSolverStereo m_cSolverPoseSTEREO;

//ds api
public:

    void addMeasurementPoint( const Eigen::Isometry3d& p_matTransformationLEFTtoWORLD, const std::shared_ptr< std::vector< CLandmark* > > p_vecLandmarks );

    const std::shared_ptr< std::vector< const CMeasurementLandmark* > > getVisibleLandmarks( const uint64_t p_uFrame,
                                                                                                      cv::Mat& p_matDisplayLEFT,
                                                                                                      cv::Mat& p_matDisplayRIGHT,
                                                                                                      const cv::Mat& p_matImageLEFT,
                                                                                                      const cv::Mat& p_matImageRIGHT,
                                                                                                      const Eigen::Isometry3d& p_matTransformationLEFTToWorldNow,
                                                                                                      cv::Mat& p_matDisplayTrajectory,
                                                                                                      const double& p_dMotionScaling );

    const std::vector< CDetectionPoint >::size_type getNumberOfActiveMeasurementPoints( ) const
    {
        return m_vecDetectionPointsActive.size( );
    }

    //ds routine that resets the visibility of all active landmarks
    void resetVisibilityActiveLandmarks( );

    //ds register keyframing on currently visible landmarks
    void setKeyFrameToVisibleLandmarks( );

};

#endif //#define CMOCKEDMATCHEREPIPOLAR_H
