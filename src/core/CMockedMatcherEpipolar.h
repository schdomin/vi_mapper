#ifndef CMOCKEDMATCHEREPIPOLAR_H
#define CMOCKEDMATCHEREPIPOLAR_H

#include <memory>

#include "vision/CMockedStereoCamera.h"
#include "types/CLandmark.h"
#include "optimization/CPositSolver.h"

class CMockedMatcherEpipolar
{

//ds private structs
private:

    struct CLandmarkCreationPoint
    {
        const UIDMeasurementPoint uID;
        const Eigen::Isometry3d matTransformationLEFTtoWORLD;
        const std::shared_ptr< std::vector< CLandmark* > > vecLandmarks;

        CLandmarkCreationPoint( const UIDMeasurementPoint& p_uID,
                           const Eigen::Isometry3d& p_matTransformationLEFTtoWORLD,
                           const std::shared_ptr< std::vector< CLandmark* > > p_vecLandmarks ): uID( p_uID ),
                                                                                                matTransformationLEFTtoWORLD( p_matTransformationLEFTtoWORLD ),
                                                                                                vecLandmarks( p_vecLandmarks )
        {
            //ds nothing to do
        }
        ~CLandmarkCreationPoint( )
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
    UIDMeasurementPoint m_uAvailableMeasurementPointID;
    std::vector< CLandmarkCreationPoint > m_vecLandmarkCreationPointsActive;

    //ds internal
    const uint8_t m_uMaximumFailedSubsequentTrackingsPerLandmark;

    //ds debug logging
    gtools::CPositSolver m_cSolverPose;
    std::FILE* m_pFileOdometryError;
    std::FILE* m_pFileEpipolarDetection;

//ds api
public:

    void addMeasurementPoint( const Eigen::Isometry3d& p_matTransformationLEFTtoWORLD, const std::shared_ptr< std::vector< CLandmark* > > p_vecLandmarks );

    const Eigen::Isometry3d getPoseOptimized( const Eigen::Isometry3d& p_matTransformationEstimateWORLDtoLEFT );

    const std::shared_ptr< std::vector< const CMeasurementLandmark* > > getVisibleLandmarksEssential( const uint64_t p_uFrame,
                                                                                                      cv::Mat& p_matDisplayLEFT,
                                                                                                      cv::Mat& p_matDisplayRIGHT,
                                                                                                      const cv::Mat& p_matImageLEFT,
                                                                                                      const cv::Mat& p_matImageRIGHT,
                                                                                                      const Eigen::Isometry3d& p_matTransformationLEFTToWorldNow,
                                                                                                      cv::Mat& p_matDisplayTrajectory );

    const std::vector< CLandmarkCreationPoint >::size_type getNumberOfActiveMeasurementPoints( ) const
    {
        return m_vecLandmarkCreationPointsActive.size( );
    }

};

#endif //#define CMOCKEDMATCHEREPIPOLAR_H
