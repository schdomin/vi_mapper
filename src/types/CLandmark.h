#ifndef CLANDMARK_H
#define CLANDMARK_H

#include "configuration/Types.h"

class CLandmark
{

public:

    CLandmark( const UIDLandmark& p_uID,
               const CDescriptor& p_matDescriptor,
               const CPoint3DInWorldFrame& p_vecPositionXYZ,
               const CPoint2DInCameraFrameHomogenized& p_cPositionUVReference );

    ~CLandmark( );

public:

    const UIDLandmark uID;

    const CDescriptor matDescriptorReference;
    CDescriptor matDescriptorLast;

    const CPoint3DInWorldFrame vecPositionXYZ;
    const CPoint2DInCameraFrameHomogenized vecPositionUVReference;

    uint8_t uFailedSubsequentTrackings;


private:

    std::vector< CPositionRaw* > m_vecPosition;

    //ds last camera position for calibration
    Eigen::Vector3d m_vecLastCameraPosition;

    //ds LM LS configuration
    const uint32_t m_uCalibrationPoints = 10 ;
    const uint32_t m_uIterations        = 100;
    const double m_dLevenbergDamping    = 100.0;
    const double m_dConvergenceDelta    = 1e-5;

public:

    //ds add detection
    void addPosition( const CPoint3DInWorldFrame& p_vecPointTriangulated,
                      const cv::Point2d& p_ptPointDetected,
                      const Eigen::Vector3d& p_vecCameraPosition,
                      const Eigen::Matrix3d& p_matKRotation,
                      const Eigen::Vector3d& p_vecKTranslation );
    const cv::Point2d getLastPosition( ) const;

};

#endif //CLANDMARK_H
