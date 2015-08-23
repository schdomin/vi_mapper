#ifndef CBRIDGEG2O_H
#define CBRIDGEG2O_H

#include "vision/CStereoCamera.h"
#include "types/CLandmark.h"
#include "types/CKeyFrame.h"

#include "g2o/types/slam3d/types_slam3d.h"

class CBridgeG2O
{
    enum EG2OParameterID
    {
        eWORLD        = 0,
        eCAMERA_LEFT  = 1,
        eCAMERA_RIGHT = 2,
        eOFFSET_IMUtoLEFT = 3
    };

private:

    static constexpr double m_dMaximumReliableDepthForPointXYZ = 2.5;
    static constexpr double m_dMaximumReliableDepthForUVDepth  = 7.5;
    static const uint8_t m_uMinimumOptimizations               = 0;
    static constexpr double m_dMaximumErrorPerOptimization     = 10.0; //ds e.g after 3 optimizations an error of 30.0 is allowed
    static constexpr uint8_t m_uMinimumKeyFramePresences       = 1;

public:

    static void saveXYZ( const std::string& p_strOutfile,
                         const CStereoCamera& p_cStereoCamera,
                         const std::vector< CLandmark* >& p_vecLandmarks,
                         const std::vector< CKeyFrame* >& p_vecKeyFrames );

    static void saveUVDepth( const std::string& p_strOutfile,
                             const CStereoCamera& p_cStereoCamera,
                             const std::vector< CLandmark* >& p_vecLandmarks,
                             const std::vector< CKeyFrame* >& p_vecKeyFrames );

    static void saveUVDisparity( const std::string& p_strOutfile,
                                 const CStereoCamera& p_cStereoCamera,
                                 const std::vector< CLandmark* >& p_vecLandmarks,
                                 const std::vector< CKeyFrame* >& p_vecKeyFrames );

    static void saveUVDepthOrDisparity( const std::string& p_strOutfile,
                                        const CStereoCamera& p_cStereoCamera,
                                        const std::vector< CLandmark* >& p_vecLandmarks,
                                        const std::vector< CKeyFrame* >& p_vecKeyFrames );

    static void saveCOMBO( const std::string& p_strOutfile,
                           const CStereoCamera& p_cStereoCamera,
                           const std::vector< CLandmark* >& p_vecLandmarks,
                           const std::vector< CKeyFrame* >& p_vecKeyFrames );

    static const bool isOptimized( const CLandmark* p_pLandmark );
    static const bool isKeyFramed( const CLandmark* p_pLandmark );

private:

    static g2o::EdgeSE3PointXYZ* _getEdgePointXYZ( g2o::VertexSE3* p_pVertexPose,
                                                   g2o::VertexPointXYZ* p_pVertexLandmark,
                                                   const EG2OParameterID& p_eParameterIDOriginWORLD,
                                                   const CPoint3DWORLD& p_vecPointXYZ );
    static g2o::EdgeSE3PointXYZDepth* _getEdgeUVDepth( g2o::VertexSE3* p_pVertexPose,
                                                       g2o::VertexPointXYZ* p_pVertexLandmark,
                                                       const EG2OParameterID& p_eParameterIDCamera,
                                                       const CMeasurementLandmark* p_pMeasurement );
    static g2o::EdgeSE3PointXYZDisparity* _getEdgeUVDisparity( g2o::VertexSE3* p_pVertexPose,
                                                               g2o::VertexPointXYZ* p_pVertexLandmark,
                                                               const EG2OParameterID& p_eParameterIDCamera,
                                                               const double& p_dDisparityPixels,
                                                               const CMeasurementLandmark* p_pMeasurement,
                                                               const double& p_dFxPixels,
                                                               const double& p_dBaselineMeters );

};

#endif //CBRIDGEG2O_H
