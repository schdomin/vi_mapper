#include "CBridgeG2O.h"

#include "g2o/core/sparse_optimizer.h"
#include "g2o/core/block_solver.h"
#include "g2o/core/hyper_graph.h"
#include "g2o/core/solver.h"
#include "g2o/core/optimization_algorithm_levenberg.h"
#include "g2o/solvers/dense/linear_solver_dense.h"
#include "g2o/types/slam3d/types_slam3d.h"

void CBridgeG2O::saveXYZAndDisparity( const std::string& p_strOutfile,
                                          const CStereoCamera& p_cStereoCamera,
                                          const std::vector< CLandmark* >& p_vecLandmarks,
                                          const std::vector< CMeasurementPose >& p_vecMeasurements )
{
    //ds allocate an optimizer
    g2o::OptimizableGraph cOptimizer;
    //g2o::SparseOptimizer cOptimizer;
    //cOptimizer.setVerbose( true );

    /*//ds set the solver
    g2o::BlockSolverX::LinearSolverType* pLinearSolver = new g2o::LinearSolverDense< g2o::BlockSolverX::PoseMatrixType> ( );
    g2o::BlockSolverX* pSolver                         = new g2o::BlockSolverX( pLinearSolver );
    g2o::OptimizationAlgorithmLevenberg* pAlgorithm    = new g2o::OptimizationAlgorithmLevenberg( pSolver );
    cOptimizer.setAlgorithm( pAlgorithm );*/

    //ds set world
    g2o::ParameterSE3Offset* pOffsetWorld = new g2o::ParameterSE3Offset( );
    pOffsetWorld->setOffset( Eigen::Isometry3d::Identity( ) );
    pOffsetWorld->setId( EG2OParameterID::eWORLD );
    cOptimizer.addParameter( pOffsetWorld );

    //ds set camera parameters
    g2o::ParameterCamera* pCameraParametersLEFT = new g2o::ParameterCamera( );
    pCameraParametersLEFT->setKcam( p_cStereoCamera.m_pCameraLEFT->m_dFx, p_cStereoCamera.m_pCameraLEFT->m_dFy, p_cStereoCamera.m_pCameraLEFT->m_dCx, p_cStereoCamera.m_pCameraLEFT->m_dCy );
    pCameraParametersLEFT->setId( EG2OParameterID::eCAMERA_LEFT );
    cOptimizer.addParameter( pCameraParametersLEFT );
    g2o::ParameterCamera* pCameraParametersRIGHT = new g2o::ParameterCamera( );
    pCameraParametersRIGHT->setOffset( p_cStereoCamera.m_matTransformLEFTtoRIGHT );
    pCameraParametersRIGHT->setKcam( p_cStereoCamera.m_pCameraRIGHT->m_dFx, p_cStereoCamera.m_pCameraRIGHT->m_dFy, p_cStereoCamera.m_pCameraRIGHT->m_dCx, p_cStereoCamera.m_pCameraRIGHT->m_dCy );
    pCameraParametersRIGHT->setId( EG2OParameterID::eCAMERA_RIGHT );
    cOptimizer.addParameter( pCameraParametersRIGHT );

    //ds g2o element identifier
    uint64_t uNextAvailableUID( 0 );

    //ds counter
    uint64_t uDroppedLandmarks( 0 );

    //ds add landmarks
    for( const CLandmark* pLandmark: p_vecLandmarks )
    {
        //ds check if calibration criteria is met
        if( m_uMinimumCalibrationsForDump < pLandmark->uCalibrations )
        {
            //ds set landmark vertex
            g2o::VertexPointXYZ* pVertexLandmark = new g2o::VertexPointXYZ( );
            pVertexLandmark->setEstimate( pLandmark->vecPointXYZCalibrated );
            pVertexLandmark->setId( uNextAvailableUID );

            assert( pLandmark->uID == uNextAvailableUID );

            //ds add vertex to optimizer
            cOptimizer.addVertex( pVertexLandmark );
        }
        else
        {
            //std::printf( "<CBridgeG2O>(savesolveAndOptimizeG2O) dropped landmark [%lu]\n", pLandmark->uID );
            ++uDroppedLandmarks;
        }

        //ds always increment
        ++uNextAvailableUID;
    }

    std::printf( "<CBridgeG2O>(saveXYZAndDisparity) dropped landmarks: %lu/%lu (%1.2f)\n", uDroppedLandmarks, p_vecLandmarks.size( ), static_cast< double >( uDroppedLandmarks )/p_vecLandmarks.size( ) );

    //ds add the first pose separately (there's no edge to the "previous" pose)
    g2o::VertexSE3 *pVertexPose = new g2o::VertexSE3( );
    pVertexPose->setEstimate( p_vecMeasurements.front( ).matTransformationLEFTtoWORLD );
    pVertexPose->setId( uNextAvailableUID );
    pVertexPose->setFixed( true );
    cOptimizer.addVertex( pVertexPose );
    ++uNextAvailableUID;

    //ds loop over the camera vertices vector (skipping the first one that we added before)
    for( std::vector< CMeasurementPose >::const_iterator pMeasurementPoint = p_vecMeasurements.begin( )+1; pMeasurementPoint != p_vecMeasurements.end( ); ++pMeasurementPoint )
    {
        //ds add current camera pose
        g2o::VertexSE3* pVertexPoseCurrent = new g2o::VertexSE3( );
        pVertexPoseCurrent->setEstimate( pMeasurementPoint->matTransformationLEFTtoWORLD );
        pVertexPoseCurrent->setId( uNextAvailableUID );
        cOptimizer.addVertex( pVertexPoseCurrent );
        ++uNextAvailableUID;

        //ds get previous vertex to link with current one
        g2o::VertexSE3* pVertexPosePrevious = dynamic_cast< g2o::VertexSE3* >( cOptimizer.vertices( ).find( uNextAvailableUID-2 )->second );

        //ds set up the edge
        g2o::EdgeSE3* pEdgePoseFromTo = new g2o::EdgeSE3( );

        //ds set viewpoints and measurement
        pEdgePoseFromTo->setVertex( 0, pVertexPosePrevious );
        pEdgePoseFromTo->setVertex( 1, pVertexPoseCurrent );
        pEdgePoseFromTo->setMeasurement( pVertexPosePrevious->estimate( ).inverse( )*pVertexPoseCurrent->estimate( ) );

        //ds add to optimizer
        cOptimizer.addEdge( pEdgePoseFromTo );

        //ds check visible landmarks and add the edges for the current pose
        for( const CMeasurementLandmark* pMeasurementLandmark: *pMeasurementPoint->vecLandmarks )
        {
            //ds find the corresponding landmark
            const g2o::HyperGraph::VertexIDMap::iterator itLandmark( cOptimizer.vertices( ).find( pMeasurementLandmark->uID ) );

            //ds if found
            if( itLandmark != cOptimizer.vertices( ).end( ) )
            {
                //ds get the respective feature vertex (this only works if the landmark id's are preserved in the optimizer)
                g2o::VertexPointXYZ* pVertexLandmark = dynamic_cast< g2o::VertexPointXYZ* >( itLandmark->second );

                //ds allocate the edges
                //g2o::EdgeSE3PointXYZUV* pEdgeLandmarkDetectionLEFT    = new g2o::EdgeSE3PointXYZUV( );
                //g2o::EdgeSE3PointXYZUV* pEdgeLandmarkDetectionRIGHT   = new g2o::EdgeSE3PointXYZUV( );
                g2o::EdgeSE3PointXYZ* pEdgeLandmarkPointXYZ           = new g2o::EdgeSE3PointXYZ( );
                g2o::EdgeSE3PointXYZDisparity* pEdgeLandmarkDisparity = new g2o::EdgeSE3PointXYZDisparity( );

                /*//ds LEFT camera UV measurement
                pEdgeLandmarkDetectionLEFT->setVertex( 0, pVertexPoseCurrent );
                pEdgeLandmarkDetectionLEFT->setVertex( 1, pVertexLandmark );
                pEdgeLandmarkDetectionLEFT->setMeasurement( CWrapperOpenCV::fromCVVector( pMeasurementLandmark->ptPositionUVLEFT ) );
                pEdgeLandmarkDetectionLEFT->setParameterId( 0, Eg2oParameterID::eCAMERA_LEFT );

                //ds RIGHT camera UV measurement
                pEdgeLandmarkDetectionRIGHT->setVertex( 0, pVertexPoseCurrent );
                pEdgeLandmarkDetectionRIGHT->setVertex( 1, pVertexLandmark );
                pEdgeLandmarkDetectionRIGHT->setMeasurement( CWrapperOpenCV::fromCVVector( pMeasurementLandmark->ptPositionUVRIGHT ) );
                pEdgeLandmarkDetectionRIGHT->setParameterId( 0, Eg2oParameterID::eCAMERA_RIGHT );*/

                //ds triangulated 3d point (uncalibrated)
                pEdgeLandmarkPointXYZ->setVertex( 0, pVertexPoseCurrent );
                pEdgeLandmarkPointXYZ->setVertex( 1, pVertexLandmark );
                pEdgeLandmarkPointXYZ->setMeasurement( pMeasurementLandmark->vecPointXYZ );
                pEdgeLandmarkPointXYZ->setParameterId( 0, EG2OParameterID::eWORLD );

                //ds disparity (for LEFT camera)
                const double dDisparity( ( pMeasurementLandmark->ptUVLEFT.x-pMeasurementLandmark->ptUVRIGHT.x )/( p_cStereoCamera.m_pCameraLEFT->m_dFx*p_cStereoCamera.m_dBaselineMeters ) );
                pEdgeLandmarkDisparity->setVertex( 0, pVertexPoseCurrent );
                pEdgeLandmarkDisparity->setVertex( 1, pVertexLandmark );
                pEdgeLandmarkDisparity->setMeasurement( g2o::Vector3D( pMeasurementLandmark->ptUVLEFT.x, pMeasurementLandmark->ptUVLEFT.y, dDisparity ) );
                pEdgeLandmarkDisparity->setParameterId( 0, EG2OParameterID::eCAMERA_LEFT );

                //ds add to optimizer
                //cOptimizer.addEdge( pEdgeLandmarkDetectionLEFT );
                //cOptimizer.addEdge( pEdgeLandmarkDetectionRIGHT );
                cOptimizer.addEdge( pEdgeLandmarkPointXYZ );
                cOptimizer.addEdge( pEdgeLandmarkDisparity );
            }
        }
    }

    cOptimizer.save( p_strOutfile.c_str( ) );

    //ds optimize!
    //cOptimizer.initializeOptimization( );
    //cOptimizer.computeActiveErrors( );
    //cOptimizer.optimize( 1 );
}

void CBridgeG2O::saveUVDepthOrDisparity( const std::string& p_strOutfile,
                                          const CStereoCamera& p_cStereoCamera,
                                          const std::vector< CLandmark* >& p_vecLandmarks,
                                          const std::vector< CMeasurementPose >& p_vecMeasurements )
{
    //ds allocate an optimizer
    g2o::OptimizableGraph cGraph;

    /*//ds set world
    g2o::ParameterSE3Offset* pOffsetWorld = new g2o::ParameterSE3Offset( );
    pOffsetWorld->setOffset( Eigen::Isometry3d::Identity( ) );
    pOffsetWorld->setId( EG2OParameterID::eWORLD );
    cGraph.addParameter( pOffsetWorld );*/

    //ds set camera parameters
    g2o::ParameterCamera* pCameraParametersLEFT = new g2o::ParameterCamera( );
    pCameraParametersLEFT->setKcam( p_cStereoCamera.m_pCameraLEFT->m_dFx, p_cStereoCamera.m_pCameraLEFT->m_dFy, p_cStereoCamera.m_pCameraLEFT->m_dCx, p_cStereoCamera.m_pCameraLEFT->m_dCy );
    pCameraParametersLEFT->setId( EG2OParameterID::eCAMERA_LEFT );
    cGraph.addParameter( pCameraParametersLEFT );
    /*g2o::ParameterCamera* pCameraParametersRIGHT = new g2o::ParameterCamera( );
    pCameraParametersRIGHT->setOffset( p_cStereoCamera.m_matTransformLEFTtoRIGHT );
    pCameraParametersRIGHT->setKcam( p_cStereoCamera.m_pCameraRIGHT->m_dFx, p_cStereoCamera.m_pCameraRIGHT->m_dFy, p_cStereoCamera.m_pCameraRIGHT->m_dCx, p_cStereoCamera.m_pCameraRIGHT->m_dCy );
    pCameraParametersRIGHT->setId( EG2OParameterID::eCAMERA_RIGHT );
    cGraph.addParameter( pCameraParametersRIGHT );*/

    /*//ds set camera parameters (NORMALIZED)
    g2o::ParameterCamera* pCameraParametersLEFT = new g2o::ParameterCamera( );
    pCameraParametersLEFT->setKcam( p_cStereoCamera.m_pCameraLEFT->m_dFxNormalized, p_cStereoCamera.m_pCameraLEFT->m_dFyNormalized, p_cStereoCamera.m_pCameraLEFT->m_dCxNormalized, p_cStereoCamera.m_pCameraLEFT->m_dCyNormalized );
    pCameraParametersLEFT->setId( EG2OParameterID::eCAMERA_LEFT );
    cGraph.addParameter( pCameraParametersLEFT );
    g2o::ParameterCamera* pCameraParametersRIGHT = new g2o::ParameterCamera( );
    pCameraParametersRIGHT->setOffset( p_cStereoCamera.m_matTransformLEFTtoRIGHT );
    pCameraParametersRIGHT->setKcam( p_cStereoCamera.m_pCameraRIGHT->m_dFxNormalized, p_cStereoCamera.m_pCameraRIGHT->m_dFyNormalized, p_cStereoCamera.m_pCameraRIGHT->m_dCxNormalized, p_cStereoCamera.m_pCameraRIGHT->m_dCyNormalized );
    pCameraParametersRIGHT->setId( EG2OParameterID::eCAMERA_RIGHT );
    cGraph.addParameter( pCameraParametersRIGHT );*/

    //ds imu offset (as IMU to LEFT)
    g2o::ParameterSE3Offset* pOffsetIMUtoLEFT = new g2o::ParameterSE3Offset( );
    pOffsetIMUtoLEFT->setOffset( p_cStereoCamera.m_pCameraLEFT->m_matTransformationFromIMU );
    pOffsetIMUtoLEFT->setId( EG2OParameterID::eOFFSET_IMUtoLEFT );
    cGraph.addParameter( pOffsetIMUtoLEFT );

    //ds g2o element identifier
    uint64_t uNextAvailableUID( 0 );

    //ds counter
    uint64_t uDroppedLandmarks( 0 );

    //ds add landmarks
    for( const CLandmark* pLandmark: p_vecLandmarks )
    {
        //ds check if calibration criteria is met
        if( m_uMinimumCalibrationsForDump < pLandmark->uCalibrations )
        {
            //ds set landmark vertex
            g2o::VertexPointXYZ* pVertexLandmark = new g2o::VertexPointXYZ( );
            pVertexLandmark->setEstimate( pLandmark->vecPointXYZCalibrated );
            pVertexLandmark->setId( uNextAvailableUID );

            assert( pLandmark->uID == uNextAvailableUID );

            //ds add vertex to optimizer
            cGraph.addVertex( pVertexLandmark );
        }
        else
        {
            //std::printf( "<CBridgeG2O>(savesolveAndOptimizeG2O) dropped landmark [%lu]\n", pLandmark->uID );
            ++uDroppedLandmarks;
        }

        //ds always increment
        ++uNextAvailableUID;
    }

    std::printf( "<CBridgeG2O>(saveUVDepthOrDisparity) dropped landmarks: %lu/%lu (%1.2f)\n", uDroppedLandmarks, p_vecLandmarks.size( ), static_cast< double >( uDroppedLandmarks )/p_vecLandmarks.size( ) );

    //ds add the first pose separately (there's no edge to the "previous" pose)
    g2o::VertexSE3 *pVertexPose = new g2o::VertexSE3( );
    pVertexPose->setEstimate( p_vecMeasurements.front( ).matTransformationLEFTtoWORLD );
    pVertexPose->setId( uNextAvailableUID );
    pVertexPose->setFixed( true );
    cGraph.addVertex( pVertexPose );
    ++uNextAvailableUID;

    //ds loop over the camera vertices vector (skipping the first one that we added before)
    for( std::vector< CMeasurementPose >::const_iterator pMeasurementPoint = p_vecMeasurements.begin( )+1; pMeasurementPoint != p_vecMeasurements.end( ); ++pMeasurementPoint )
    {
        //ds add current camera pose
        g2o::VertexSE3* pVertexPoseCurrent = new g2o::VertexSE3( );
        pVertexPoseCurrent->setEstimate( pMeasurementPoint->matTransformationLEFTtoWORLD );
        pVertexPoseCurrent->setId( uNextAvailableUID );
        cGraph.addVertex( pVertexPoseCurrent );
        ++uNextAvailableUID;

        //ds get previous vertex to link with current one
        g2o::VertexSE3* pVertexPosePrevious = dynamic_cast< g2o::VertexSE3* >( cGraph.vertices( ).find( uNextAvailableUID-2 )->second );

        //ds set up the edge
        g2o::EdgeSE3* pEdgePoseFromTo = new g2o::EdgeSE3( );

        //ds set viewpoints and measurement
        pEdgePoseFromTo->setVertex( 0, pVertexPosePrevious );
        pEdgePoseFromTo->setVertex( 1, pVertexPoseCurrent );
        pEdgePoseFromTo->setMeasurement( pVertexPosePrevious->estimate( ).inverse( )*pVertexPoseCurrent->estimate( ) );

        //ds information quality
        const double arrInformationMatrixPose[36] = { 100,0,0,0,0,0,
                                                  0,100,0,0,0,0,
                                                  0,0,100,0,0,0,
                                                  0,0,0,10000,0,0,
                                                  0,0,0,0,10000,0,
                                                  0,0,0,0,0,10000 };
        pEdgePoseFromTo->setInformation( Eigen::Matrix< double, 6, 6 >( arrInformationMatrixPose ) );

        //ds add to optimizer
        cGraph.addEdge( pEdgePoseFromTo );

        //ds always save acceleration data
        g2o::EdgeSE3LinearAcceleration* pEdgeLinearAcceleration = new g2o::EdgeSE3LinearAcceleration( );

        pEdgeLinearAcceleration->setVertex( 0, pVertexPoseCurrent );
        pEdgeLinearAcceleration->setMeasurement( pMeasurementPoint->vecLinearAccelerationNormalized );
        pEdgeLinearAcceleration->setParameterId( 0, EG2OParameterID::eOFFSET_IMUtoLEFT );
        const double arrInformationMatrixLinearAcceleration[9] = { 100000, 0, 0, 0, 100000, 0, 0, 0, 100000 };
        pEdgeLinearAcceleration->setInformation( g2o::Matrix3D( arrInformationMatrixLinearAcceleration ) );
        cGraph.addEdge( pEdgeLinearAcceleration );

        //ds check visible landmarks and add the edges for the current pose
        for( const CMeasurementLandmark* pMeasurementLandmark: *pMeasurementPoint->vecLandmarks )
        {
            //ds find the corresponding landmark
            const g2o::HyperGraph::VertexIDMap::iterator itLandmark( cGraph.vertices( ).find( pMeasurementLandmark->uID ) );

            //ds if found
            if( itLandmark != cGraph.vertices( ).end( ) )
            {
                //ds get the respective feature vertex (this only works if the landmark id's are preserved in the optimizer)
                g2o::VertexPointXYZ* pVertexLandmark = dynamic_cast< g2o::VertexPointXYZ* >( itLandmark->second );

                //ds get key values
                const int32_t iDisparity( pMeasurementLandmark->ptUVLEFT.x-pMeasurementLandmark->ptUVRIGHT.x );
                const double dDepthMeters( pMeasurementLandmark->vecPointXYZ(2) );

                //ds quality rating (the bigger the worse: far away and low disparity)
                const double dImprecision( dDepthMeters/( std::abs( iDisparity )+1 ) );

                //ds minimum quality to produce a reliable depth estimate
                if( 2.0 > dImprecision )
                {
                    //ds projected depth (LEFT camera)
                    g2o::EdgeSE3PointXYZDepth* pEdgeProjectedDepth = new g2o::EdgeSE3PointXYZDepth( );

                    pEdgeProjectedDepth->setVertex( 0, pVertexPoseCurrent );
                    pEdgeProjectedDepth->setVertex( 1, pVertexLandmark );
                    pEdgeProjectedDepth->setMeasurement( g2o::Vector3D( pMeasurementLandmark->ptUVLEFT.x, pMeasurementLandmark->ptUVLEFT.y, dDepthMeters ) );
                    pEdgeProjectedDepth->setParameterId( 0, EG2OParameterID::eCAMERA_LEFT );

                    //ds information matrix
                    const double dInformationQualityDepth( 1.0/dImprecision );
                    const double arrInformationMatrixDepth[9] = { 1, 0, 0, 0, 1, 0, 0, 0, 100*dInformationQualityDepth };
                    pEdgeProjectedDepth->setInformation( g2o::Matrix3D( arrInformationMatrixDepth ) );

                    cGraph.addEdge( pEdgeProjectedDepth );
                }
                else
                {
                    //ds disparity (LEFT camera)
                    g2o::EdgeSE3PointXYZDisparity* pEdgeLandmarkDisparity = new g2o::EdgeSE3PointXYZDisparity( );

                    const double dDisparity( iDisparity/( p_cStereoCamera.m_pCameraLEFT->m_dFx*p_cStereoCamera.m_dBaselineMeters ) );
                    pEdgeLandmarkDisparity->setVertex( 0, pVertexPoseCurrent );
                    pEdgeLandmarkDisparity->setVertex( 1, pVertexLandmark );
                    pEdgeLandmarkDisparity->setMeasurement( g2o::Vector3D( pMeasurementLandmark->ptUVLEFT.x, pMeasurementLandmark->ptUVLEFT.y, dDisparity ) );
                    pEdgeLandmarkDisparity->setParameterId( 0, EG2OParameterID::eCAMERA_LEFT );

                    //ds information matrix
                    const double dInformationQualityDisparity( std::abs( iDisparity )+1 );
                    const double arrInformationMatrixDisparity[9] = { 1, 0, 0, 0, 1, 0, 0, 0, dInformationQualityDisparity };
                    pEdgeLandmarkDisparity->setInformation( g2o::Matrix3D( arrInformationMatrixDisparity ) );

                    cGraph.addEdge( pEdgeLandmarkDisparity );

                    std::printf( "<CBridgeG2O>(saveUVDepthOrDisparity) landmark [%lu] imprecision: %f - stored disparity measurement (%ip) instead of depth (%fm)\n", pMeasurementLandmark->uID, dImprecision, iDisparity, dDepthMeters );
                }
            }
        }
    }

    cGraph.save( p_strOutfile.c_str( ) );
}
