#include "CBridgeG2O.h"

#include "g2o/core/sparse_optimizer.h"
#include "g2o/core/block_solver.h"
#include "g2o/core/hyper_graph.h"
#include "g2o/core/solver.h"
#include "g2o/core/optimization_algorithm_levenberg.h"
#include "g2o/solvers/dense/linear_solver_dense.h"

void CBridgeG2O::saveXYZ( const std::string& p_strOutfile,
                          const CStereoCamera& p_cStereoCamera,
                          const std::vector< CLandmark* >& p_vecLandmarks,
                          const std::vector< CKeyFrame >& p_vecKeyFrames )
{
    //ds append postfix
    const std::string strOutfile( p_strOutfile+"_XYZ.g2o" );

    //ds validate input
    if( p_vecLandmarks.empty( ) )
    {
        std::printf( "<CBridgeG2O>(saveXYZ) received empty landmarks vector, call ignored\n" );
        return;
    }
    if( p_vecKeyFrames.empty( ) )
    {
        std::printf( "<CBridgeG2O>(saveXYZ) received empty measurements vector, call ignored\n" );
        return;
    }

    //ds allocate an optimizer
    g2o::OptimizableGraph cGraph;
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
    cGraph.addParameter( pOffsetWorld );

    /*//ds set camera parameters
    g2o::ParameterCamera* pCameraParametersLEFT = new g2o::ParameterCamera( );
    pCameraParametersLEFT->setKcam( p_cStereoCamera.m_pCameraLEFT->m_dFx, p_cStereoCamera.m_pCameraLEFT->m_dFy, p_cStereoCamera.m_pCameraLEFT->m_dCx, p_cStereoCamera.m_pCameraLEFT->m_dCy );
    pCameraParametersLEFT->setId( EG2OParameterID::eCAMERA_LEFT );
    cGraph.addParameter( pCameraParametersLEFT );
    g2o::ParameterCamera* pCameraParametersRIGHT = new g2o::ParameterCamera( );
    pCameraParametersRIGHT->setOffset( p_cStereoCamera.m_matTransformLEFTtoRIGHT );
    pCameraParametersRIGHT->setKcam( p_cStereoCamera.m_pCameraRIGHT->m_dFx, p_cStereoCamera.m_pCameraRIGHT->m_dFy, p_cStereoCamera.m_pCameraRIGHT->m_dCx, p_cStereoCamera.m_pCameraRIGHT->m_dCy );
    pCameraParametersRIGHT->setId( EG2OParameterID::eCAMERA_RIGHT );
    cGraph.addParameter( pCameraParametersRIGHT );*/

    //ds imu offset (as IMU to LEFT)
    g2o::ParameterSE3Offset* pOffsetIMUtoLEFT = new g2o::ParameterSE3Offset( );
    pOffsetIMUtoLEFT->setOffset( p_cStereoCamera.m_pCameraLEFT->m_matTransformationIMUtoLEFT );
    pOffsetIMUtoLEFT->setId( EG2OParameterID::eOFFSET_IMUtoLEFT );
    cGraph.addParameter( pOffsetIMUtoLEFT );

    //ds counter
    uint64_t uDroppedLandmarks = 0;

    //ds add landmarks
    for( const CLandmark* pLandmark: p_vecLandmarks )
    {
        //ds check if calibration criteria is met
        if( CBridgeG2O::isOptimized( pLandmark ) && CBridgeG2O::isKeyFramed( pLandmark ) )
        {
            //ds set landmark vertex
            g2o::VertexPointXYZ* pVertexLandmark = new g2o::VertexPointXYZ( );
            pVertexLandmark->setEstimate( pLandmark->vecPointXYZOptimized );
            pVertexLandmark->setId( pLandmark->uID );

            //ds add vertex to optimizer
            cGraph.addVertex( pVertexLandmark );
        }
        else
        {
            //std::printf( "<CBridgeG2O>(savesolveAndOptimizeG2O) dropped landmark [%lu]\n", pLandmark->uID );
            ++uDroppedLandmarks;
        }
    }

    std::printf( "<CBridgeG2O>(saveXYZ) dropped landmarks: %lu/%lu (%1.2f)\n", uDroppedLandmarks, p_vecLandmarks.size( ), static_cast< double >( uDroppedLandmarks )/p_vecLandmarks.size( ) );

    const uint64_t uMeasurements = p_vecLandmarks.size( )-uDroppedLandmarks;

    //ds g2o element identifier
    uint64_t uNextAvailableUID( p_vecLandmarks.back( )->uID+1 );

    //ds add the first pose separately (there's no edge to the "previous" pose)
    g2o::VertexSE3 *pVertexPoseInitial = new g2o::VertexSE3( );
    pVertexPoseInitial->setEstimate( p_vecKeyFrames.front( ).matTransformationLEFTtoWORLD );
    pVertexPoseInitial->setId( uNextAvailableUID );
    pVertexPoseInitial->setFixed( true );
    cGraph.addVertex( pVertexPoseInitial );
    ++uNextAvailableUID;

    //ds always save acceleration data
    g2o::EdgeSE3LinearAcceleration* pEdgeLinearAccelerationInitial = new g2o::EdgeSE3LinearAcceleration( );

    pEdgeLinearAccelerationInitial->setVertex( 0, pVertexPoseInitial );
    pEdgeLinearAccelerationInitial->setMeasurement( p_vecKeyFrames.front( ).vecLinearAccelerationNormalized );
    pEdgeLinearAccelerationInitial->setParameterId( 0, EG2OParameterID::eOFFSET_IMUtoLEFT );
    const double arrInformationMatrixLinearAcceleration[9] = { 1000, 0, 0, 0, 1000, 0, 0, 0, 1000 };
    pEdgeLinearAccelerationInitial->setInformation( g2o::Matrix3D( arrInformationMatrixLinearAcceleration ) );
    cGraph.addEdge( pEdgeLinearAccelerationInitial );

    uint64_t uMeasurementsStoredXYZ       = 0;
    //uint64_t uMeasurementsStoredDisparity = 0;

    //ds loop over the camera vertices vector (skipping the first one that we added before)
    for( std::vector< CKeyFrame >::const_iterator pKeyFrame = p_vecKeyFrames.begin( )+1; pKeyFrame != p_vecKeyFrames.end( ); ++pKeyFrame )
    {
        //ds add current camera pose
        g2o::VertexSE3* pVertexPoseCurrent = new g2o::VertexSE3( );
        pVertexPoseCurrent->setEstimate( pKeyFrame->matTransformationLEFTtoWORLD );
        pVertexPoseCurrent->setId( uNextAvailableUID );
        cGraph.addVertex( pVertexPoseCurrent );

        //ds get previous vertex to link with current one
        g2o::VertexSE3* pVertexPosePrevious = dynamic_cast< g2o::VertexSE3* >( cGraph.vertices( ).find( uNextAvailableUID-1 )->second );
        ++uNextAvailableUID;

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
        pEdgeLinearAcceleration->setMeasurement( pKeyFrame->vecLinearAccelerationNormalized );
        pEdgeLinearAcceleration->setParameterId( 0, EG2OParameterID::eOFFSET_IMUtoLEFT );
        const double arrInformationMatrixLinearAcceleration[9] = { 1000, 0, 0, 0, 1000, 0, 0, 0, 1000 };
        pEdgeLinearAcceleration->setInformation( g2o::Matrix3D( arrInformationMatrixLinearAcceleration ) );
        cGraph.addEdge( pEdgeLinearAcceleration );

        //ds check visible landmarks and add the edges for the current pose
        for( const CMeasurementLandmark* pMeasurementLandmark: *pKeyFrame->vecLandmarkMeasurements )
        {
            //ds find the corresponding landmark
            const g2o::HyperGraph::VertexIDMap::iterator itLandmark( cGraph.vertices( ).find( pMeasurementLandmark->uID ) );

            //ds if found
            if( itLandmark != cGraph.vertices( ).end( ) )
            {
                //ds get the respective feature vertex (this only works if the landmark id's are preserved in the optimizer)
                g2o::VertexPointXYZ* pVertexLandmark = dynamic_cast< g2o::VertexPointXYZ* >( itLandmark->second );

                //ds maximum depth to produce a reliable XYZ estimate
                //if( CBridgeG2O::m_dMaximumReliableDepthForPointXYZ > pMeasurementLandmark->vecPointXYZLEFT.z( ) )
                //{
                    cGraph.addEdge( _getEdgePointXYZ( pVertexPoseCurrent, pVertexLandmark, EG2OParameterID::eWORLD, pMeasurementLandmark->vecPointXYZLEFT ) );
                    ++uMeasurementsStoredXYZ;
                //}
            }
        }
    }

    std::printf( "<CBridgeG2O>(saveXYZ) stored measurements EdgeSE3PointXYZ: %lu/%lu\n", uMeasurementsStoredXYZ, uMeasurements );

    cGraph.save( strOutfile.c_str( ) );

    //ds optimize!
    //cOptimizer.initializeOptimization( );
    //cOptimizer.computeActiveErrors( );
    //cOptimizer.optimize( 1 );
}

void CBridgeG2O::saveUVDepth( const std::string& p_strOutfile,
                              const CStereoCamera& p_cStereoCamera,
                              const std::vector< CLandmark* >& p_vecLandmarks,
                              const std::vector< CKeyFrame >& p_vecKeyFrames )
{
    //ds append postfix
    const std::string strOutfile( p_strOutfile+"_UVDepth.g2o" );

    //ds validate input
    if( p_vecLandmarks.empty( ) )
    {
        std::printf( "<CBridgeG2O>(saveUVDepth) received empty landmarks vector, call ignored\n" );
        return;
    }
    if( p_vecKeyFrames.empty( ) )
    {
        std::printf( "<CBridgeG2O>(saveUVDepth) received empty measurements vector, call ignored\n" );
        return;
    }

    //ds allocate an optimizer
    g2o::OptimizableGraph cGraph;

    //ds set camera parameters
    g2o::ParameterCamera* pCameraParametersLEFT = new g2o::ParameterCamera( );
    pCameraParametersLEFT->setKcam( p_cStereoCamera.m_pCameraLEFT->m_dFxP, p_cStereoCamera.m_pCameraLEFT->m_dFyP, p_cStereoCamera.m_pCameraLEFT->m_dCxP, p_cStereoCamera.m_pCameraLEFT->m_dCyP );
    pCameraParametersLEFT->setId( EG2OParameterID::eCAMERA_LEFT );
    cGraph.addParameter( pCameraParametersLEFT );

    //ds imu offset (as IMU to LEFT)
    g2o::ParameterSE3Offset* pOffsetIMUtoLEFT = new g2o::ParameterSE3Offset( );
    pOffsetIMUtoLEFT->setOffset( p_cStereoCamera.m_pCameraLEFT->m_matTransformationIMUtoLEFT );
    pOffsetIMUtoLEFT->setId( EG2OParameterID::eOFFSET_IMUtoLEFT );
    cGraph.addParameter( pOffsetIMUtoLEFT );

    //ds counter
    uint64_t uDroppedLandmarks( 0 );

    //ds add landmarks
    for( const CLandmark* pLandmark: p_vecLandmarks )
    {
        //ds check if calibration criteria is met
        if( CBridgeG2O::isOptimized( pLandmark ) && CBridgeG2O::isKeyFramed( pLandmark ) )
        {
            //ds set landmark vertex
            g2o::VertexPointXYZ* pVertexLandmark = new g2o::VertexPointXYZ( );
            pVertexLandmark->setEstimate( pLandmark->vecPointXYZOptimized );
            pVertexLandmark->setId( pLandmark->uID );

            //ds add vertex to optimizer
            cGraph.addVertex( pVertexLandmark );
        }
        else
        {
            //std::printf( "<CBridgeG2O>(savesolveAndOptimizeG2O) dropped landmark [%lu]\n", pLandmark->uID );
            ++uDroppedLandmarks;
        }
    }

    std::printf( "<CBridgeG2O>(saveUVDepth) dropped landmarks: %lu/%lu (%1.2f)\n", uDroppedLandmarks, p_vecLandmarks.size( ), static_cast< double >( uDroppedLandmarks )/p_vecLandmarks.size( ) );

    //ds g2o element identifier
    uint64_t uNextAvailableUID( p_vecLandmarks.back( )->uID+1 );

    //ds add the first pose separately (there's no edge to the "previous" pose)
    g2o::VertexSE3 *pVertexPoseInitial = new g2o::VertexSE3( );
    pVertexPoseInitial->setEstimate( p_vecKeyFrames.front( ).matTransformationLEFTtoWORLD );
    pVertexPoseInitial->setId( uNextAvailableUID );
    pVertexPoseInitial->setFixed( true );
    cGraph.addVertex( pVertexPoseInitial );
    ++uNextAvailableUID;

    //ds always save acceleration data
    g2o::EdgeSE3LinearAcceleration* pEdgeLinearAccelerationInitial = new g2o::EdgeSE3LinearAcceleration( );

    pEdgeLinearAccelerationInitial->setVertex( 0, pVertexPoseInitial );
    pEdgeLinearAccelerationInitial->setMeasurement( p_vecKeyFrames.front( ).vecLinearAccelerationNormalized );
    pEdgeLinearAccelerationInitial->setParameterId( 0, EG2OParameterID::eOFFSET_IMUtoLEFT );
    const double arrInformationMatrixLinearAcceleration[9] = { 1000, 0, 0, 0, 1000, 0, 0, 0, 1000 };
    pEdgeLinearAccelerationInitial->setInformation( g2o::Matrix3D( arrInformationMatrixLinearAcceleration ) );
    cGraph.addEdge( pEdgeLinearAccelerationInitial );

    uint64_t uMeasurementsStoredDepth = 0;

    //ds loop over the camera vertices vector (skipping the first one that we added before)
    for( std::vector< CKeyFrame >::const_iterator pKeyFrame = p_vecKeyFrames.begin( )+1; pKeyFrame != p_vecKeyFrames.end( ); ++pKeyFrame )
    {
        //ds add current camera pose
        g2o::VertexSE3* pVertexPoseCurrent = new g2o::VertexSE3( );
        pVertexPoseCurrent->setEstimate( pKeyFrame->matTransformationLEFTtoWORLD );
        pVertexPoseCurrent->setId( uNextAvailableUID );
        cGraph.addVertex( pVertexPoseCurrent );

        //ds get previous vertex to link with current one
        g2o::VertexSE3* pVertexPosePrevious = dynamic_cast< g2o::VertexSE3* >( cGraph.vertices( ).find( uNextAvailableUID-1 )->second );
        ++uNextAvailableUID;

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
        pEdgeLinearAcceleration->setMeasurement( pKeyFrame->vecLinearAccelerationNormalized );
        pEdgeLinearAcceleration->setParameterId( 0, EG2OParameterID::eOFFSET_IMUtoLEFT );
        const double arrInformationMatrixLinearAcceleration[9] = { 1000, 0, 0, 0, 1000, 0, 0, 0, 1000 };
        pEdgeLinearAcceleration->setInformation( g2o::Matrix3D( arrInformationMatrixLinearAcceleration ) );
        cGraph.addEdge( pEdgeLinearAcceleration );

        //ds check visible landmarks and add the edges for the current pose
        for( const CMeasurementLandmark* pMeasurementLandmark: *pKeyFrame->vecLandmarkMeasurements )
        {
            //ds find the corresponding landmark
            const g2o::HyperGraph::VertexIDMap::iterator itLandmark( cGraph.vertices( ).find( pMeasurementLandmark->uID ) );

            //ds if found
            if( itLandmark != cGraph.vertices( ).end( ) )
            {
                //ds get the respective feature vertex (this only works if the landmark id's are preserved in the optimizer)
                g2o::VertexPointXYZ* pVertexLandmark = dynamic_cast< g2o::VertexPointXYZ* >( itLandmark->second );

                //ds get key values
                const double dDepthMeters = pMeasurementLandmark->vecPointXYZLEFT.z( );

                assert( 0.0 < dDepthMeters );

                //ds projected depth (LEFT camera)
                cGraph.addEdge( _getEdgeUVDepth( pVertexPoseCurrent, pVertexLandmark, EG2OParameterID::eCAMERA_LEFT, pMeasurementLandmark ) );
                ++uMeasurementsStoredDepth;
            }
        }
    }

    std::printf( "<CBridgeG2O>(saveUVDepth) stored measurements EdgeSE3PointXYZDepth: %lu\n", uMeasurementsStoredDepth );

    cGraph.save( strOutfile.c_str( ) );
}

void CBridgeG2O::saveUVDisparity( const std::string& p_strOutfile,
                                  const CStereoCamera& p_cStereoCamera,
                                  const std::vector< CLandmark* >& p_vecLandmarks,
                                  const std::vector< CKeyFrame >& p_vecKeyFrames )
{
    //ds append postfix
    const std::string strOutfile( p_strOutfile+"_UVDisparity.g2o" );

    //ds validate input
    if( p_vecLandmarks.empty( ) )
    {
        std::printf( "<CBridgeG2O>(saveUVDisparity) received empty landmarks vector, call ignored\n" );
        return;
    }
    if( p_vecKeyFrames.empty( ) )
    {
        std::printf( "<CBridgeG2O>(saveUVDisparity) received empty measurements vector, call ignored\n" );
        return;
    }

    //ds allocate an optimizer
    g2o::OptimizableGraph cGraph;

    //ds set camera parameters
    g2o::ParameterCamera* pCameraParametersLEFT = new g2o::ParameterCamera( );
    pCameraParametersLEFT->setKcam( p_cStereoCamera.m_pCameraLEFT->m_dFxP, p_cStereoCamera.m_pCameraLEFT->m_dFyP, p_cStereoCamera.m_pCameraLEFT->m_dCxP, p_cStereoCamera.m_pCameraLEFT->m_dCyP );
    pCameraParametersLEFT->setId( EG2OParameterID::eCAMERA_LEFT );
    cGraph.addParameter( pCameraParametersLEFT );

    //ds imu offset (as IMU to LEFT)
    g2o::ParameterSE3Offset* pOffsetIMUtoLEFT = new g2o::ParameterSE3Offset( );
    pOffsetIMUtoLEFT->setOffset( p_cStereoCamera.m_pCameraLEFT->m_matTransformationIMUtoLEFT );
    pOffsetIMUtoLEFT->setId( EG2OParameterID::eOFFSET_IMUtoLEFT );
    cGraph.addParameter( pOffsetIMUtoLEFT );

    //ds counter
    uint64_t uDroppedLandmarks( 0 );

    //ds add landmarks
    for( const CLandmark* pLandmark: p_vecLandmarks )
    {
        //ds check if calibration criteria is met
        if( CBridgeG2O::isOptimized( pLandmark ) && CBridgeG2O::isKeyFramed( pLandmark ) )
        {
            //ds set landmark vertex
            g2o::VertexPointXYZ* pVertexLandmark = new g2o::VertexPointXYZ( );
            pVertexLandmark->setEstimate( pLandmark->vecPointXYZOptimized );
            pVertexLandmark->setId( pLandmark->uID );

            //ds add vertex to optimizer
            cGraph.addVertex( pVertexLandmark );
        }
        else
        {
            //std::printf( "<CBridgeG2O>(savesolveAndOptimizeG2O) dropped landmark [%lu]\n", pLandmark->uID );
            ++uDroppedLandmarks;
        }
    }

    std::printf( "<CBridgeG2O>(saveUVDisparity) dropped landmarks: %lu/%lu (%1.2f)\n", uDroppedLandmarks, p_vecLandmarks.size( ), static_cast< double >( uDroppedLandmarks )/p_vecLandmarks.size( ) );

    //ds g2o element identifier
    uint64_t uNextAvailableUID( p_vecLandmarks.back( )->uID+1 );

    //ds add the first pose separately (there's no edge to the "previous" pose)
    g2o::VertexSE3 *pVertexPoseInitial = new g2o::VertexSE3( );
    pVertexPoseInitial->setEstimate( p_vecKeyFrames.front( ).matTransformationLEFTtoWORLD );
    pVertexPoseInitial->setId( uNextAvailableUID );
    pVertexPoseInitial->setFixed( true );
    cGraph.addVertex( pVertexPoseInitial );
    ++uNextAvailableUID;

    //ds always save acceleration data
    g2o::EdgeSE3LinearAcceleration* pEdgeLinearAccelerationInitial = new g2o::EdgeSE3LinearAcceleration( );

    pEdgeLinearAccelerationInitial->setVertex( 0, pVertexPoseInitial );
    pEdgeLinearAccelerationInitial->setMeasurement( p_vecKeyFrames.front( ).vecLinearAccelerationNormalized );
    pEdgeLinearAccelerationInitial->setParameterId( 0, EG2OParameterID::eOFFSET_IMUtoLEFT );
    const double arrInformationMatrixLinearAcceleration[9] = { 1000, 0, 0, 0, 1000, 0, 0, 0, 1000 };
    pEdgeLinearAccelerationInitial->setInformation( g2o::Matrix3D( arrInformationMatrixLinearAcceleration ) );
    cGraph.addEdge( pEdgeLinearAccelerationInitial );

    uint64_t uMeasurementsStoredDisparity = 0;

    //ds loop over the camera vertices vector (skipping the first one that we added before)
    for( std::vector< CKeyFrame >::const_iterator pKeyFrame = p_vecKeyFrames.begin( )+1; pKeyFrame != p_vecKeyFrames.end( ); ++pKeyFrame )
    {
        //ds add current camera pose
        g2o::VertexSE3* pVertexPoseCurrent = new g2o::VertexSE3( );
        pVertexPoseCurrent->setEstimate( pKeyFrame->matTransformationLEFTtoWORLD );
        pVertexPoseCurrent->setId( uNextAvailableUID );
        cGraph.addVertex( pVertexPoseCurrent );

        //ds get previous vertex to link with current one
        g2o::VertexSE3* pVertexPosePrevious = dynamic_cast< g2o::VertexSE3* >( cGraph.vertices( ).find( uNextAvailableUID-1 )->second );
        ++uNextAvailableUID;

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
        pEdgeLinearAcceleration->setMeasurement( pKeyFrame->vecLinearAccelerationNormalized );
        pEdgeLinearAcceleration->setParameterId( 0, EG2OParameterID::eOFFSET_IMUtoLEFT );
        const double arrInformationMatrixLinearAcceleration[9] = { 1000, 0, 0, 0, 1000, 0, 0, 0, 1000 };
        pEdgeLinearAcceleration->setInformation( g2o::Matrix3D( arrInformationMatrixLinearAcceleration ) );
        cGraph.addEdge( pEdgeLinearAcceleration );

        //ds check visible landmarks and add the edges for the current pose
        for( const CMeasurementLandmark* pMeasurementLandmark: *pKeyFrame->vecLandmarkMeasurements )
        {
            //ds find the corresponding landmark
            const g2o::HyperGraph::VertexIDMap::iterator itLandmark( cGraph.vertices( ).find( pMeasurementLandmark->uID ) );

            //ds if found
            if( itLandmark != cGraph.vertices( ).end( ) )
            {
                //ds get the respective feature vertex (this only works if the landmark id's are preserved in the optimizer)
                g2o::VertexPointXYZ* pVertexLandmark = dynamic_cast< g2o::VertexPointXYZ* >( itLandmark->second );

                //ds get key values
                const double dDisparity   = pMeasurementLandmark->ptUVLEFT.x-pMeasurementLandmark->ptUVRIGHT.x;
                assert( 0.0 != dDisparity );

                //ds disparity (LEFT camera)
                cGraph.addEdge( _getEdgeUVDisparity( pVertexPoseCurrent, pVertexLandmark, EG2OParameterID::eCAMERA_LEFT, dDisparity, pMeasurementLandmark, p_cStereoCamera.m_pCameraLEFT->m_dFxP, p_cStereoCamera.m_dBaselineMeters ) );
                ++uMeasurementsStoredDisparity;
            }
        }
    }

    std::printf( "<CBridgeG2O>(saveUVDisparity) stored measurements EdgeSE3PointXYZDisparity: %lu\n", uMeasurementsStoredDisparity );

    cGraph.save( strOutfile.c_str( ) );
}

void CBridgeG2O::saveUVDepthOrDisparity( const std::string& p_strOutfile,
                                          const CStereoCamera& p_cStereoCamera,
                                          const std::vector< CLandmark* >& p_vecLandmarks,
                                          const std::vector< CKeyFrame >& p_vecKeyFrames )
{
    //ds append postfix
    const std::string strOutfile( p_strOutfile+"_UVDepthOrDisparity.g2o" );

    //ds validate input
    if( p_vecLandmarks.empty( ) )
    {
        std::printf( "<CBridgeG2O>(saveUVDepthOrDisparity) received empty landmarks vector, call ignored\n" );
        return;
    }
    if( p_vecKeyFrames.empty( ) )
    {
        std::printf( "<CBridgeG2O>(saveUVDepthOrDisparity) received empty measurements vector, call ignored\n" );
        return;
    }

    //ds allocate an optimizer
    g2o::OptimizableGraph cGraph;

    //ds set camera parameters
    g2o::ParameterCamera* pCameraParametersLEFT = new g2o::ParameterCamera( );
    pCameraParametersLEFT->setKcam( p_cStereoCamera.m_pCameraLEFT->m_dFxP, p_cStereoCamera.m_pCameraLEFT->m_dFyP, p_cStereoCamera.m_pCameraLEFT->m_dCxP, p_cStereoCamera.m_pCameraLEFT->m_dCyP );
    pCameraParametersLEFT->setId( EG2OParameterID::eCAMERA_LEFT );
    cGraph.addParameter( pCameraParametersLEFT );

    //ds imu offset (as IMU to LEFT)
    g2o::ParameterSE3Offset* pOffsetIMUtoLEFT = new g2o::ParameterSE3Offset( );
    pOffsetIMUtoLEFT->setOffset( p_cStereoCamera.m_pCameraLEFT->m_matTransformationIMUtoLEFT );
    pOffsetIMUtoLEFT->setId( EG2OParameterID::eOFFSET_IMUtoLEFT );
    cGraph.addParameter( pOffsetIMUtoLEFT );

    //ds counter
    uint64_t uDroppedLandmarks( 0 );

    //ds add landmarks
    for( const CLandmark* pLandmark: p_vecLandmarks )
    {
        //ds check if calibration criteria is met
        if( CBridgeG2O::isOptimized( pLandmark ) && CBridgeG2O::isKeyFramed( pLandmark ) )
        {
            //ds set landmark vertex
            g2o::VertexPointXYZ* pVertexLandmark = new g2o::VertexPointXYZ( );
            pVertexLandmark->setEstimate( pLandmark->vecPointXYZOptimized );
            pVertexLandmark->setId( pLandmark->uID );

            //ds add vertex to optimizer
            cGraph.addVertex( pVertexLandmark );
        }
        else
        {
            //std::printf( "<CBridgeG2O>(savesolveAndOptimizeG2O) dropped landmark [%lu]\n", pLandmark->uID );
            ++uDroppedLandmarks;
        }
    }

    std::printf( "<CBridgeG2O>(saveUVDepthOrDisparity) dropped landmarks: %lu/%lu (%1.2f)\n", uDroppedLandmarks, p_vecLandmarks.size( ), static_cast< double >( uDroppedLandmarks )/p_vecLandmarks.size( ) );

    //ds g2o element identifier
    uint64_t uNextAvailableUID( p_vecLandmarks.back( )->uID+1 );

    //ds add the first pose separately (there's no edge to the "previous" pose)
    g2o::VertexSE3 *pVertexPoseInitial = new g2o::VertexSE3( );
    pVertexPoseInitial->setEstimate( p_vecKeyFrames.front( ).matTransformationLEFTtoWORLD );
    pVertexPoseInitial->setId( uNextAvailableUID );
    pVertexPoseInitial->setFixed( true );
    cGraph.addVertex( pVertexPoseInitial );
    ++uNextAvailableUID;

    //ds always save acceleration data
    g2o::EdgeSE3LinearAcceleration* pEdgeLinearAccelerationInitial = new g2o::EdgeSE3LinearAcceleration( );

    pEdgeLinearAccelerationInitial->setVertex( 0, pVertexPoseInitial );
    pEdgeLinearAccelerationInitial->setMeasurement( p_vecKeyFrames.front( ).vecLinearAccelerationNormalized );
    pEdgeLinearAccelerationInitial->setParameterId( 0, EG2OParameterID::eOFFSET_IMUtoLEFT );
    const double arrInformationMatrixLinearAcceleration[9] = { 1000, 0, 0, 0, 1000, 0, 0, 0, 1000 };
    pEdgeLinearAccelerationInitial->setInformation( g2o::Matrix3D( arrInformationMatrixLinearAcceleration ) );
    cGraph.addEdge( pEdgeLinearAccelerationInitial );

    uint64_t uMeasurementsStoredDepth     = 0;
    uint64_t uMeasurementsStoredDisparity = 0;

    //ds loop over the camera vertices vector (skipping the first one that we added before)
    for( std::vector< CKeyFrame >::const_iterator pKeyFrame = p_vecKeyFrames.begin( )+1; pKeyFrame != p_vecKeyFrames.end( ); ++pKeyFrame )
    {
        //ds add current camera pose
        g2o::VertexSE3* pVertexPoseCurrent = new g2o::VertexSE3( );
        pVertexPoseCurrent->setEstimate( pKeyFrame->matTransformationLEFTtoWORLD );
        pVertexPoseCurrent->setId( uNextAvailableUID );
        cGraph.addVertex( pVertexPoseCurrent );

        //ds get previous vertex to link with current one
        g2o::VertexSE3* pVertexPosePrevious = dynamic_cast< g2o::VertexSE3* >( cGraph.vertices( ).find( uNextAvailableUID-1 )->second );
        ++uNextAvailableUID;

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
        pEdgeLinearAcceleration->setMeasurement( pKeyFrame->vecLinearAccelerationNormalized );
        pEdgeLinearAcceleration->setParameterId( 0, EG2OParameterID::eOFFSET_IMUtoLEFT );
        const double arrInformationMatrixLinearAcceleration[9] = { 1000, 0, 0, 0, 1000, 0, 0, 0, 1000 };
        pEdgeLinearAcceleration->setInformation( g2o::Matrix3D( arrInformationMatrixLinearAcceleration ) );
        cGraph.addEdge( pEdgeLinearAcceleration );

        //ds check visible landmarks and add the edges for the current pose
        for( const CMeasurementLandmark* pMeasurementLandmark: *pKeyFrame->vecLandmarkMeasurements )
        {
            //ds find the corresponding landmark based on its ID
            const g2o::HyperGraph::VertexIDMap::iterator itLandmark( cGraph.vertices( ).find( pMeasurementLandmark->uID ) );

            //ds if found
            if( itLandmark != cGraph.vertices( ).end( ) )
            {
                //ds get the respective feature vertex (this only works if the landmark id's are preserved in the optimizer)
                g2o::VertexPointXYZ* pVertexLandmark = dynamic_cast< g2o::VertexPointXYZ* >( itLandmark->second );

                //ds get optimized landmark into current pose
                const CPoint3DInCameraFrame vecPointOptimized( pVertexPoseCurrent->estimate( ).inverse( )*pVertexLandmark->estimate( ) );

                //ds get key values
                const double dDisparity   = pMeasurementLandmark->ptUVLEFT.x-pMeasurementLandmark->ptUVRIGHT.x;
                const double dDepthMeters = pMeasurementLandmark->vecPointXYZLEFT.z( );

                assert( 0.0 < dDepthMeters );
                assert( 0.0 != dDisparity );

                /*const double dDepthChange = std::fabs( vecPointOptimized.z( )-dDepthMeters );
                if( CBridgeG2O::m_dMaximumReliableDepthForUVDepth > dDepthMeters && 1.0 < dDepthChange )
                {
                    std::printf( "<CBridgeG2O>(saveUVDepthOrDisparity) depth change: %f\n", dDepthChange );
                }*/

                //ds minimum quality to produce a reliable depth estimate
                if( CBridgeG2O::m_dMaximumReliableDepthForUVDepth > vecPointOptimized.z( ) )
                {
                    //ds projected depth (LEFT camera)
                    cGraph.addEdge( _getEdgeUVDepth( pVertexPoseCurrent, pVertexLandmark, EG2OParameterID::eCAMERA_LEFT, pMeasurementLandmark ) );
                    ++uMeasurementsStoredDepth;
                }
                else
                {
                    //ds disparity (LEFT camera)
                    cGraph.addEdge( _getEdgeUVDisparity( pVertexPoseCurrent, pVertexLandmark, EG2OParameterID::eCAMERA_LEFT, dDisparity, pMeasurementLandmark, p_cStereoCamera.m_pCameraLEFT->m_dFxP, p_cStereoCamera.m_dBaselineMeters ) );
                    ++uMeasurementsStoredDisparity;

                    //std::printf( "<CBridgeG2O>(saveUVDepthOrDisparity) landmark [%lu] imprecision: %f - stored disparity measurement (%ip) instead of depth (%fm)\n", pMeasurementLandmark->uID, dImprecision, iDisparity, dDepthMeters );
                }
            }
        }
    }

    std::printf( "<CBridgeG2O>(saveUVDepthOrDisparity) stored measurements EdgeSE3PointXYZDepth: %lu\n", uMeasurementsStoredDepth );
    std::printf( "<CBridgeG2O>(saveUVDepthOrDisparity) stored measurements EdgeSE3PointXYZDisparity: %lu\n", uMeasurementsStoredDisparity );

    cGraph.save( strOutfile.c_str( ) );
}

void CBridgeG2O::saveCOMBO( const std::string& p_strOutfile,
                            const CStereoCamera& p_cStereoCamera,
                            const std::vector< CLandmark* >& p_vecLandmarks,
                            const std::vector< CKeyFrame >& p_vecKeyFrames )
{
    //ds append postfix
    const std::string strOutfile( p_strOutfile+"_COMBO.g2o" );

    //ds validate input
    if( p_vecLandmarks.empty( ) )
    {
        std::printf( "<CBridgeG2O>(saveCOMBO) received empty landmarks vector, call ignored\n" );
        return;
    }
    if( p_vecKeyFrames.empty( ) )
    {
        std::printf( "<CBridgeG2O>(saveCOMBO) received empty measurements vector, call ignored\n" );
        return;
    }

    //ds allocate an optimizer
    g2o::OptimizableGraph cGraph;

    //ds set world
    g2o::ParameterSE3Offset* pOffsetWorld = new g2o::ParameterSE3Offset( );
    pOffsetWorld->setOffset( Eigen::Isometry3d::Identity( ) );
    pOffsetWorld->setId( EG2OParameterID::eWORLD );
    cGraph.addParameter( pOffsetWorld );

    //ds set camera parameters
    g2o::ParameterCamera* pCameraParametersLEFT = new g2o::ParameterCamera( );
    pCameraParametersLEFT->setKcam( p_cStereoCamera.m_pCameraLEFT->m_dFxP, p_cStereoCamera.m_pCameraLEFT->m_dFyP, p_cStereoCamera.m_pCameraLEFT->m_dCxP, p_cStereoCamera.m_pCameraLEFT->m_dCyP );
    pCameraParametersLEFT->setId( EG2OParameterID::eCAMERA_LEFT );
    cGraph.addParameter( pCameraParametersLEFT );

    //ds imu offset (as IMU to LEFT)
    g2o::ParameterSE3Offset* pOffsetIMUtoLEFT = new g2o::ParameterSE3Offset( );
    pOffsetIMUtoLEFT->setOffset( p_cStereoCamera.m_pCameraLEFT->m_matTransformationIMUtoLEFT );
    pOffsetIMUtoLEFT->setId( EG2OParameterID::eOFFSET_IMUtoLEFT );
    cGraph.addParameter( pOffsetIMUtoLEFT );

    //ds counter
    uint64_t uDroppedLandmarks = 0;

    //ds add landmarks
    for( const CLandmark* pLandmark: p_vecLandmarks )
    {
        //ds check if optimiziation criterias are met
        if( CBridgeG2O::isOptimized( pLandmark ) && CBridgeG2O::isKeyFramed( pLandmark ) )
        {
            //ds set landmark vertex
            g2o::VertexPointXYZ* pVertexLandmark = new g2o::VertexPointXYZ( );
            pVertexLandmark->setEstimate( pLandmark->vecPointXYZOptimized );
            pVertexLandmark->setId( pLandmark->uID );

            //ds add vertex to optimizer
            cGraph.addVertex( pVertexLandmark );
        }
        else
        {
            ++uDroppedLandmarks;
        }
    }

    std::printf( "<CBridgeG2O>(saveCOMBO) dropped landmarks: %lu/%lu (%1.2f)\n", uDroppedLandmarks, p_vecLandmarks.size( ), static_cast< double >( uDroppedLandmarks )/p_vecLandmarks.size( ) );

    //ds g2o element identifier
    uint64_t uNextAvailableUID( p_vecLandmarks.back( )->uID+1 );

    //ds add the first pose separately (there's no edge to the "previous" pose)
    g2o::VertexSE3 *pVertexPoseInitial = new g2o::VertexSE3( );
    pVertexPoseInitial->setEstimate( p_vecKeyFrames.front( ).matTransformationLEFTtoWORLD );
    pVertexPoseInitial->setId( uNextAvailableUID );
    pVertexPoseInitial->setFixed( true );
    cGraph.addVertex( pVertexPoseInitial );
    ++uNextAvailableUID;

    //ds always save acceleration data
    g2o::EdgeSE3LinearAcceleration* pEdgeLinearAccelerationInitial = new g2o::EdgeSE3LinearAcceleration( );

    pEdgeLinearAccelerationInitial->setVertex( 0, pVertexPoseInitial );
    pEdgeLinearAccelerationInitial->setMeasurement( p_vecKeyFrames.front( ).vecLinearAccelerationNormalized );
    pEdgeLinearAccelerationInitial->setParameterId( 0, EG2OParameterID::eOFFSET_IMUtoLEFT );
    const double arrInformationMatrixLinearAcceleration[9] = { 1000, 0, 0, 0, 1000, 0, 0, 0, 1000 };
    pEdgeLinearAccelerationInitial->setInformation( g2o::Matrix3D( arrInformationMatrixLinearAcceleration ) );
    cGraph.addEdge( pEdgeLinearAccelerationInitial );

    uint64_t uMeasurementsStoredPointXYZ          = 0;
    uint64_t uMeasurementsStoredUVDepth           = 0;
    uint64_t uMeasurementsStoredUVDisparity       = 0;

    //ds loop over the camera vertices vector (skipping the first one that we added before)
    for( std::vector< CKeyFrame >::const_iterator pKeyFrame = p_vecKeyFrames.begin( )+1; pKeyFrame != p_vecKeyFrames.end( ); ++pKeyFrame )
    {
        //ds add current camera pose
        g2o::VertexSE3* pVertexPoseCurrent = new g2o::VertexSE3( );
        pVertexPoseCurrent->setEstimate( pKeyFrame->matTransformationLEFTtoWORLD );
        pVertexPoseCurrent->setId( uNextAvailableUID );
        cGraph.addVertex( pVertexPoseCurrent );

        //ds get previous vertex to link with current one
        g2o::VertexSE3* pVertexPosePrevious = dynamic_cast< g2o::VertexSE3* >( cGraph.vertices( ).find( uNextAvailableUID-1 )->second );
        ++uNextAvailableUID;

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
        pEdgeLinearAcceleration->setMeasurement( pKeyFrame->vecLinearAccelerationNormalized );
        pEdgeLinearAcceleration->setParameterId( 0, EG2OParameterID::eOFFSET_IMUtoLEFT );
        const double arrInformationMatrixLinearAcceleration[9] = { 1000, 0, 0, 0, 1000, 0, 0, 0, 1000 };
        pEdgeLinearAcceleration->setInformation( g2o::Matrix3D( arrInformationMatrixLinearAcceleration ) );
        cGraph.addEdge( pEdgeLinearAcceleration );

        //ds check visible landmarks and add the edges for the current pose
        for( const CMeasurementLandmark* pMeasurementLandmark: *pKeyFrame->vecLandmarkMeasurements )
        {
            //ds find the corresponding landmark
            const g2o::HyperGraph::VertexIDMap::iterator itLandmark( cGraph.vertices( ).find( pMeasurementLandmark->uID ) );

            //ds if found
            if( itLandmark != cGraph.vertices( ).end( ) )
            {
                //ds get the respective feature vertex (this only works if the landmark id's are preserved in the optimizer)
                g2o::VertexPointXYZ* pVertexLandmark = dynamic_cast< g2o::VertexPointXYZ* >( itLandmark->second );

                //ds get optimized landmark into current pose
                const CPoint3DInCameraFrame vecPointOptimized( pVertexPoseCurrent->estimate( ).inverse( )*pVertexLandmark->estimate( ) );

                //ds current depth
                //const double dDepthMeters = pMeasurementLandmark->vecPointXYZLEFT.z( );

                /*ds check depth change
                const double dDepthChange = std::fabs( vecPointOptimized.z( )-dDepthMeters );
                if( CBridgeG2O::m_dMaximumReliableDepthForUVDepth > dDepthMeters && 1.0 < dDepthChange )
                {
                    std::printf( "<CBridgeG2O>(saveCOMBO) depth change: %f\n", dDepthChange );
                }*/

                //ds maximum depth to produce a reliable XYZ estimate
                if( CBridgeG2O::m_dMaximumReliableDepthForPointXYZ > vecPointOptimized.z( ) )
                {
                    cGraph.addEdge( _getEdgePointXYZ( pVertexPoseCurrent, pVertexLandmark, EG2OParameterID::eWORLD, pMeasurementLandmark->vecPointXYZLEFT ) );
                    ++uMeasurementsStoredPointXYZ;
                }

                //ds still good enough for uvdepth
                else if( CBridgeG2O::m_dMaximumReliableDepthForUVDepth > vecPointOptimized.z( ) )
                {
                    //ds projected depth (LEFT camera)
                    cGraph.addEdge( _getEdgeUVDepth( pVertexPoseCurrent, pVertexLandmark, EG2OParameterID::eCAMERA_LEFT, pMeasurementLandmark ) );
                    ++uMeasurementsStoredUVDepth;
                }

                //ds go with disparity
                else
                {
                    //ds current disparity
                    const double dDisparity = pMeasurementLandmark->ptUVLEFT.x-pMeasurementLandmark->ptUVRIGHT.x;

                    //ds disparity (LEFT camera)
                    cGraph.addEdge( _getEdgeUVDisparity( pVertexPoseCurrent, pVertexLandmark, EG2OParameterID::eCAMERA_LEFT, dDisparity, pMeasurementLandmark, p_cStereoCamera.m_pCameraLEFT->m_dFxP, p_cStereoCamera.m_dBaselineMeters ) );
                    ++uMeasurementsStoredUVDisparity;
                }
            }
        }
    }

    std::printf( "<CBridgeG2O>(saveCOMBO) stored measurements EdgeSE3PointXYZ: %lu\n", uMeasurementsStoredPointXYZ );
    std::printf( "<CBridgeG2O>(saveCOMBO) stored measurements EdgeSE3PointXYZDepth: %lu\n", uMeasurementsStoredUVDepth );
    std::printf( "<CBridgeG2O>(saveCOMBO) stored measurements EdgeSE3PointXYZDisparity: %lu\n", uMeasurementsStoredUVDisparity );

    cGraph.save( strOutfile.c_str( ) );

    //ds optimize!
    //cOptimizer.initializeOptimization( );
    //cOptimizer.computeActiveErrors( );
    //cOptimizer.optimize( 1 );
}

const bool CBridgeG2O::isOptimized( const CLandmark* p_pLandmark )
{
    //ds criteria
    return ( CBridgeG2O::m_uMinimumOptimizations < p_pLandmark->uOptimizationsSuccessful ) && ( p_pLandmark->uOptimizationsSuccessful*CBridgeG2O::m_dMaximumErrorPerOptimization > p_pLandmark->dCurrentAverageSquaredError );
}

const bool CBridgeG2O::isKeyFramed( const CLandmark* p_pLandmark )
{
    //ds criteria
    return ( CBridgeG2O::m_uMinimumKeyFramePresences < p_pLandmark->uNumberOfKeyFramePresences );
}

g2o::EdgeSE3PointXYZ* CBridgeG2O::_getEdgePointXYZ( g2o::VertexSE3* p_pVertexPose,
                                               g2o::VertexPointXYZ* p_pVertexLandmark,
                                               const EG2OParameterID& p_eParameterIDOriginWORLD,
                                               const CPoint3DInWorldFrame& p_vecPointXYZ )
{
    g2o::EdgeSE3PointXYZ* pEdgePointXYZ( new g2o::EdgeSE3PointXYZ( ) );

    //ds triangulated 3d point (uncalibrated)
    pEdgePointXYZ->setVertex( 0, p_pVertexPose );
    pEdgePointXYZ->setVertex( 1, p_pVertexLandmark );
    pEdgePointXYZ->setMeasurement( p_vecPointXYZ );
    pEdgePointXYZ->setParameterId( 0, p_eParameterIDOriginWORLD );

    //ds the closer to the camera the point is the more meaningful is the measurement
    //const double dInformationStrengthXYZ( m_dMaximumReliableDepth/( 1+pMeasurementLandmark->vecPointXYZ.z( ) ) );
    const double arrInformationMatrixXYZ[9] = { 1000, 0, 0, 0, 1000, 0, 0, 0, 1000 };
    pEdgePointXYZ->setInformation( g2o::Matrix3D( arrInformationMatrixXYZ ) );

    return pEdgePointXYZ;
}

g2o::EdgeSE3PointXYZDepth* CBridgeG2O::_getEdgeUVDepth( g2o::VertexSE3* p_pVertexPose,
                                                               g2o::VertexPointXYZ* p_pVertexLandmark,
                                                               const EG2OParameterID& p_eParameterIDCamera,
                                                               const CMeasurementLandmark* p_pMeasurement )
{
    //ds projected depth
    g2o::EdgeSE3PointXYZDepth* pEdgeProjectedDepth( new g2o::EdgeSE3PointXYZDepth( ) );

    pEdgeProjectedDepth->setVertex( 0, p_pVertexPose );
    pEdgeProjectedDepth->setVertex( 1, p_pVertexLandmark );
    pEdgeProjectedDepth->setMeasurement( g2o::Vector3D( p_pMeasurement->ptUVLEFT.x, p_pMeasurement->ptUVLEFT.y, p_pMeasurement->vecPointXYZLEFT.z( ) ) );
    pEdgeProjectedDepth->setParameterId( 0, p_eParameterIDCamera );

    //ds information matrix
    //const double dInformationQualityDepth( m_dMaximumReliableDepth/dDepthMeters );
    const double arrInformationMatrixDepth[9] = { 1, 0, 0, 0, 1, 0, 0, 0, 1000 };
    pEdgeProjectedDepth->setInformation( g2o::Matrix3D( arrInformationMatrixDepth ) );

    return pEdgeProjectedDepth;
}

g2o::EdgeSE3PointXYZDisparity* CBridgeG2O::_getEdgeUVDisparity( g2o::VertexSE3* p_pVertexPose,
                                                           g2o::VertexPointXYZ* p_pVertexLandmark,
                                                           const EG2OParameterID& p_eParameterIDCamera,
                                                           const double& p_dDisparityPixels,
                                                           const CMeasurementLandmark* p_pMeasurement,
                                                           const double& p_dFxPixels,
                                                           const double& p_dBaselineMeters )
{
    //ds disparity
    g2o::EdgeSE3PointXYZDisparity* pEdgeDisparity( new g2o::EdgeSE3PointXYZDisparity( ) );

    const double dDisparityNormalized( p_dDisparityPixels/( p_dFxPixels*p_dBaselineMeters ) );
    pEdgeDisparity->setVertex( 0, p_pVertexPose );
    pEdgeDisparity->setVertex( 1, p_pVertexLandmark );
    pEdgeDisparity->setMeasurement( g2o::Vector3D( p_pMeasurement->ptUVLEFT.x, p_pMeasurement->ptUVLEFT.y, dDisparityNormalized ) );
    pEdgeDisparity->setParameterId( 0, p_eParameterIDCamera );

    //ds information matrix
    //const double dInformationQualityDisparity( std::abs( dDisparity )+1.0 );
    const double arrInformationMatrixDisparity[9] = { 1, 0, 0, 0, 1, 0, 0, 0, 1000 };
    pEdgeDisparity->setInformation( g2o::Matrix3D( arrInformationMatrixDisparity ) );

    return pEdgeDisparity;
}
