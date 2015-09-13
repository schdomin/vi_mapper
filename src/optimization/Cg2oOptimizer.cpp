#include "Cg2oOptimizer.h"

#include <dirent.h>
#include <stdio.h>

#include "g2o/core/block_solver.h"
#include "g2o/core/hyper_graph.h"
#include "g2o/core/solver.h"
#include "g2o/core/optimization_algorithm_levenberg.h"
#include "g2o/core/optimization_algorithm_gauss_newton.h"
#include "g2o/solvers/cholmod/linear_solver_cholmod.h"
#include "configuration/CConfigurationCamera.h"
#include "exceptions/CExceptionLogfileTree.h"

Cg2oOptimizer::Cg2oOptimizer( const std::shared_ptr< CStereoCamera > p_pCameraSTEREO,
                              const std::shared_ptr< std::vector< CLandmark* > > p_vecLandmarks,
                              const std::shared_ptr< std::vector< CKeyFrame* > > p_vecKeyFrames,
                              const Eigen::Isometry3d& p_matTransformationLEFTtoWORLDInitial ): m_pCameraSTEREO( p_pCameraSTEREO ),
                                                                                       m_vecLandmarks( p_vecLandmarks ),
                                                                                       m_vecKeyFrames( p_vecKeyFrames )
{
    m_vecLandmarksInGraph.clear( );
    m_vecKeyFramesInGraph.clear( );

    //ds configure the solver (var_cholmod)
    m_cOptimizerSparse.setVerbose( true );
    g2o::BlockSolverX::LinearSolverType* pLinearSolver( new g2o::LinearSolverCholmod< g2o::BlockSolverX::PoseMatrixType >( ) );
    g2o::BlockSolverX* pSolver( new g2o::BlockSolverX( pLinearSolver ) );
    //g2o::OptimizationAlgorithmGaussNewton* pAlgorithm( new g2o::OptimizationAlgorithmGaussNewton( pSolver ) );
    g2o::OptimizationAlgorithmLevenberg* pAlgorithm( new g2o::OptimizationAlgorithmLevenberg( pSolver ) );
    m_cOptimizerSparse.setAlgorithm( pAlgorithm );

    //ds set world
    g2o::ParameterSE3Offset* pOffsetWorld = new g2o::ParameterSE3Offset( );
    pOffsetWorld->setOffset( Eigen::Isometry3d::Identity( ) );
    pOffsetWorld->setId( EG2OParameterID::eWORLD );
    m_cOptimizerSparse.addParameter( pOffsetWorld );

    //ds set camera parameters
    g2o::ParameterCamera* pCameraParametersLEFT = new g2o::ParameterCamera( );
    pCameraParametersLEFT->setKcam( m_pCameraSTEREO->m_pCameraLEFT->m_dFxP, m_pCameraSTEREO->m_pCameraLEFT->m_dFyP, m_pCameraSTEREO->m_pCameraLEFT->m_dCxP, m_pCameraSTEREO->m_pCameraLEFT->m_dCyP );
    pCameraParametersLEFT->setId( EG2OParameterID::eCAMERA_LEFT );
    m_cOptimizerSparse.addParameter( pCameraParametersLEFT );
    g2o::ParameterCamera* pCameraParametersRIGHT = new g2o::ParameterCamera( );
    pCameraParametersRIGHT->setKcam( m_pCameraSTEREO->m_pCameraRIGHT->m_dFxP, m_pCameraSTEREO->m_pCameraRIGHT->m_dFyP, m_pCameraSTEREO->m_pCameraRIGHT->m_dCxP, m_pCameraSTEREO->m_pCameraRIGHT->m_dCyP );
    pCameraParametersRIGHT->setId( EG2OParameterID::eCAMERA_RIGHT );
    m_cOptimizerSparse.addParameter( pCameraParametersRIGHT );

    //ds imu offset (as IMU to LEFT)
    g2o::ParameterSE3Offset* pOffsetIMUtoLEFT = new g2o::ParameterSE3Offset( );
    pOffsetIMUtoLEFT->setOffset( m_pCameraSTEREO->m_pCameraLEFT->m_matTransformationIMUtoCAMERA );
    pOffsetIMUtoLEFT->setId( EG2OParameterID::eOFFSET_IMUtoLEFT );
    m_cOptimizerSparse.addParameter( pOffsetIMUtoLEFT );

    //ds add the first pose separately
    g2o::VertexSE3* m_pVertexPoseLAST = new g2o::VertexSE3( );
    m_pVertexPoseLAST->setEstimate( p_matTransformationLEFTtoWORLDInitial );
    m_pVertexPoseLAST->setId( m_uIDShift-1 );
    m_pVertexPoseLAST->setFixed( true );
    m_cOptimizerSparse.addVertex( m_pVertexPoseLAST );

    CLogger::openBox( );
    //std::printf( "<Cg2oOptimizer>(Cg2oOptimizer) configuration: gn_var_cholmod\n" );
    std::printf( "<Cg2oOptimizer>(Cg2oOptimizer) configuration: lm_var_cholmod COMPLETE\n" );
    std::printf( "<Cg2oOptimizer>(Cg2oOptimizer) iterations: %u\n", m_uIterations );
    std::printf( "<Cg2oOptimizer>(Cg2oOptimizer) maximum reliable depth for measurement PointXYZ: %f\n", m_dMaximumReliableDepthForPointXYZ );
    std::printf( "<Cg2oOptimizer>(Cg2oOptimizer) maximum reliable depth for measurement UVDepth: %f\n", m_dMaximumReliableDepthForUVDepth );
    std::printf( "<Cg2oOptimizer>(Cg2oOptimizer) maximum optimization error (landmark): %f\n", m_dMaximumErrorSquared );
    std::printf( "<Cg2oOptimizer>(Cg2oOptimizer) instance allocated\n" );
    CLogger::closeBox( );
}

Cg2oOptimizer::Cg2oOptimizer( const std::shared_ptr< CStereoCamera > p_pCameraSTEREO,
                              const std::shared_ptr< std::vector< CLandmark* > > p_vecLandmarks,
                              const std::shared_ptr< std::vector< CKeyFrame* > > p_vecKeyFrames ): m_pCameraSTEREO( p_pCameraSTEREO ),
                                                                                       m_vecLandmarks( p_vecLandmarks ),
                                                                                       m_vecKeyFrames( p_vecKeyFrames )
{
    m_vecLandmarksInGraph.clear( );
    m_vecKeyFramesInGraph.clear( );

    //ds configure the solver (var_cholmod)
    m_cOptimizerSparse.setVerbose( true );
    g2o::BlockSolverX::LinearSolverType* pLinearSolver( new g2o::LinearSolverCholmod< g2o::BlockSolverX::PoseMatrixType >( ) );
    g2o::BlockSolverX* pSolver( new g2o::BlockSolverX( pLinearSolver ) );
    //g2o::OptimizationAlgorithmGaussNewton* pAlgorithm( new g2o::OptimizationAlgorithmGaussNewton( pSolver ) );
    g2o::OptimizationAlgorithmLevenberg* pAlgorithm( new g2o::OptimizationAlgorithmLevenberg( pSolver ) );
    m_cOptimizerSparse.setAlgorithm( pAlgorithm );

    //ds set world
    g2o::ParameterSE3Offset* pOffsetWorld = new g2o::ParameterSE3Offset( );
    pOffsetWorld->setOffset( Eigen::Isometry3d::Identity( ) );
    pOffsetWorld->setId( EG2OParameterID::eWORLD );
    m_cOptimizerSparse.addParameter( pOffsetWorld );

    //ds set camera parameters
    g2o::ParameterCamera* pCameraParametersLEFT = new g2o::ParameterCamera( );
    pCameraParametersLEFT->setKcam( m_pCameraSTEREO->m_pCameraLEFT->m_dFxP, m_pCameraSTEREO->m_pCameraLEFT->m_dFyP, m_pCameraSTEREO->m_pCameraLEFT->m_dCxP, m_pCameraSTEREO->m_pCameraLEFT->m_dCyP );
    pCameraParametersLEFT->setId( EG2OParameterID::eCAMERA_LEFT );
    m_cOptimizerSparse.addParameter( pCameraParametersLEFT );
    g2o::ParameterCamera* pCameraParametersRIGHT = new g2o::ParameterCamera( );
    pCameraParametersRIGHT->setKcam( m_pCameraSTEREO->m_pCameraRIGHT->m_dFxP, m_pCameraSTEREO->m_pCameraRIGHT->m_dFyP, m_pCameraSTEREO->m_pCameraRIGHT->m_dCxP, m_pCameraSTEREO->m_pCameraRIGHT->m_dCyP );
    pCameraParametersRIGHT->setId( EG2OParameterID::eCAMERA_RIGHT );
    m_cOptimizerSparse.addParameter( pCameraParametersRIGHT );

    //ds imu offset (as IMU to LEFT)
    g2o::ParameterSE3Offset* pOffsetIMUtoLEFT = new g2o::ParameterSE3Offset( );
    pOffsetIMUtoLEFT->setOffset( m_pCameraSTEREO->m_pCameraLEFT->m_matTransformationIMUtoCAMERA );
    pOffsetIMUtoLEFT->setId( EG2OParameterID::eOFFSET_IMUtoLEFT );
    m_cOptimizerSparse.addParameter( pOffsetIMUtoLEFT );

    CLogger::openBox( );
    //std::printf( "<Cg2oOptimizer>(Cg2oOptimizer) configuration: gn_var_cholmod\n" );
    std::printf( "<Cg2oOptimizer>(Cg2oOptimizer) configuration: lm_var_cholmod TAILWISE\n" );
    std::printf( "<Cg2oOptimizer>(Cg2oOptimizer) iterations: %u\n", m_uIterations );
    std::printf( "<Cg2oOptimizer>(Cg2oOptimizer) maximum reliable depth for measurement PointXYZ: %f\n", m_dMaximumReliableDepthForPointXYZ );
    std::printf( "<Cg2oOptimizer>(Cg2oOptimizer) maximum reliable depth for measurement UVDepth: %f\n", m_dMaximumReliableDepthForUVDepth );
    std::printf( "<Cg2oOptimizer>(Cg2oOptimizer) maximum optimization error (landmark): %f\n", m_dMaximumErrorSquared );
    std::printf( "<Cg2oOptimizer>(Cg2oOptimizer) instance allocated\n" );
    CLogger::closeBox( );
}

Cg2oOptimizer::~Cg2oOptimizer( )
{
    //ds nothing to do
}

void Cg2oOptimizer::optimizeTailLoopClosuresOnly( const UIDKeyFrame& p_uIDBeginKeyFrame )
{
    //ds clear nodes and edges
    m_cOptimizerSparse.clear( );

    //ds local optimization chunks: +1 at the end because std::vector[begin,end)
    const std::vector< CKeyFrame* > vecChunkKeyFrames( m_vecKeyFrames->begin( )+p_uIDBeginKeyFrame, m_vecKeyFrames->end( ) );

    //ds add the first pose separately
    g2o::VertexSE3* pVertexPoseFrom = new g2o::VertexSE3( );
    pVertexPoseFrom->setEstimate( vecChunkKeyFrames.front( )->matTransformationLEFTtoWORLD );
    pVertexPoseFrom->setId( vecChunkKeyFrames.front( )->uID+m_uIDShift );
    pVertexPoseFrom->setFixed( true );
    m_cOptimizerSparse.addVertex( pVertexPoseFrom );

    //ds info
    uint32_t uLoopClosures = 0;

    //ds loop over the camera vertices vector (skipping the first one that we added before)
    for( std::vector< CKeyFrame* >::size_type u = 1; u < vecChunkKeyFrames.size( ); ++u )
    {
        //ds buffer current keyframe
        const CKeyFrame* pKeyFrame( vecChunkKeyFrames[u] );

        //ds set the vertex to the graph
        g2o::VertexSE3* pVertexPoseCurrent = _setAndgetPose( pVertexPoseFrom, pKeyFrame );

        //ds check if we got a loop closure for this frame
        if( 0 != pKeyFrame->pLoopClosure )
        {
            //ds set it to the graph
            _setLoopClosure( pVertexPoseCurrent, pKeyFrame );
            ++uLoopClosures;
        }

        //ds update checker window
        //m_cClosureBuffer.updateList( 3 );

        //ds update from
        pVertexPoseFrom = pVertexPoseCurrent;
    }

    //ds save initial situation
    char chBuffer[256];
    std::snprintf( chBuffer, 256, "g2o/local/keyframes_%06lu-%06lu", vecChunkKeyFrames.front( )->uID, vecChunkKeyFrames.back( )->uID );
    std::string strPrefix( chBuffer );
    m_cOptimizerSparse.save( ( strPrefix + ".g2o" ).c_str( ) );

    std::printf( "<Cg2oOptimizer>(optimizeTailLoopClosuresOnly) optimizing [keyframes: %06lu-%06lu] (loop closures: %u)\n",
                 vecChunkKeyFrames.front( )->uID,
                 vecChunkKeyFrames.back( )->uID,
                 uLoopClosures );

    //ds initialize optimization
    m_cOptimizerSparse.initializeOptimization( );

    //ds do ten iterations
    m_cOptimizerSparse.optimize( m_uIterations );

    //ds save optimized situation
    m_cOptimizerSparse.save( ( strPrefix + "_optimized.g2o" ).c_str( ) );

    //ds update trajectory
    _applyOptimization( vecChunkKeyFrames );

    //ds done
    std::printf( "<Cg2oOptimizer>(optimizeTailLoopClosuresOnly) optimized poses: %lu (error final: %f)\n", vecChunkKeyFrames.size( ), m_cOptimizerSparse.activeChi2( ) );
    ++m_uOptimizations;
}

void Cg2oOptimizer::optimizeTail( const UIDKeyFrame& p_uIDBeginKeyFrame )
{
    //ds clear nodes and edges
    m_cOptimizerSparse.clear( );

    //ds local optimization chunks: +1 at the end because std::vector[begin,end)
    const std::vector< CKeyFrame* > vecChunkKeyFrames( m_vecKeyFrames->begin( )+p_uIDBeginKeyFrame, m_vecKeyFrames->end( ) );
    const std::vector< CLandmark* > vecChunkLandmarks( m_vecLandmarks->begin( )+vecChunkKeyFrames.front( )->vecMeasurements.front( )->uID, m_vecLandmarks->end( ) );

    //ds set landmarks
    _loadLandmarksToGraph( vecChunkLandmarks );

    //ds add the first pose separately and fix it
    const CKeyFrame* pKeyFrameInitial = vecChunkKeyFrames.front( );
    g2o::VertexSE3* pVertexPoseFrom = new g2o::VertexSE3( );
    pVertexPoseFrom->setEstimate( pKeyFrameInitial->matTransformationLEFTtoWORLD );
    pVertexPoseFrom->setId( pKeyFrameInitial->uID+m_uIDShift );
    pVertexPoseFrom->setFixed( true );
    m_cOptimizerSparse.addVertex( pVertexPoseFrom );

    //ds always save acceleration data
    //m_cOptimizerSparse.addEdge( _getEdgeLinearAcceleration( pVertexPoseFrom, vecChunkKeyFrames.front( )->vecLinearAccelerationNormalized ) );

    //ds info
    UIDLandmark uMeasurementsStoredPointXYZ    = 0;
    UIDLandmark uMeasurementsStoredUVDepth     = 0;
    UIDLandmark uMeasurementsStoredUVDisparity = 0;
    uint32_t    uLoopClosures                  = 0;

    //ds loop over the camera vertices vector (skipping the first one that we added before)
    for( std::vector< CKeyFrame* >::size_type u = 1; u < vecChunkKeyFrames.size( ); ++u )
    {
        //ds buffer current keyframe
        const CKeyFrame* pKeyFrame( vecChunkKeyFrames[u] );

        //ds add current camera pose
        g2o::VertexSE3* pVertexPoseCurrent = _setAndgetPose( pVertexPoseFrom, pKeyFrame );

        //ds check if we got a loop closure for this frame
        if( 0 != pKeyFrame->pLoopClosure )
        {
            _setLoopClosure( pVertexPoseCurrent, pKeyFrame );
            ++uLoopClosures;
        }

        //ds always save acceleration data
        //m_cOptimizerSparse.addEdge( _getEdgeLinearAcceleration( pVertexPoseCurrent, pKeyFrame->vecLinearAccelerationNormalized ) );

        //ds set landmark measurements
        _setLandmarkMeasurements( pVertexPoseCurrent, pKeyFrame, uMeasurementsStoredPointXYZ, uMeasurementsStoredUVDepth, uMeasurementsStoredUVDisparity );

        //ds update from
        pVertexPoseFrom = pVertexPoseCurrent;
    }

    //ds save initial situation
    char chBuffer[256];
    std::snprintf( chBuffer, 256, "g2o/local/keyframes_%06lu-%06lu", vecChunkKeyFrames.front( )->uID, vecChunkKeyFrames.back( )->uID );
    std::string strPrefix( chBuffer );
    m_cOptimizerSparse.save( ( strPrefix + ".g2o" ).c_str( ) );

    std::printf( "<Cg2oOptimizer>(optimizeTail) optimizing [keyframes: %06lu-%06lu landmarks: %06lu-%06lu] (measurements xyz: %lu depth: %lu disparity: %lu, loop closures: %u)\n",
                 vecChunkKeyFrames.front( )->uID,
                 vecChunkKeyFrames.back( )->uID,
                 vecChunkLandmarks.front( )->uID,
                 vecChunkLandmarks.back( )->uID,
                 uMeasurementsStoredPointXYZ,
                 uMeasurementsStoredUVDepth,
                 uMeasurementsStoredUVDisparity,
                 uLoopClosures );

    //ds initialize optimization
    m_cOptimizerSparse.initializeOptimization( );

    //ds do ten iterations
    m_cOptimizerSparse.optimize( m_uIterations );

    //ds save optimized situation
    m_cOptimizerSparse.save( ( strPrefix + "_optimized.g2o" ).c_str( ) );

    //ds update landmarks and trajectory
    _applyOptimization( vecChunkLandmarks );
    _applyOptimization( vecChunkKeyFrames );

    //ds done
    std::printf( "<Cg2oOptimizer>(optimizeTail) optimized poses: %lu and landmarks: %lu (error final: %f)\n", vecChunkKeyFrames.size( ), vecChunkLandmarks.size( ), m_cOptimizerSparse.activeChi2( ) );
    ++m_uOptimizations;
}

void Cg2oOptimizer::optimizeContinuous( const UIDKeyFrame& p_uIDBeginKeyFrame )
{
    //ds local optimization chunks: +1 at the end because std::vector[begin,end)
    const std::vector< CKeyFrame* > vecChunkKeyFrames( m_vecKeyFrames->begin( )+p_uIDBeginKeyFrame, m_vecKeyFrames->end( ) );
    const std::vector< CLandmark* > vecChunkLandmarks( m_vecLandmarks->begin( )+vecChunkKeyFrames.front( )->vecMeasurements.front( )->uID, m_vecLandmarks->end( ) );

    //ds set landmarks
    _loadLandmarksToGraph( vecChunkLandmarks );

    //ds info
    UIDLandmark uMeasurementsStoredPointXYZ    = 0;
    UIDLandmark uMeasurementsStoredUVDepth     = 0;
    UIDLandmark uMeasurementsStoredUVDisparity = 0;
    uint32_t    uLoopClosures                  = 0;

    //ds loop over the camera vertices vector
    for( const CKeyFrame* pKeyFrame: vecChunkKeyFrames )
    {
        //ds add current camera pose
        g2o::VertexSE3* pVertexPoseCurrent = _setAndgetPose( m_pVertexPoseLAST, pKeyFrame );

        //ds check if we got a loop closure for this frame
        if( 0 != pKeyFrame->pLoopClosure )
        {
            //ds set to the graph
            _setLoopClosure( pVertexPoseCurrent, pKeyFrame );
            ++uLoopClosures;
        }

        //ds always save acceleration data
        //m_cOptimizerSparse.addEdge( _getEdgeLinearAcceleration( pVertexPoseCurrent, pKeyFrame->vecLinearAccelerationNormalized ) );

        //ds set landmark measurements
        _setLandmarkMeasurements( pVertexPoseCurrent, pKeyFrame, uMeasurementsStoredPointXYZ, uMeasurementsStoredUVDepth, uMeasurementsStoredUVDisparity );

        //ds update last
        m_pVertexPoseLAST = pVertexPoseCurrent;
    }

    //ds we always process all keyframes
    m_vecKeyFramesInGraph.insert( m_vecKeyFramesInGraph.begin( ), vecChunkKeyFrames.begin( ), vecChunkKeyFrames.end( ) );

    //ds save initial situation
    char chBuffer[256];
    std::snprintf( chBuffer, 256, "g2o/local/keyframes_%06lu-%06lu", m_vecKeyFramesInGraph.front( )->uID, m_vecKeyFramesInGraph.back( )->uID );
    std::string strPrefix( chBuffer );
    m_cOptimizerSparse.save( ( strPrefix + ".g2o" ).c_str( ) );

    std::printf( "<Cg2oOptimizer>(optimizeSegment) optimizing [keyframes: %06lu-%06lu landmarks: %06lu-%06lu] (measurements xyz: %lu depth: %lu disparity: %lu, loop closures: %u)\n",
                 m_vecKeyFramesInGraph.front( )->uID, m_vecKeyFramesInGraph.back( )->uID, vecChunkLandmarks.front( )->uID, vecChunkLandmarks.back( )->uID, uMeasurementsStoredPointXYZ, uMeasurementsStoredUVDepth, uMeasurementsStoredUVDisparity, uLoopClosures );

    //ds initialize optimization
    m_cOptimizerSparse.initializeOptimization( );

    //ds do ten iterations
    m_cOptimizerSparse.optimize( m_uIterations );

    //ds save optimized situation
    m_cOptimizerSparse.save( ( strPrefix + "_optimized.g2o" ).c_str( ) );

    //ds update landmarks and keyframes
    _applyOptimization( m_vecLandmarksInGraph );
    _applyOptimization( m_vecKeyFramesInGraph );

    //ds done
    std::printf( "<Cg2oOptimizer>(optimizeSegment) optimized poses: %lu and landmarks: %lu (error final: %f)\n", vecChunkKeyFrames.size( ), m_vecLandmarksInGraph.size( ), m_cOptimizerSparse.activeChi2( ) );
    ++m_uOptimizations;
}

bool Cg2oOptimizer::isOptimized( const CLandmark* p_pLandmark )
{
    //ds criteria
    return ( Cg2oOptimizer::m_uMinimumOptimizations < p_pLandmark->uOptimizationsSuccessful ) && ( Cg2oOptimizer::m_dMaximumErrorSquared > p_pLandmark->dCurrentAverageSquaredError );
}

bool Cg2oOptimizer::isKeyFramed( const CLandmark* p_pLandmark )
{
    //ds criteria
    return ( Cg2oOptimizer::m_uMinimumKeyFramePresences < p_pLandmark->uNumberOfKeyFramePresences );
}

void Cg2oOptimizer::clearFiles( ) const
{
    //ds directory handle
    DIR *pDirectory = opendir ( "g2o/local" );

    //ds try to open the directory
    if( 0 != pDirectory )
    {
        //ds file handle
        struct dirent *pFile;

        //ds delete all files
        while( ( pFile = readdir( pDirectory ) ) != NULL )
        {
            //ds validate filename
            if( '.' != pFile->d_name[0] )
            {
                //ds construct full filename and delete
                std::string strFile( "g2o/local/" );
                strFile += pFile->d_name;
                //std::printf( "<Cg2oOptimizer>(clearFiles) removing file: %s\n", strFile.c_str( ) );
                std::remove( strFile.c_str( ) );
            }
        }

        //ds close directory
        closedir( pDirectory );
    }
    else
    {
        //ds FATAL
        throw CExceptionLogfileTree( "g2o folder: g2o/local not set" );
    }
}

g2o::EdgeSE3LinearAcceleration* Cg2oOptimizer::_getEdgeLinearAcceleration( g2o::VertexSE3* p_pVertexPose,
                                                                           const CLinearAccelerationIMU& p_vecLinearAccelerationNormalized ) const
{
    g2o::EdgeSE3LinearAcceleration* pEdgeLinearAcceleration( new g2o::EdgeSE3LinearAcceleration( ) );

    pEdgeLinearAcceleration->setVertex( 0, p_pVertexPose );
    pEdgeLinearAcceleration->setMeasurement( p_vecLinearAccelerationNormalized );
    pEdgeLinearAcceleration->setParameterId( 0, EG2OParameterID::eOFFSET_IMUtoLEFT );
    const double arrInformationMatrixLinearAcceleration[9] = { 100, 0, 0, 0, 100, 0, 0, 0, 100 };
    pEdgeLinearAcceleration->setInformation( g2o::Matrix3D( arrInformationMatrixLinearAcceleration ) );

    return pEdgeLinearAcceleration;
}

g2o::EdgeSE3PointXYZ* Cg2oOptimizer::_getEdgePointXYZ( g2o::VertexSE3* p_pVertexPose,
                                               g2o::VertexPointXYZ* p_pVertexLandmark,
                                               const CPoint3DWORLD& p_vecPointXYZ ) const
{
    g2o::EdgeSE3PointXYZ* pEdgePointXYZ( new g2o::EdgeSE3PointXYZ( ) );

    //ds triangulated 3d point (uncalibrated)
    pEdgePointXYZ->setVertex( 0, p_pVertexPose );
    pEdgePointXYZ->setVertex( 1, p_pVertexLandmark );
    pEdgePointXYZ->setMeasurement( p_vecPointXYZ );
    pEdgePointXYZ->setParameterId( 0, EG2OParameterID::eWORLD );

    //ds the closer to the camera the point is the more meaningful is the measurement
    //const double dInformationStrengthXYZ( m_dMaximumReliableDepth/( 1+pMeasurementLandmark->vecPointXYZ.z( ) ) );
    const double arrInformationMatrixXYZ[9] = { 1000, 0, 0, 0, 1000, 0, 0, 0, 1000 };
    pEdgePointXYZ->setInformation( g2o::Matrix3D( arrInformationMatrixXYZ ) );

    return pEdgePointXYZ;
}

g2o::EdgeSE3PointXYZDepth* Cg2oOptimizer::_getEdgeUVDepthLEFT( g2o::VertexSE3* p_pVertexPose,
                                                               g2o::VertexPointXYZ* p_pVertexLandmark,
                                                               const CMeasurementLandmark* p_pMeasurement ) const
{
    //ds projected depth
    g2o::EdgeSE3PointXYZDepth* pEdgeProjectedDepth( new g2o::EdgeSE3PointXYZDepth( ) );

    pEdgeProjectedDepth->setVertex( 0, p_pVertexPose );
    pEdgeProjectedDepth->setVertex( 1, p_pVertexLandmark );
    pEdgeProjectedDepth->setMeasurement( g2o::Vector3D( p_pMeasurement->ptUVLEFT.x, p_pMeasurement->ptUVLEFT.y, p_pMeasurement->vecPointXYZLEFT.z( ) ) );
    pEdgeProjectedDepth->setParameterId( 0, EG2OParameterID::eCAMERA_LEFT );

    //ds information matrix
    //const double dInformationQualityDepth( m_dMaximumReliableDepth/dDepthMeters );
    const double arrInformationMatrixDepth[9] = { 1, 0, 0, 0, 1, 0, 0, 0, 1000 };
    pEdgeProjectedDepth->setInformation( g2o::Matrix3D( arrInformationMatrixDepth ) );

    return pEdgeProjectedDepth;
}

g2o::EdgeSE3PointXYZDisparity* Cg2oOptimizer::_getEdgeUVDisparityLEFT( g2o::VertexSE3* p_pVertexPose,
                                                           g2o::VertexPointXYZ* p_pVertexLandmark,
                                                           const double& p_dDisparityPixels,
                                                           const CMeasurementLandmark* p_pMeasurement,
                                                           const double& p_dFxPixels,
                                                           const double& p_dBaselineMeters ) const
{
    //ds disparity
    g2o::EdgeSE3PointXYZDisparity* pEdgeDisparity( new g2o::EdgeSE3PointXYZDisparity( ) );

    const double dDisparityNormalized( p_dDisparityPixels/( p_dFxPixels*p_dBaselineMeters ) );
    pEdgeDisparity->setVertex( 0, p_pVertexPose );
    pEdgeDisparity->setVertex( 1, p_pVertexLandmark );
    pEdgeDisparity->setMeasurement( g2o::Vector3D( p_pMeasurement->ptUVLEFT.x, p_pMeasurement->ptUVLEFT.y, dDisparityNormalized ) );
    pEdgeDisparity->setParameterId( 0, EG2OParameterID::eCAMERA_LEFT );

    //ds information matrix
    //const double dInformationQualityDisparity( std::abs( dDisparity )+1.0 );
    const double arrInformationMatrixDisparity[9] = { 1, 0, 0, 0, 1, 0, 0, 0, 1000 };
    pEdgeDisparity->setInformation( g2o::Matrix3D( arrInformationMatrixDisparity ) );

    return pEdgeDisparity;
}

void Cg2oOptimizer::_loadLandmarksToGraph( const std::vector< CLandmark* >& p_vecChunkLandmarks )
{
    //ds info
    UIDLandmark uDroppedLandmarksInvalid    = 0;
    UIDLandmark uDroppedLandmarksKeyFraming = 0;

    //ds add landmarks
    for( CLandmark* pLandmark: p_vecChunkLandmarks )
    {
        //ds check if optimiziation criterias are met
        if( isOptimized( pLandmark ) )
        {
            if( isKeyFramed( pLandmark ) )
            {
                //ds set landmark vertex
                g2o::VertexPointXYZ* pVertexLandmark = new g2o::VertexPointXYZ( );
                pVertexLandmark->setEstimate( pLandmark->vecPointXYZOptimized );
                pVertexLandmark->setId( pLandmark->uID );

                //ds add vertex to optimizer
                m_cOptimizerSparse.addVertex( pVertexLandmark );
                m_vecLandmarksInGraph.push_back( pLandmark );
            }
            else
            {
                ++uDroppedLandmarksKeyFraming;
            }
        }
        else
        {
            ++uDroppedLandmarksInvalid;
        }
    }

    std::printf( "<Cg2oOptimizer>(_loadLandmarksToGraph) dropped landmarks due to invalid optimization: %lu (%4.2f)\n", uDroppedLandmarksInvalid, static_cast< double >( uDroppedLandmarksInvalid )/p_vecChunkLandmarks.size( ) );
    std::printf( "<Cg2oOptimizer>(_loadLandmarksToGraph) dropped landmarks due to missing key frame presence: %lu (%4.2f)\n", uDroppedLandmarksKeyFraming, static_cast< double >( uDroppedLandmarksKeyFraming )/p_vecChunkLandmarks.size( ) );
}

g2o::VertexSE3* Cg2oOptimizer::_setAndgetPose( g2o::VertexSE3* p_pVertexPoseFrom, const CKeyFrame* pKeyFrameCurrent )
{
    //ds add current camera pose
    g2o::VertexSE3* pVertexPoseCurrent( new g2o::VertexSE3( ) );
    pVertexPoseCurrent->setEstimate( pKeyFrameCurrent->matTransformationLEFTtoWORLD );
    pVertexPoseCurrent->setId( pKeyFrameCurrent->uID+m_uIDShift );
    m_cOptimizerSparse.addVertex( pVertexPoseCurrent );

    //ds set up the edge to connect the poses
    g2o::EdgeSE3* pEdgePoseFromTo( new g2o::EdgeSE3( ) );

    //ds set viewpoints and measurement
    pEdgePoseFromTo->setVertex( 0, p_pVertexPoseFrom );
    pEdgePoseFromTo->setVertex( 1, pVertexPoseCurrent );
    pEdgePoseFromTo->setMeasurement( p_pVertexPoseFrom->estimate( ).inverse( )*pVertexPoseCurrent->estimate( ) );

    //ds information quality
    const double arrInformationMatrixPose[36] = { 100,0,0,0,0,0,
                                                  0,100,0,0,0,0,
                                                  0,0,100,0,0,0,
                                                  0,0,0,10000,0,0,
                                                  0,0,0,0,10000,0,
                                                  0,0,0,0,0,10000 };
    pEdgePoseFromTo->setInformation( Eigen::Matrix< double, 6, 6 >( arrInformationMatrixPose ) );

    //ds add to optimizer
    m_cOptimizerSparse.addEdge( pEdgePoseFromTo );

    return pVertexPoseCurrent;
}

void Cg2oOptimizer::_setLoopClosure( g2o::VertexSE3* p_pVertexPoseCurrent, const CKeyFrame* pKeyFrameCurrent )
{
    //ds loop closure reference
    const CKeyFrame* pKeyFrameClosure = pKeyFrameCurrent->pLoopClosure->pKeyFrameReference;

    //ds find the corresponding pose in the current graph
    const g2o::HyperGraph::VertexIDMap::iterator itClosure( m_cOptimizerSparse.vertices( ).find( pKeyFrameClosure->uID+m_uIDShift ) );

    //ds pose to close
    g2o::VertexSE3* pVertexClosure = 0;

    //ds if found
    if( itClosure != m_cOptimizerSparse.vertices( ).end( ) )
    {
        //ds extract the pose
         pVertexClosure = dynamic_cast< g2o::VertexSE3* >( itClosure->second );
    }
    else
    {
        //ds we have to add the pose and fix it
        pVertexClosure = new g2o::VertexSE3( );
        pVertexClosure->setEstimate( pKeyFrameClosure->matTransformationLEFTtoWORLD );
        pVertexClosure->setId( pKeyFrameClosure->uID+m_uIDShift );
        pVertexClosure->setFixed( true );
        m_cOptimizerSparse.addVertex( pVertexClosure );
    }

    //ds set up the edge
    g2o::EdgeSE3* pEdgeLoopClosure( new g2o::EdgeSE3( ) );

    //ds set viewpoints and measurement
    pEdgeLoopClosure->setVertex( 0, pVertexClosure );
    pEdgeLoopClosure->setVertex( 1, p_pVertexPoseCurrent );
    pEdgeLoopClosure->setMeasurement( pKeyFrameCurrent->pLoopClosure->matTransformationToClosure );

    //ds information quality (10x pose)
    const double arrInformationMatrixLoopClosure[36] = { 1000,0,0,0,0,0,
                                                         0,1000,0,0,0,0,
                                                         0,0,1000,0,0,0,
                                                         0,0,0,100000,0,0,
                                                         0,0,0,0,100000,0,
                                                         0,0,0,0,0,100000 };

    //ds information quality (10x more relevant than regular pose measurement)
    pEdgeLoopClosure->setInformation( Eigen::Matrix< double, 6, 6 >( arrInformationMatrixLoopClosure ) );

    /*ds set kernel (instead of doing closure evaluation)
    g2o::RobustKernelDCS* pKernel = new g2o::RobustKernelDCS( );
    pKernel->setDelta( 2000.0 );
    pEdgeLoopClosure->setRobustKernel( pKernel );*/
    pEdgeLoopClosure->setRobustKernel( new g2o::RobustKernelCauchy( ) );

    //ds add to optimizer
    m_cOptimizerSparse.addEdge( pEdgeLoopClosure );

    /*ds add closure to buffer
    m_cClosureBuffer.addEdge( pEdgeLoopClosure );
    m_cClosureBuffer.addVertex( p_pVertexPoseCurrent );

    //ds check if we have good closures and add them on success
    if( m_cClosureBuffer.checkList( Cg2oOptimizer::m_uClosureBufferWindowSize ) )
    {
        m_cClosureChecker.init( m_cClosureBuffer.vertices( ), m_cClosureBuffer.edgeSet( ), Cg2oOptimizer::m_dClosureCheckerInlierThreshold );
        m_cClosureChecker.check( );

        std::printf( "<Cg2oOptimizer>(_setLoopClosure) chi2: %f inliers: %u\n", m_cClosureChecker.chi2( ), m_cClosureChecker.inliers( ) );

        if (m_cClosureChecker.inliers( ) > Cg2oOptimizer::m_uMinimumNumberOfInliers){
          LoopClosureChecker::EdgeDoubleMap results = m_cClosureChecker.closures();
          cout << "Results:" << endl;
          for (LoopClosureChecker::EdgeDoubleMap::iterator it= results.begin(); it!= results.end(); it++){
        EdgeSE3* e = (EdgeSE3*) (it->first);
        assert( 0 != e );
        cout << "Edge from: " << e->vertices()[0]->id() << " to: " << e->vertices()[1]->id() << ". Chi2 = " << it->second <<  endl;

        if (it->second < Cg2oOptimizer::m_dClosureCheckerInlierThreshold){
          cout << "Is an inlier. Adding to Graph" << endl;
          //ds add to optimizer
          m_cOptimizerSparse.addEdge( e );
        }
          }
        }
    }*/
}

void Cg2oOptimizer::_setLandmarkMeasurements( g2o::VertexSE3* p_pVertexPoseCurrent,
                                      const CKeyFrame* pKeyFrameCurrent,
                                      UIDLandmark& p_uMeasurementsStoredPointXYZ,
                                      UIDLandmark& p_uMeasurementsStoredUVDepth,
                                      UIDLandmark& p_uMeasurementsStoredUVDisparity )
{
    //ds check visible landmarks and add the edges for the current pose
    for( const CMeasurementLandmark* pMeasurementLandmark: pKeyFrameCurrent->vecMeasurements )
    {
        //ds find the corresponding landmark
        const g2o::HyperGraph::VertexIDMap::iterator itLandmark( m_cOptimizerSparse.vertices( ).find( pMeasurementLandmark->uID ) );

        //ds if found
        if( itLandmark != m_cOptimizerSparse.vertices( ).end( ) )
        {
            //ds get the respective feature vertex (this only works if the landmark id's are preserved in the optimizer)
            g2o::VertexPointXYZ* pVertexLandmark( dynamic_cast< g2o::VertexPointXYZ* >( itLandmark->second ) );

            //ds get optimized landmark into current pose
            //const CPoint3DCAMERA vecPointOptimized( pVertexPoseCurrent->estimate( ).inverse( )*pVertexLandmark->estimate( ) );

            //ds current depth
            const double dDepthMeters = pMeasurementLandmark->vecPointXYZLEFT.z( );

            /*ds check depth change
            const double dDepthChange = std::fabs( vecPointOptimized.z( )-dDepthMeters );
            if( CBridgeG2O::m_dMaximumReliableDepthForUVDepth > dDepthMeters && 1.0 < dDepthChange )
            {
                std::printf( "<CBridgeG2O>(saveCOMBO) depth change: %f\n", dDepthChange );
            }*/

            //ds maximum depth to produce a reliable XYZ estimate
            if( m_dMaximumReliableDepthForPointXYZ > dDepthMeters )
            {
                //ds retrieve the edge
                g2o::EdgeSE3PointXYZ* pEdgePointXYZ = _getEdgePointXYZ( p_pVertexPoseCurrent, pVertexLandmark, pMeasurementLandmark->vecPointXYZLEFT );

                //ds set robust kernel
                //pEdgePointXYZ->setRobustKernel( new g2o::RobustKernelCauchy( ) );

                //ds add the edge to the graph
                m_cOptimizerSparse.addEdge( pEdgePointXYZ );
                ++p_uMeasurementsStoredPointXYZ;
            }

            /*ds still good enough for uvdepth
            else if( m_dMaximumReliableDepthForUVDepth > dDepthMeters )
            {
                //ds get the edge
                g2o::EdgeSE3PointXYZDepth* pEdgeUVDepth( _getEdgeUVDepthLEFT( pVertexPoseCurrent, pVertexLandmark, pMeasurementLandmark ) );

                //ds set robust kernel
                //pEdgeUVDepth->setRobustKernel( new g2o::RobustKernelCauchy( ) );

                //ds projected depth (LEFT camera)
                m_cOptimizerSparse.addEdge( pEdgeUVDepth );
                ++p_uMeasurementsStoredUVDepth;
            }

            //ds go with disparity
            else
            {
                //ds current disparity
                const double dDisparity = pMeasurementLandmark->ptUVLEFT.x-pMeasurementLandmark->ptUVRIGHT.x;

                //ds get a disparity edge
                g2o::EdgeSE3PointXYZDisparity* pEdgeUVDisparity( _getEdgeUVDisparityLEFT( pVertexPoseCurrent, pVertexLandmark, dDisparity, pMeasurementLandmark, m_pCameraSTEREO->m_pCameraLEFT->m_dFxP, m_pCameraSTEREO->m_dBaselineMeters ) );

                //ds set robust kernel
                //pEdgeUVDisparity->setRobustKernel( new g2o::RobustKernelCauchy( ) );

                //ds disparity (LEFT camera)
                m_cOptimizerSparse.addEdge( pEdgeUVDisparity );
                ++p_uMeasurementsStoredUVDisparity;
            }*/
        }
    }
}

void Cg2oOptimizer::_applyOptimization( const std::vector< CLandmark* >& p_vecChunkLandmarks )
{
    //ds update landmarks
    for( CLandmark* pLandmark: p_vecChunkLandmarks )
    {
        //ds find the corresponding landmark
        const g2o::HyperGraph::VertexIDMap::iterator itLandmark( m_cOptimizerSparse.vertices( ).find( pLandmark->uID ) );

        //ds must be present
        //assert( itLandmark != m_cOptimizerSparse.vertices( ).end( ) );
        if( itLandmark != m_cOptimizerSparse.vertices( ).end( ) )
        {
            const g2o::VertexPointXYZ* pVertexLandmark( dynamic_cast< g2o::VertexPointXYZ* >( itLandmark->second ) );

            //ds update position
            pLandmark->vecPointXYZOptimized = pVertexLandmark->estimate( );
        }
    }
}

void Cg2oOptimizer::_applyOptimization( const std::vector< CKeyFrame* >& p_vecChunkKeyFrames )
{
    //ds update trajectory
    for( CKeyFrame* pKeyFrame: p_vecChunkKeyFrames )
    {
        //ds find the corresponding pose
        const g2o::HyperGraph::VertexIDMap::iterator itPose( m_cOptimizerSparse.vertices( ).find( pKeyFrame->uID+m_uIDShift ) );

        //ds must be present
        assert( itPose != m_cOptimizerSparse.vertices( ).end( ) );

        const g2o::VertexSE3* pVertexPose( dynamic_cast< g2o::VertexSE3* >( itPose->second ) );

        //ds update position
        pKeyFrame->matTransformationLEFTtoWORLD = pVertexPose->estimate( );
        pKeyFrame->bIsOptimized                 = true;
    }
}
