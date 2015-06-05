#include "CBridgeG2O.h"

#include "g2o/core/sparse_optimizer.h"
#include "g2o/core/block_solver.h"
#include "g2o/core/solver.h"
#include "g2o/core/optimization_algorithm_levenberg.h"
#include "g2o/solvers/dense/linear_solver_dense.h"
#include "g2o/types/icp/types_icp.h"
#include "g2o/types/sba/types_sba.h"
#include "g2o/types/sba/types_six_dof_expmap.h"
#include "g2o/types/slam3d/edge_se3_pointxyz_uv.h"

void CBridgeG2O::savesolveAndOptimizeG2O( const std::string& p_strOutfile,
                                          const CStereoCamera& p_cStereoCamera,
                                          const std::vector< CLandmark* >& p_vecLandmarks,
                                          const std::vector< std::pair< Eigen::Isometry3d, std::shared_ptr< std::vector< CLandmarkMeasurement* > > > >& p_vecMeasurements )
{
    //ds allocate an optimizer
    g2o::SparseOptimizer cOptimizer;
    cOptimizer.setVerbose( true );

    //ds set the solver
    g2o::BlockSolverX::LinearSolverType* pLinearSolver = new g2o::LinearSolverDense< g2o::BlockSolverX::PoseMatrixType> ( );
    g2o::BlockSolverX* pSolver                         = new g2o::BlockSolverX( pLinearSolver );
    g2o::OptimizationAlgorithmLevenberg* pAlgorithm    = new g2o::OptimizationAlgorithmLevenberg( pSolver );
    cOptimizer.setAlgorithm( pAlgorithm );

    //ds set camera parameters
    g2o::ParameterCamera* pCameraParametersLEFT = new g2o::ParameterCamera( );
    pCameraParametersLEFT->setKcam( p_cStereoCamera.m_pCameraLEFT->m_dFxNormalized, p_cStereoCamera.m_pCameraLEFT->m_dFyNormalized, p_cStereoCamera.m_pCameraLEFT->m_dCxNormalized, p_cStereoCamera.m_pCameraLEFT->m_dCyNormalized );
    pCameraParametersLEFT->setId( 0 );
    cOptimizer.addParameter( pCameraParametersLEFT );
    g2o::ParameterCamera* pCameraParametersRIGHT = new g2o::ParameterCamera( );
    pCameraParametersRIGHT->setKcam( p_cStereoCamera.m_pCameraRIGHT->m_dFxNormalized, p_cStereoCamera.m_pCameraRIGHT->m_dFyNormalized, p_cStereoCamera.m_pCameraRIGHT->m_dCxNormalized, p_cStereoCamera.m_pCameraRIGHT->m_dCyNormalized );
    pCameraParametersRIGHT->setId( 1 );
    cOptimizer.addParameter( pCameraParametersRIGHT );

    //ds g2o element identifier
    uint64_t uNextAvailableUID( 0 );

    //ds add landmarks
    for( const CLandmark* pLandmark: p_vecLandmarks )
    {
        //ds set landmark vertex
        g2o::VertexPointXYZ* pVertexLandmark = new g2o::VertexPointXYZ( );
        pVertexLandmark->setEstimate( pLandmark->vecPositionXYZ.head( 3 ) );
        pVertexLandmark->setId( uNextAvailableUID );

        assert( pLandmark->uID == uNextAvailableUID );

        //ds add vertex to optimizer
        cOptimizer.addVertex( pVertexLandmark );
        ++uNextAvailableUID;
    }

    //ds add the first pose separately (there's no edge to the "previous" pose)
    g2o::VertexSE3 *pVertexPose = new g2o::VertexSE3( );
    pVertexPose->setEstimate( p_vecMeasurements.front( ).first );
    pVertexPose->setId( uNextAvailableUID );
    pVertexPose->setFixed( true );
    cOptimizer.addVertex( pVertexPose );
    ++uNextAvailableUID;

    //ds loop over the camera vertices vector (skipping the first one that we added before)
    for( std::vector< std::pair< Eigen::Isometry3d, std::shared_ptr< std::vector< CLandmarkMeasurement* > > > >::const_iterator pMeasurement = p_vecMeasurements.begin( )+1; pMeasurement != p_vecMeasurements.end( ); ++pMeasurement )
    {
        //ds add current camera pose
        g2o::VertexSE3* pVertexPoseCurrent = new g2o::VertexSE3( );
        pVertexPoseCurrent->setEstimate( pMeasurement->first );
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
        for( const CLandmarkMeasurement* pLandmark: *pMeasurement->second )
        {
            //ds set up the edges
            g2o::EdgeSE3PointXYZUV* pEdgeLandmarkCoordinates      = new g2o::EdgeSE3PointXYZUV( );
            //g2o::EdgeSE3PointXYZDisparity* pEdgeLandmarkDisparity = new g2o::EdgeSE3PointXYZDisparity( );

            //ds get the respective feature vertex (this only works if the landmark id's are preserved in the optimizer)
            g2o::VertexPointXYZ* pVertexLandmark = dynamic_cast< g2o::VertexPointXYZ* >( cOptimizer.vertices( ).find( pLandmark->uID )->second );

            //ds set viewpoints and measurements
            pEdgeLandmarkCoordinates->setVertex( 0, pVertexPoseCurrent );
            pEdgeLandmarkCoordinates->setVertex( 1, pVertexLandmark );
            pEdgeLandmarkCoordinates->setMeasurement( static_cast< g2o::Vector2D >( pLandmark->vecPositionUV ) );
            pEdgeLandmarkCoordinates->setParameterId( 0, 0 );

            /*lo dasdasd
            pEdgeLandmarkDisparity->setVertex( 0, pVertexPoseCurrent );
            pEdgeLandmarkDisparity->setVertex( 1, pVertexLandmark );
            pEdgeLandmarkDisparity->setParameterId( 0, 0 );*/

            //ds add to optimizer
            cOptimizer.addEdge( pEdgeLandmarkCoordinates );
            //cOptimizer.addEdge( pEdgeLandmarkDisparity );
        }
    }

    cOptimizer.save( p_strOutfile.c_str( ) );

    //ds optimize!
    //cOptimizer.initializeOptimization( );
    //cOptimizer.computeActiveErrors( );
    //cOptimizer.optimize( 1 );
}
