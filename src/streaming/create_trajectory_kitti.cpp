//ds std
#include <iostream>

//ds custom
#include "g2o/core/sparse_optimizer.h"
#include "configuration/CConfigurationCameraKITTI.h"
#include "utility/CLogger.h"



int main( int argc, char **argv )
{
    //ds pwd info
    std::printf( "(main) launched: %s\n", argv[0] );

    //ds configuration parameters
    std::string strInfileName  = "/home/n551jw/Downloads/dataset/poses/00.txt";
    std::string strOutfileName = strInfileName.substr( 0, strInfileName.size( )-3 );
    strOutfileName += "g2o";

    //ds if specific filename set TODO real parsing
    if( 2 == argc )
    {
        strInfileName = argv[1];
    }

    //ds log configuration
    CLogger::openBox( );
    std::printf( "(main) strInfileName  := '%s'\n", strInfileName.c_str( ) );
    std::printf( "(main) strOutfileName := '%s'\n", strOutfileName.c_str( ) );
    CLogger::closeBox( );

    //ds graph handle
    g2o::OptimizableGraph cGraph;
    cGraph.clear( );

    //ds to our WORLD
    const Eigen::Isometry3d matTransformationLEFTtoWORLDInitial( Eigen::Matrix4d( CConfigurationCameraKITTI::matTransformationIntialLEFTtoWORLD ) );

    //ds set first pose
    UIDKeyFrame uID = 0;
    g2o::VertexSE3* pVertexPoseLAST = new g2o::VertexSE3( );
    pVertexPoseLAST->setEstimate( matTransformationLEFTtoWORLDInitial );
    pVertexPoseLAST->setId( uID );
    pVertexPoseLAST->setFixed( true );
    cGraph.addVertex( pVertexPoseLAST );
    ++uID;

    //ds read file and create graph
    std::ifstream ifTrajectory( strInfileName, std::ifstream::in );
    while( ifTrajectory.good( ) )
    {
        //ds line buffer
        std::string strLineBuffer;

        //ds read one line
        std::getline( ifTrajectory, strLineBuffer );

        //ds get it to a stringstream
        std::istringstream issLine( strLineBuffer );

        //ds information fields
        Eigen::Matrix4d matTransformationRAW( Eigen::Matrix4d::Identity( ) );

        for( uint8_t u = 0; u < 3; ++u )
        {
            for( uint8_t v = 0; v < 4; ++v )
            {
                issLine >> matTransformationRAW(u,v);
            }
        }

        //ds set transformation matrix
        const Eigen::Isometry3d matTransformationLEFTtoWORLD( matTransformationRAW );

        //ds add current camera pose
        g2o::VertexSE3* pVertexPoseCurrent = new g2o::VertexSE3( );
        pVertexPoseCurrent->setEstimate( matTransformationLEFTtoWORLDInitial*matTransformationLEFTtoWORLD );
        pVertexPoseCurrent->setId( uID );
        cGraph.addVertex( pVertexPoseCurrent );

        //ds set up the edge to connect the poses
        g2o::EdgeSE3* pEdgePoseFromTo = new g2o::EdgeSE3( );

        //ds set viewpoints and measurement
        pEdgePoseFromTo->setVertex( 0, pVertexPoseLAST );
        pEdgePoseFromTo->setVertex( 1, pVertexPoseCurrent );
        pEdgePoseFromTo->setMeasurement( pVertexPoseLAST->estimate( ).inverse( )*pVertexPoseCurrent->estimate( ) );

        //ds add to optimizer
        cGraph.addEdge( pEdgePoseFromTo );

        //ds update last
        pVertexPoseLAST = pVertexPoseCurrent;
        ++uID;
    }

    std::printf( "(main) successfully added poses: %lu\n", uID );
    cGraph.save( strOutfileName.c_str( ) );
    std::printf( "(main) g2o file saved to: %s\n", strOutfileName.c_str( ) );
    std::printf( "(main) terminated: %s\n", argv[0] );
    return 0;
}
