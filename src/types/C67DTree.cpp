#include "C67DTree.h"

C67DTree::C67DTree( const UIDCloud& p_uID,
          const Eigen::Isometry3d& p_matPose,
          const UIDDescriptorPoint3D m_uPoints,
          float* p_arrDescriptorPoints ): uID( p_uID ),
                                                                       matPose( p_matPose ),
                                                                       m_matData( flann::Matrix< float >( &p_arrDescriptorPoints[0], m_uPoints, 67 ) ),
                                                                       m_matIndicies( flann::Matrix< int >( new int[m_uPoints], m_uPoints, 1 ) ),
                                                                       m_matDistances( flann::Matrix< float >( new float[m_uPoints], m_uPoints, 1 ) )
{
    //ds allocate the index and try to build it
    pIndex = new flann::Index< flann::L2< float > >( m_matData, flann::KDTreeIndexParams( ) );
    pIndex->buildIndex( );
}

C67DTree::~C67DTree( )
{
    delete[] m_matData.ptr( );
    delete[] m_matIndicies.ptr( );
    delete[] m_matDistances.ptr( );
    delete pIndex;
}

int32_t C67DTree::getMatches( const C67DTree* p_pTreeReference )
{
    //ds do a radius search on this tree (being the query)
    return p_pTreeReference->pIndex->radiusSearch( m_matData, m_matIndicies, m_matDistances, 1000.0, flann::SearchParams( 16 ) );
}
