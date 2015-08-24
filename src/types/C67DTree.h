#ifndef C67DTREE_H
#define C67DTREE_H

#include <flann/flann.hpp>

#include "TypesCloud.h"

class C67DTree
{

public:

    C67DTree( const UIDCloud& p_uID,
              const Eigen::Isometry3d& p_matPose,
              const UIDDescriptorPoint3D m_uPoints,
              float* p_arrDescriptorPoints );
    ~C67DTree( );

public:

    const UIDKeyFrame uID;
    const Eigen::Isometry3d matPose;
    flann::Index< flann::L2< float > >* pIndex;

private:

    const flann::Matrix< float > m_matData;
    flann::Matrix< int > m_matIndicies;
    flann::Matrix< float > m_matDistances;

public:

    int32_t getMatches( const C67DTree* p_pTreeQuery );

};

#endif //C67DTREE_H
