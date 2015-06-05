#ifndef CBRIDGEG2O_H
#define CBRIDGEG2O_H

#include "CStereoCamera.h"
#include "types/CLandmark.h"

class CBridgeG2O
{

public:

    static void savesolveAndOptimizeG2O( const std::string& p_strOutfile,
                                         const CStereoCamera& p_cStereoCamera,
                                         const std::vector< CLandmark* >& p_vecLandmarks,
                                         const std::vector< std::pair< Eigen::Isometry3d, std::shared_ptr< std::vector< CLandmarkMeasurement* > > > >& p_vecMeasurements );

};

#endif //CBRIDGEG2O_H
