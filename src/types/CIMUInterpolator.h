#ifndef CIMUINTERPOLATOR_H
#define CIMUINTERPOLATOR_H

#include <map>
#include <Eigen/Core>

#include "vision/CMiniVisionToolbox.h"

class CIMUInterpolator
{

//ds ctor/dtor
public:

    CIMUInterpolator( const double& p_dMaximumDeltaTimeSeconds = 0.1 ): m_vecLastVelocityLinear( 0.0, 0.0, 0.0 ),
                                                                        m_vecLastVelocityAngular( 0.0, 0.0, 0.0 ),
                                                                        m_vecOffsetAccelerationLinear( 0.0, 9.80665, 0.0 ),
                                                                        m_vecOffsetVelocityAngular( 0.0, 0.0, 0.0 ),
                                                                        m_dLastTimestamp( 0.0 ),
                                                                        m_dMaximumDeltaTimeSeconds( p_dMaximumDeltaTimeSeconds )
    {
        m_mapTransformationsIMU.clear( );
        m_vecCalibration.clear( );
    }

    ~CIMUInterpolator( )
    {
        m_mapTransformationsIMU.clear( );
        m_vecCalibration.clear( );
    }

//ds fields
private:

    std::map< double, Eigen::Isometry3d > m_mapTransformationsIMU;

    Eigen::Vector3d m_vecLastVelocityLinear;
    Eigen::Vector3d m_vecLastVelocityAngular;

    std::vector< std::pair< Eigen::Vector3d, Eigen::Vector3d > > m_vecCalibration;
    Eigen::Vector3d m_vecOffsetAccelerationLinear; //ds TODO calibration
    Eigen::Vector3d m_vecOffsetVelocityAngular; //ds TODO calibration

    double m_dLastTimestamp;
    const double m_dMaximumDeltaTimeSeconds;

//ds accessors
public:

    void addMeasurementCalibration( const Eigen::Vector3d& p_vecAccelerationLinear, const Eigen::Vector3d& p_vecVelocityAngular )
    {
        m_vecCalibration.push_back( std::pair< Eigen::Vector3d, Eigen::Vector3d >( p_vecAccelerationLinear, p_vecVelocityAngular ) );
    }

    void calibrateOffsets( )
    {
        std::for_each( m_vecCalibration.begin( ),m_vecCalibration.end( ),[&]( std::pair< Eigen::Vector3d, Eigen::Vector3d > prMeasurement )
        {
            m_vecOffsetAccelerationLinear += prMeasurement.first;
            m_vecOffsetVelocityAngular    += prMeasurement.second;
        });

        //ds compute means
        m_vecOffsetAccelerationLinear /= m_vecCalibration.size( );
        m_vecOffsetVelocityAngular    /= m_vecCalibration.size( );
    }

    void addMeasurement( Eigen::Vector3d& p_vecAccelerationLinear, const Eigen::Vector3d& p_vecVelocityAngular, const double& p_dTimestampSeconds )
    {
        p_vecAccelerationLinear.x( ) *= 0.05;
        p_vecAccelerationLinear.z( ) *= 0.15;
        //std::cout << p_vecAccelerationLinear.transpose( ) << std::endl;

        //ds calibrated measurements
        const Eigen::Vector3d& vecAccelerationLinearClean( p_vecAccelerationLinear-m_vecOffsetAccelerationLinear );
        const Eigen::Vector3d& vecVelocityAngularClean( p_vecVelocityAngular-m_vecOffsetVelocityAngular );

        //ds compute time delta
        const double dTimespanSeconds = p_dTimestampSeconds-m_dLastTimestamp;

        if( m_dMaximumDeltaTimeSeconds > dTimespanSeconds )
        {
            //std::cout << vecAccelerationLinearClean.transpose( ) << std::endl;
            //std::cout << vecVelocityAngularClean.transpose( ) << std::endl;

            //ds set current transformation
            Eigen::Isometry3d matTransformationIMU( Eigen::Matrix4d::Identity( ) );
            matTransformationIMU.translation( ) = m_vecLastVelocityLinear*dTimespanSeconds + 1/2.0*vecAccelerationLinearClean*dTimespanSeconds*dTimespanSeconds;
            matTransformationIMU.linear( )      = CMiniVisionToolbox::fromOrientationRodrigues( static_cast< Eigen::Vector3d >( dTimespanSeconds*m_vecLastVelocityAngular ) );

            //std::cout << matTransformationIMU.matrix( ) << std::endl;

            //ds compute next transformation and add it to the map
            m_mapTransformationsIMU.insert( std::pair< double, Eigen::Isometry3d >( p_dTimestampSeconds, matTransformationIMU*m_mapTransformationsIMU.at( m_dLastTimestamp ) ) );

            //std::cout << m_mapTransformationsIMU.at( p_dTimestampSeconds ).matrix( ) << std::endl;

            //ds update references
            m_vecLastVelocityLinear += vecAccelerationLinearClean*dTimespanSeconds;
            m_vecLastVelocityAngular = vecVelocityAngularClean;
        }
        else
        {
            //ds add identity transform
            m_mapTransformationsIMU.insert( std::pair< double, Eigen::Isometry3d >( p_dTimestampSeconds, Eigen::Isometry3d( Eigen::Matrix4d::Identity( ) ) ) );
        }

        //ds update timestamp
        m_dLastTimestamp = p_dTimestampSeconds;
    }

    const Eigen::Isometry3d getTransformation( const double& p_dTimestampSeconds ) const
    {
        return m_mapTransformationsIMU.at( p_dTimestampSeconds );
    }

};

#endif //CIMUINTERPOLATOR_H
