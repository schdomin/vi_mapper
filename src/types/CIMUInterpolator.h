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
                                                                        m_vecOffsetAccelerationLinear( 0.2, 9.80665, 0.5 ),
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

    //ds IMU filtering (calibrated 2015-07-25)
    static constexpr double m_dImprecisionAngularVelocity    = 0.2;
    static constexpr double m_dImprecisionLinearAcceleration = 0.5; //{ 0.5, 0.1, 0.1 };
    static constexpr double m_vecBiasLinearAcceleration[3]   = { 0.0, 0.0, -9.80665 }; //ds compensate gravitational component (http://en.wikipedia.org/wiki/ISO_80000-3)

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

        std::printf( "<CIMUInterpolator>(calibrateOffsets) calibrated offsets for linear acceleration: %5.2f %5.2f %5.2f\n", m_vecOffsetAccelerationLinear.x( ), m_vecOffsetAccelerationLinear.y( ), m_vecOffsetAccelerationLinear.z( ) );
        std::printf( "<CIMUInterpolator>(calibrateOffsets) calibrated offsets for angular velocity: %5.2f %5.2f %5.2f\n", m_vecOffsetVelocityAngular.x( ), m_vecOffsetVelocityAngular.y( ), m_vecOffsetVelocityAngular.z( ) );
    }

    void addMeasurement( Eigen::Vector3d& p_vecAccelerationLinear, const Eigen::Vector3d& p_vecVelocityAngular, const double& p_dTimestampSeconds )
    {
        //std::cout << p_vecAccelerationLinear.transpose( ) << std::endl;

        //ds calibrated measurements
        const Eigen::Vector3d vecAccelerationLinearClean( p_vecAccelerationLinear-m_vecOffsetAccelerationLinear );
        const Eigen::Vector3d vecVelocityAngularClean( p_vecVelocityAngular-m_vecOffsetVelocityAngular );

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

//ds statics
public:

    static const CLinearAccelerationInIMUFrame getLinearAccelerationFiltered( const CLinearAccelerationInIMUFrame& p_vecLinearAcceleration )
    {
        const double dAccelerationX = p_vecLinearAcceleration.x( )-CIMUInterpolator::m_vecBiasLinearAcceleration[0];
        const double dAccelerationY = p_vecLinearAcceleration.y( )-CIMUInterpolator::m_vecBiasLinearAcceleration[1];
        const double dAccelerationZ = p_vecLinearAcceleration.z( )-CIMUInterpolator::m_vecBiasLinearAcceleration[2];
        const double dAbsoluteAccelerationX = std::fabs( dAccelerationX )-CIMUInterpolator::m_dImprecisionLinearAcceleration;
        const double dAbsoluteAccelerationY = std::fabs( dAccelerationY )-CIMUInterpolator::m_dImprecisionLinearAcceleration;
        const double dAbsoluteAccelerationZ = std::fabs( dAccelerationZ )-CIMUInterpolator::m_dImprecisionLinearAcceleration;

        CLinearAccelerationLEFT vecLinearAccelerationFiltered( 0.0, 0.0, 0.0 );

        if( 0 < dAbsoluteAccelerationX ){ vecLinearAccelerationFiltered.x( ) = CIMUInterpolator::sign( dAccelerationX )*dAbsoluteAccelerationX; }
        if( 0 < dAbsoluteAccelerationY ){ vecLinearAccelerationFiltered.y( ) = CIMUInterpolator::sign( dAccelerationY )*dAbsoluteAccelerationY; }
        if( 0 < dAbsoluteAccelerationZ ){ vecLinearAccelerationFiltered.z( ) = CIMUInterpolator::sign( dAccelerationZ )*dAbsoluteAccelerationZ; }

        std::printf( "(getLinearAccelerationFiltered) DEPRECATED\n" );

        return vecLinearAccelerationFiltered;
    }

    static const CLinearAccelerationWORLD getLinearAccelerationFiltered( const CLinearAccelerationInIMUFrame& p_vecLinearAcceleration, const Eigen::Isometry3d& p_matTransformationIMUtoWORLD )
    {
        //ds get the vector to the world frame in order to take care of gravity
        const CLinearAccelerationWORLD vecLinearAcceleration( p_matTransformationIMUtoWORLD.linear( )*p_vecLinearAcceleration );

        const double dAccelerationX = vecLinearAcceleration.x( )-CIMUInterpolator::m_vecBiasLinearAcceleration[0];
        const double dAccelerationY = vecLinearAcceleration.y( )-CIMUInterpolator::m_vecBiasLinearAcceleration[1];
        const double dAccelerationZ = vecLinearAcceleration.z( )-CIMUInterpolator::m_vecBiasLinearAcceleration[2];

        const double dAbsoluteAccelerationX = std::fabs( dAccelerationX )-CIMUInterpolator::m_dImprecisionLinearAcceleration;
        const double dAbsoluteAccelerationY = std::fabs( dAccelerationY )-CIMUInterpolator::m_dImprecisionLinearAcceleration;
        const double dAbsoluteAccelerationZ = std::fabs( dAccelerationZ )-CIMUInterpolator::m_dImprecisionLinearAcceleration;

        CLinearAccelerationWORLD vecLinearAccelerationFiltered( 0.0, 0.0, 0.0 );

        if( 0 < dAbsoluteAccelerationX ){ vecLinearAccelerationFiltered.x( ) = CIMUInterpolator::sign( dAccelerationX )*dAbsoluteAccelerationX; }
        if( 0 < dAbsoluteAccelerationY ){ vecLinearAccelerationFiltered.y( ) = CIMUInterpolator::sign( dAccelerationY )*dAbsoluteAccelerationY; }
        if( 0 < dAbsoluteAccelerationZ ){ vecLinearAccelerationFiltered.z( ) = CIMUInterpolator::sign( dAccelerationZ )*dAbsoluteAccelerationZ; }

        return vecLinearAccelerationFiltered;
    }

    static const CAngularVelocityInIMUFrame getAngularVelocityFiltered( const CAngularVelocityInIMUFrame& p_vecAngularVelocity )
    {
        //ds filter imprecision and bias
        const double dRotationX = p_vecAngularVelocity.x( );
        const double dRotationY = p_vecAngularVelocity.y( );
        const double dRotationZ = p_vecAngularVelocity.z( );
        const double dAbsoluteRotationX = std::fabs( dRotationX )-CIMUInterpolator::m_dImprecisionAngularVelocity;
        const double dAbsoluteRotationY = std::fabs( dRotationY )-CIMUInterpolator::m_dImprecisionAngularVelocity;
        const double dAbsoluteRotationZ = std::fabs( dRotationZ )-CIMUInterpolator::m_dImprecisionAngularVelocity;

        //ds noise free vectors
        CAngularVelocityInCameraFrame vecAngularVelocityFiltered( 0.0, 0.0, 0.0 );

        //ds update only if meaningful
        if( 0 < dAbsoluteRotationX ){ vecAngularVelocityFiltered.x( ) = CIMUInterpolator::sign( dRotationX )*dAbsoluteRotationX; }
        if( 0 < dAbsoluteRotationY ){ vecAngularVelocityFiltered.y( ) = CIMUInterpolator::sign( dRotationY )*dAbsoluteRotationY; }
        if( 0 < dAbsoluteRotationZ ){ vecAngularVelocityFiltered.z( ) = CIMUInterpolator::sign( dRotationZ )*dAbsoluteRotationZ; }

        return vecAngularVelocityFiltered;
    }

    //ds nasty ghastly hacky function
    inline static int8_t sign( const double& p_fNumber )
    {
        assert( 0.0 != p_fNumber );

        if( 0.0 < p_fNumber ){ return 1; }
        if( 0.0 > p_fNumber ){ return -1; }

        //ds never gets called, just pleasing the compiler
        assert( false );
        return 0;
    }

};

#endif //CIMUINTERPOLATOR_H
