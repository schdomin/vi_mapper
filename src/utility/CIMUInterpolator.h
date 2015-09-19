#ifndef CIMUINTERPOLATOR_H
#define CIMUINTERPOLATOR_H

//#include <map>
#include "vision/CMiniVisionToolbox.h"

class CIMUInterpolator
{

//ds ctor/dtor
public:

    CIMUInterpolator( );
    ~CIMUInterpolator( );

//ds fields
private:

    std::vector< std::pair< CLinearAccelerationIMU, CAngularVelocityIMU > > m_vecCalibration;
    std::vector< std::pair< CLinearAccelerationIMU, CAngularVelocityIMU > >::size_type uMeasurementsForOptimization = 10; //10
    double m_dLastTimestamp                             = 0.0;
    const double m_dCalibrationConvergenceDeltaRotation = 1e-3;
    const double m_dCalibrationConvergenceDeltaFinal    = 1e-3;
    bool m_bIsCalibrated                                = false;
    bool m_bIsRotationComputed                          = false;

    Eigen::Vector3d m_vecOffsetAccelerationLinear;
    Eigen::Vector3d m_vecOffsetVelocityAngular;
    Eigen::Matrix3d m_matRotationIMUtoWORLD;
    Eigen::Matrix3d m_matRotationWORLDtoIMU;

public:

    //ds IMU filtering (calibrated 2015-07-25)
    static constexpr double m_dImprecisionAngularVelocity     = 0.01;
    static constexpr double m_dImprecisionLinearAcceleration  = 0.5; //0.1
    static constexpr double m_vecBiasLinearAccelerationXYZ[3] = { 0.0, 0.0, -9.80665 }; //ds compensate gravitational component (http://en.wikipedia.org/wiki/ISO_80000-3)
    static constexpr double dMaximumDeltaTimeSeconds          = 0.11;
    static constexpr double dMaximumDeltaTimeSecondsSquared   = 0.11*0.11;

//ds accessors
public:

    void addMeasurementCalibration( const CLinearAccelerationIMU& p_vecAccelerationLinear, const CAngularVelocityIMU& p_vecVelocityAngular );

    void calibrateOffsets( );
    void calibrateRotation( );

    const bool isCalibrated( ) const;
    const Eigen::Isometry3d getTransformationWORLDtoCAMERA( const Eigen::Matrix3d& p_matRotationIMUtoCAMERA ) const;

//ds statics
public:

    static const CLinearAccelerationWORLD getLinearAccelerationFiltered( const CLinearAccelerationWORLD& p_vecLinearAcceleration );
    static const CAngularVelocityIMU getAngularVelocityFiltered( const CAngularVelocityIMU& p_vecAngularVelocity );

    //ds nasty ghastly hacky function
    inline static const int8_t sign( const double& p_fNumber );

};

#endif //CIMUINTERPOLATOR_H
