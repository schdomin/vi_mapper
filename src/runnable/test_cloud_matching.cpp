#include <iostream>
#include <qapplication.h>

//ds custom
#include "utility/CCloudStreamer.h"
#include "utility/CCloudMatcher.h"
#include "vision/CMiniVisionToolbox.h"
#include "configuration/CConfigurationCamera.h"
#include "gui/CViewerCloud.h"
#include "vision/CStereoCamera.h"

int main( int argc, char** argv )
{
    std::printf( "(main) launched: %s\n", argv[0] );

    //ds if no clouds provided (at least two)
    if( 3 > argc )
    {
        std::printf( "(main) insufficient amount of clouds (provide at least 2)\n" );
        std::printf( "(main) calling syntax: test_cloud_matching <cloud> <cloud>\n" );
        std::printf( "(main) terminated: %s\n", argv[0] );
        std::fflush( stdout);
        return 0;
    }

    //ds clouds
    std::vector< CDescriptorVectorPointCloud > vecClouds;

    try
    {
        //ds load clouds
        while( 1 < argc-vecClouds.size( ) )
        {
            std::printf( "(main) loading cloud: %s - ", argv[vecClouds.size( )+1] );
            vecClouds.push_back( CCloudstreamer::loadCloud( argv[vecClouds.size( )+1] ) );
            std::printf( "points: %lu\n", vecClouds.back( ).vecPoints.size( ) );
        }
    }
    catch( const CExceptionInvalidFile& p_cException )
    {
        std::printf( "\n(main) unable to load cloud, exception: '%s'\n", p_cException.what( ) );
        std::printf( "(main) terminated: %s\n", argv[0] );
        std::fflush( stdout);
        return 0;
    }

    std::printf( "(main) successfully loaded %lu clouds\n", vecClouds.size( ) );

    //ds start the qt application
    QApplication cApplicationQT( argc, argv );

    //ds camera handle
    //const CStereoCamera cCameraSTEREO( CConfigurationCamera::LEFT::cPinholeCamera, CConfigurationCamera::RIGHT::cPinholeCamera );

    //ds match the query cloud (last one)
    const CDescriptorVectorPointCloud cCloudQuery = vecClouds.back( );

    //ds against one training cloud each (also itself for consistency purposes)
    for( const CDescriptorVectorPointCloud& cCloudReference: vecClouds )
    {
        //ds get matches
        const std::shared_ptr< const std::vector< CMatchCloud > > vecMatches( CCloudMatcher::getMatches( cCloudQuery.vecPoints, cCloudReference.vecPoints ) );
        const uint32_t uMinimumInliers = 10;

        std::printf( "(main) clouds [%06lu] > [%06lu] matches: %3lu | ", cCloudQuery.uID, cCloudReference.uID, vecMatches->size( ) );

        if( uMinimumInliers < vecMatches->size( ) )
        {
            /*Eigen::Vector3d vecTranslationToClosure( Eigen::Vector3d::Zero( ) ); //( cCloudReference.matTransformationLEFTtoWORLD.inverse( )*cCloudQuery.matTransformationLEFTtoWORLD ).translation( ) );
            Eigen::Isometry3d matTransformationToClosureInitial( cCloudReference.matTransformationLEFTtoWORLD.inverse( )*cCloudQuery.matTransformationLEFTtoWORLD );
            matTransformationToClosureInitial.translation( ) = vecTranslationToClosure;

            std::printf( "t: %4.1f %4.1f %4.1f > ", vecTranslationToClosure.x( ), vecTranslationToClosure.y( ), vecTranslationToClosure.z( ) );

            //ds 1mm for convergence
            const double dErrorDeltaForConvergence      = 1e-5;
            double dErrorSquaredTotalPrevious           = 0.0;
            const double dMaximumErrorForInlier         = 0.5;
            const double dMaximumErrorAverageForClosure = 0.4;
            const uint32_t uMinimumInliers              = 10;

            //ds LS setup - the jacobian is constant for this setup (and the transposed equals the regular)
            Eigen::Matrix< double, 3, 3 > matH;
            Eigen::Matrix< double, 3, 1 > vecB;
            Eigen::Matrix3d matOmega( Eigen::Matrix3d::Identity( ) );
            const Eigen::Matrix< double, 3, 3 > matJacobian( Eigen::Matrix3d::Identity( ) );

            //ds run least-squares maximum 100 times
            for( uint8_t uLS = 0; uLS < 100; ++uLS )
            {
                //ds error
                double dErrorSquaredTotal = 0.0;
                uint32_t uInliers         = 0;

                //ds initialize setup
                matH.setZero( );
                vecB.setZero( );

                //ds for all the points
                for( const CMatchCloud& cMatch: *vecMatches )
                {
                    //ds compute projection into closure
                    const CPoint3DCAMERA vecPointXYZQuery( cMatch.cPointQuery.vecPointXYZCAMERA+vecTranslationToClosure );
                    if( 0.0 < vecPointXYZQuery.z( ) )
                    {
                        assert( 0.0 < cMatch.cPointReference.vecPointXYZCAMERA.z( ) );

                        //ds adjust omega to inverse depth value (the further away the point, the less weight)
                        matOmega(2,2) = 1.0/( cMatch.cPointReference.vecPointXYZCAMERA.z( ) );

                        //ds compute error
                        const Eigen::Vector3d vecError( vecPointXYZQuery-cMatch.cPointReference.vecPointXYZCAMERA );

                        //ds update chi
                        const double dErrorSquared = vecError.transpose( )*matOmega*vecError;

                        //ds check if outlier
                        double dWeight = 1.0;
                        if( dMaximumErrorForInlier < dErrorSquared )
                        {
                            dWeight = dMaximumErrorForInlier/dErrorSquared;
                        }
                        else
                        {
                            ++uInliers;
                        }
                        dErrorSquaredTotal += dWeight*dErrorSquared;

                        //ds accumulate
                        matH += dWeight*matJacobian*matOmega*matJacobian;
                        vecB += dWeight*matJacobian*matOmega*vecError;
                    }
                }

                //ds solve the system and update the estimate
                vecTranslationToClosure += matH.ldlt( ).solve( -vecB );

                //ds check if converged (no descent required)
                if( dErrorDeltaForConvergence > std::fabs( dErrorSquaredTotalPrevious-dErrorSquaredTotal ) )
                {
                    //ds compute average error
                    const double dErrorAverage = dErrorSquaredTotal/(1+uInliers);

                    //ds final transform
                    Eigen::Isometry3d matTransformationToClosureFinal( matTransformationToClosureInitial );
                    matTransformationToClosureFinal.translation( ) = vecTranslationToClosure;

                    //ds if the solution is acceptable
                    if( dMaximumErrorAverageForClosure > dErrorAverage && uMinimumInliers < uInliers )
                    {
                        std::printf( "%4.1f %4.1f %4.1f | ", vecTranslationToClosure.x( ), vecTranslationToClosure.y( ), vecTranslationToClosure.z( ) );
                        std::printf( "system converged in %2u iterations, average error: %5.3f (inliers: %2u) - MATCH\n", uLS, dErrorAverage, uInliers );

                        //ds display result
                        CViewerCloud cViewer( vecMatches, matTransformationToClosureInitial, matTransformationToClosureFinal );
                        cApplicationQT.exec( );
                        break;
                    }
                    else
                    {
                        //ds keep looping through keyframes
                        std::printf( "%4.1f %4.1f %4.1f | ", vecTranslationToClosure.x( ), vecTranslationToClosure.y( ), vecTranslationToClosure.z( ) );
                        std::printf( "system converged in %2u iterations, average error: %5.3f (inliers: %2u) - discarded\n", uLS, dErrorAverage, uInliers );

                        //ds display result
                        CViewerCloud cViewer( vecMatches, matTransformationToClosureInitial, matTransformationToClosureFinal );
                        cApplicationQT.exec( );
                        break;
                    }
                }
                else
                {
                    dErrorSquaredTotalPrevious = dErrorSquaredTotal;
                }

                //ds if not converged
                if( 99 == uLS )
                {
                    std::printf( "system did not converge\n" );
                }
            }*/






            /*ds translation between this keyframe and the loop closure one (take current measurement as prior)
            Eigen::Vector3d vecTranslationToClosure( ( cCloudReference.matTransformationLEFTtoWORLD.inverse( )*cCloudQuery.matTransformationLEFTtoWORLD ).translation( ) );
            //Eigen::Isometry3d matTransformationToCLOSURE( Eigen::Matrix4d::Identity( ) );

            std::printf( "t: %4.1f %4.1f %4.1f > ", vecTranslationToClosure.x( ), vecTranslationToClosure.y( ), vecTranslationToClosure.z( ) );

            //ds 1mm for convergence
            const double dErrorDeltaForConvergence      = 1e-5;
            double dErrorSquaredTotalPrevious           = 0.0;
            const double dMaximumErrorForInlier         = 0.25;
            const double dMaximumErrorAverageForClosure = 0.25;
            const uint32_t uMinimumInliers              = 10;

            //ds LS setup - the jacobian is constant for this setup (and the transposed equals the regular)
            Eigen::Matrix< double, 3, 3 > matH;
            Eigen::Matrix< double, 3, 1 > vecB;
            Eigen::Matrix3d matOmega( Eigen::Matrix3d::Identity( ) );
            const Eigen::Matrix< double, 3, 3 > matJacobian( Eigen::Matrix3d::Identity( ) );

            //ds run least-squares maximum 100 times
            for( uint8_t uLS = 0; uLS < 100; ++uLS )
            {
                //ds error
                double dErrorSquaredTotal = 0.0;
                uint32_t uOutliers        = 0;

                //ds initialize setup
                matH.setZero( );
                vecB.setZero( );

                std::vector< CMatchCloud > vecInliers;

                //ds for all the points
                for( const CMatchCloud& cMatch: *vecMatches )
                {
                    //ds compute projection into closure
                    const CPoint3DCAMERA vecPointXYZQuery( cMatch.cPointQuery.vecPointXYZCAMERA+vecTranslationToClosure );
                    if( 0.0 < vecPointXYZQuery.z( ) )
                    {
                        assert( 0.0 < cMatch.cPointReference.vecPointXYZCAMERA.z( ) );

                        //ds adjust omega to inverse depth value (the further away the point, the less weight)
                        matOmega(2,2) = 1.0/( cMatch.cPointReference.vecPointXYZCAMERA.z( ) );

                        //ds compute error
                        const Eigen::Vector3d vecError( vecPointXYZQuery-cMatch.cPointReference.vecPointXYZCAMERA );

                        //ds update chi
                        const double dErrorSquared = vecError.transpose( )*matOmega*vecError;

                        //ds check if outlier
                        double dWeight = 1.0;
                        if( dMaximumErrorForInlier < dErrorSquared )
                        {
                            dWeight = dMaximumErrorForInlier/dErrorSquared;
                            ++uOutliers;
                        }
                        else
                        {
                            vecInliers.push_back( cMatch );
                        }
                        dErrorSquaredTotal += dWeight*dErrorSquared;

                        //ds accumulate
                        matH += dWeight*matJacobian*matOmega*matJacobian;
                        vecB += dWeight*matJacobian*matOmega*vecError;
                    }
                    else
                    {
                        ++uOutliers;
                    }
                }

                //ds solve the system and update the estimate
                vecTranslationToClosure += matH.ldlt( ).solve( -vecB );

                //ds check if converged (no descent required)
                if( dErrorDeltaForConvergence > std::fabs( dErrorSquaredTotalPrevious-dErrorSquaredTotal ) )
                {
                    //ds compute average error
                    const double dErrorAverage = dErrorSquaredTotal/vecMatches->size( );
                    const uint32_t uInliers    = vecMatches->size( )-uOutliers;

                    //ds if the solution is acceptable
                    if( dMaximumErrorAverageForClosure > dErrorAverage && uMinimumInliers < uInliers )
                    {
                        std::printf( "%4.1f %4.1f %4.1f | ", vecTranslationToClosure.x( ), vecTranslationToClosure.y( ), vecTranslationToClosure.z( ) );
                        std::printf( "system converged in %2u iterations, average error: %5.3f (outliers: %2u) - MATCH\n", uLS, dErrorAverage, uOutliers );



                        //ds transformation between this keyframe and the loop closure one (take current measurement as prior)
                        Eigen::Isometry3d matTransformationToCLOSURE( cCloudReference.matTransformationLEFTtoWORLD.inverse( )*cCloudQuery.matTransformationLEFTtoWORLD );
                        matTransformationToCLOSURE.translation( ) = vecTranslationToClosure;

                        //ds 1mm for convergence
                        double dErrorSquaredTotalPrevious           = 0.0;
                        const double dMaximumErrorAverageForClosure = 200.0;

                        //ds LS setup
                        Eigen::Matrix< double, 6, 6 > matH;
                        Eigen::Matrix< double, 6, 1 > vecB;

                        //ds run least-squares maximum 100 times
                        for( uint8_t uLS = 0; uLS < 100; ++uLS )
                        {
                            //ds error
                            double dErrorSquaredTotal = 0.0;
                            uint32_t uOutliersInner = 0;

                            //ds initialize setup
                            matH.setZero( );
                            vecB.setZero( );

                            //ds for all the points
                            for( const CMatchCloud& cMatch: vecInliers )
                            {
                                //ds compute projection into closure
                                const CPoint3DCAMERA vecPointXYZQuery( matTransformationToCLOSURE*cMatch.cPointQuery.vecPointXYZCAMERA );
                                const double dDepthQuery = vecPointXYZQuery.z( );
                                if( 0.0 < dDepthQuery )
                                {
                                    assert( 0.0 < cMatch.cPointReference.vecPointXYZCAMERA.z( ) );

                                    //ds compute error - apply the projection to the transformed point
                                    Eigen::Vector4d vecPointHomogeneous( vecPointXYZQuery.x( ), vecPointXYZQuery.y( ), vecPointXYZQuery.z( ), 1.0 );
                                    Eigen::Vector3d pp1LEFT = cCameraSTEREO.m_pCameraLEFT->m_matProjection*vecPointHomogeneous;
                                    Eigen::Vector3d pp1RIGHT = cCameraSTEREO.m_pCameraRIGHT->m_matProjection*vecPointHomogeneous;
                                    Eigen::Vector2d ppLEFT (pp1LEFT.x()/pp1LEFT.z(), pp1LEFT.y()/pp1LEFT.z());
                                    Eigen::Vector2d ppRIGHT (pp1RIGHT.x()/pp1RIGHT.z(), pp1RIGHT.y()/pp1RIGHT.z());

                                    const Eigen::Vector4d vecError( ppLEFT.x( )-cMatch.cPointReference.ptUVLEFT.x,
                                                                    ppLEFT.y( )-cMatch.cPointReference.ptUVLEFT.y,
                                                                    ppRIGHT.x( )-cMatch.cPointReference.ptUVRIGHT.x,
                                                                    ppRIGHT.y( )-cMatch.cPointReference.ptUVRIGHT.y );

                                    //ds depth affects disparity (y)
                                    //matOmega(1,1) = 1.0/cMatch.cPointReference.vecPointXYZCAMERA.z( );
                                    //matOmega(3,3) = matOmega(1,1);

                                    //ds update chi
                                    dErrorSquaredTotal += vecError.transpose( )*vecError;

                                    //ds get the jacobian of the transform part = [I 2*skew(T*modelPoint)]
                                    Eigen::Matrix< double, 4, 6 > matJacobianTransform;
                                    matJacobianTransform.setZero( );
                                    matJacobianTransform.block<3,3>(0,0).setIdentity( );
                                    matJacobianTransform.block<3,3>(0,3) = -2*CMiniVisionToolbox::getSkew( vecPointXYZQuery );

                                    // jacobian of the homogeneous division
                                    // 1/z   0    -x/z^2
                                    // 0     1/z  -y/z^2
                                    Eigen::Matrix<double, 2, 3> JpLEFT;
                                    JpLEFT <<
                                      1/dDepthQuery, 0,   -pp1LEFT.x()/(dDepthQuery*dDepthQuery),
                                      0,   1/dDepthQuery, -pp1LEFT.y()/(dDepthQuery*dDepthQuery);

                                    Eigen::Matrix<double, 2, 3> JpRIGHT;
                                    JpRIGHT <<
                                      1/dDepthQuery, 0,   -pp1RIGHT.x()/(dDepthQuery*dDepthQuery),
                                      0,   1/dDepthQuery, -pp1RIGHT.y()/(dDepthQuery*dDepthQuery);

                                    //ds final jacobian
                                    Eigen::Matrix< double, 4, 6 > matJacobian;
                                    matJacobian.setZero( );
                                    matJacobian.block< 2,6 >(0,0) = JpLEFT*cCameraSTEREO.m_pCameraLEFT->m_matProjection*matJacobianTransform;
                                    matJacobian.block< 2,6 >(2,0) = JpRIGHT*cCameraSTEREO.m_pCameraRIGHT->m_matProjection*matJacobianTransform;

                                    //ds precompute transposed
                                    const Eigen::Matrix< double, 6, 4 > matJacobianTransposed( matJacobian.transpose( ) );

                                    //ds accumulate
                                    matH += matJacobianTransposed*matJacobian;
                                    vecB += matJacobianTransposed*vecError;
                                }
                                else
                                {
                                    ++uOutliersInner;
                                }
                            }

                            //ds solve the system and update the estimate
                            matTransformationToCLOSURE = CMiniVisionToolbox::getTransformationFromVector( matH.ldlt( ).solve( -vecB ) )*matTransformationToCLOSURE;

                            //ds enforce rotation symmetry
                            const Eigen::Matrix3d matRotation        = matTransformationToCLOSURE.linear( );
                            Eigen::Matrix3d matRotationSquared       = matRotation.transpose( )*matRotation;
                            matRotationSquared.diagonal( ).array( ) -= 1;
                            matTransformationToCLOSURE.linear( )    -= 0.5*matRotation*matRotationSquared;

                            //ds check if converged (no descent required)
                            if( dErrorDeltaForConvergence > std::fabs( dErrorSquaredTotalPrevious-dErrorSquaredTotal ) )
                            {
                                //ds compute average error
                                const uint32_t uInliersInner = vecInliers.size( )-uOutliersInner;
                                const double dErrorAverage = dErrorSquaredTotal/uInliersInner;

                                //ds if the solution is acceptable
                                if( dMaximumErrorAverageForClosure > dErrorAverage )
                                {
                                    std::printf( "                                                 second translation: %4.1f %4.1f %4.1f | ", matTransformationToCLOSURE.translation( ).x( ), matTransformationToCLOSURE.translation( ).y( ), matTransformationToCLOSURE.translation( ).z( ) );
                                    std::printf( "system converged in %2u iterations, average error: %5.3f (inliers: %2u) - MATCH\n", uLS, dErrorAverage, uInliersInner );
                                    break;
                                }
                                else
                                {
                                    //ds keep looping through keyframes
                                    std::printf( "                                                 second translation: %4.1f %4.1f %4.1f | ", matTransformationToCLOSURE.translation( ).x( ), matTransformationToCLOSURE.translation( ).y( ), matTransformationToCLOSURE.translation( ).z( ) );
                                    std::printf( "system converged in %2u iterations, average error: %5.3f (inliers: %2u) - discarded\n", uLS, dErrorAverage, uInliersInner );
                                    break;
                                }
                            }
                            else
                            {
                                dErrorSquaredTotalPrevious = dErrorSquaredTotal;
                            }

                            //ds if not converged
                            if( 99 == uLS )
                            {
                                std::printf( "system did not converge\n" );
                            }
                        }
                        break;
                    }
                    else
                    {
                        //ds keep looping through keyframes
                        std::printf( "%4.1f %4.1f %4.1f | ", vecTranslationToClosure.x( ), vecTranslationToClosure.y( ), vecTranslationToClosure.z( ) );
                        std::printf( "system converged in %2u iterations, average error: %5.3f (outliers: %2u) - discarded\n", uLS, dErrorAverage, uOutliers );
                        break;
                    }
                }
                else
                {
                    dErrorSquaredTotalPrevious = dErrorSquaredTotal;
                }

                //ds if not converged
                if( 99 == uLS )
                {
                    std::printf( "system did not converge\n" );
                }
            }*/




            /*
            //ds transformation between this keyframe and the loop closure one (take current measurement as prior)
            Eigen::Isometry3d matTransformationToCLOSURE( cCloudReference.matTransformationLEFTtoWORLD.inverse( )*cCloudQuery.matTransformationLEFTtoWORLD );
            //Eigen::Isometry3d matTransformationToCLOSURE( Eigen::Matrix4d::Identity( ) );

            const CPoint3DWORLD vecTranslationInitial( matTransformationToCLOSURE.translation( ) );
            std::printf( "t: %4.1f %4.1f %4.1f > ", vecTranslationInitial.x( ), vecTranslationInitial.y( ), vecTranslationInitial.z( ) );

            //ds 1mm for convergence
            const double dErrorDeltaForConvergence      = 1e-5;
            double dErrorSquaredTotalPrevious           = 0.0;
            const double dMaximumErrorForInlier         = 10.0;
            const double dMaximumErrorAverageForClosure = 10.0;
            const uint32_t uMinimumInliers              = 10;

            //ds LS setup
            Eigen::Matrix< double, 6, 6 > matH;
            Eigen::Matrix< double, 6, 1 > vecB;
            Eigen::Matrix4d matOmega( Eigen::Matrix4d::Identity( ) );

            //ds run least-squares maximum 100 times
            for( uint8_t uLS = 0; uLS < 100; ++uLS )
            {
                //ds error
                double dErrorSquaredTotal = 0.0;
                uint32_t uOutliers        = 0;

                //ds initialize setup
                matH.setZero( );
                vecB.setZero( );

                //ds for all the points
                for( const CMatchCloud& cMatch: *vecMatches )
                {
                    //ds compute projection into closure
                    const CPoint3DCAMERA vecPointXYZQuery( matTransformationToCLOSURE*cMatch.cPointQuery.vecPointXYZCAMERA );
                    const double dDepthQuery = vecPointXYZQuery.z( );
                    if( 0.0 < dDepthQuery )
                    {
                        assert( 0.0 < cMatch.cPointReference.vecPointXYZCAMERA.z( ) );

                        //ds compute error - apply the projection to the transformed point
                        Eigen::Vector4d vecPointHomogeneous( vecPointXYZQuery.x( ), vecPointXYZQuery.y( ), vecPointXYZQuery.z( ), 1.0 );
                        Eigen::Vector3d pp1LEFT = cCameraSTEREO.m_pCameraLEFT->m_matProjection*vecPointHomogeneous;
                        Eigen::Vector3d pp1RIGHT = cCameraSTEREO.m_pCameraRIGHT->m_matProjection*vecPointHomogeneous;
                        Eigen::Vector2d ppLEFT (pp1LEFT.x()/pp1LEFT.z(), pp1LEFT.y()/pp1LEFT.z());
                        Eigen::Vector2d ppRIGHT (pp1RIGHT.x()/pp1RIGHT.z(), pp1RIGHT.y()/pp1RIGHT.z());

                        const Eigen::Vector4d vecError( ppLEFT.x( )-cMatch.cPointReference.ptUVLEFT.x,
                                                        ppLEFT.y( )-cMatch.cPointReference.ptUVLEFT.y,
                                                        ppRIGHT.x( )-cMatch.cPointReference.ptUVRIGHT.x,
                                                        ppRIGHT.y( )-cMatch.cPointReference.ptUVRIGHT.y );

                        //ds depth affects disparity (y)
                        matOmega(1,1) = 1.0/cMatch.cPointReference.vecPointXYZCAMERA.z( );
                        matOmega(3,3) = matOmega(1,1);

                        //ds update chi
                        const double dErrorSquared = vecError.transpose( )*matOmega*vecError;

                        //ds check if outlier
                        double dWeight = 1.0;
                        if( dMaximumErrorForInlier < dErrorSquared )
                        {
                            dWeight = dMaximumErrorForInlier/dErrorSquared;
                            ++uOutliers;
                        }
                        dErrorSquaredTotal += dWeight*dErrorSquared;

                        //ds get the jacobian of the transform part = [I 2*skew(T*modelPoint)]
                        Eigen::Matrix< double, 4, 6 > matJacobianTransform;
                        matJacobianTransform.setZero( );
                        matJacobianTransform.block<3,3>(0,0).setIdentity( );
                        matJacobianTransform.block<3,3>(0,3) = -2*CMiniVisionToolbox::getSkew( vecPointXYZQuery );

                        // jacobian of the homogeneous division
                        // 1/z   0    -x/z^2
                        // 0     1/z  -y/z^2
                        Eigen::Matrix<double, 2, 3> JpLEFT;
                        JpLEFT <<
                          1/dDepthQuery, 0,   -pp1LEFT.x()/(dDepthQuery*dDepthQuery),
                          0,   1/dDepthQuery, -pp1LEFT.y()/(dDepthQuery*dDepthQuery);

                        Eigen::Matrix<double, 2, 3> JpRIGHT;
                        JpRIGHT <<
                          1/dDepthQuery, 0,   -pp1RIGHT.x()/(dDepthQuery*dDepthQuery),
                          0,   1/dDepthQuery, -pp1RIGHT.y()/(dDepthQuery*dDepthQuery);

                        //ds final jacobian
                        Eigen::Matrix< double, 4, 6 > matJacobian;
                        matJacobian.setZero( );
                        matJacobian.block< 2,6 >(0,0) = JpLEFT*cCameraSTEREO.m_pCameraLEFT->m_matProjection*matJacobianTransform;
                        matJacobian.block< 2,6 >(2,0) = JpRIGHT*cCameraSTEREO.m_pCameraRIGHT->m_matProjection*matJacobianTransform;

                        //ds precompute transposed
                        const Eigen::Matrix< double, 6, 4 > matJacobianTransposed( matJacobian.transpose( ) );

                        //ds accumulate
                        matH += dWeight*matJacobianTransposed*matOmega*matJacobian;
                        vecB += dWeight*matJacobianTransposed*matOmega*vecError;
                    }
                    else
                    {
                        ++uOutliers;
                    }
                }

                //ds solve the system and update the estimate
                matTransformationToCLOSURE = CMiniVisionToolbox::getTransformationFromVector( matH.ldlt( ).solve( -vecB ) )*matTransformationToCLOSURE;

                //ds enforce rotation symmetry
                const Eigen::Matrix3d matRotation        = matTransformationToCLOSURE.linear( );
                Eigen::Matrix3d matRotationSquared       = matRotation.transpose( )*matRotation;
                matRotationSquared.diagonal( ).array( ) -= 1;
                matTransformationToCLOSURE.linear( )    -= 0.5*matRotation*matRotationSquared;

                //ds check if converged (no descent required)
                if( dErrorDeltaForConvergence > std::fabs( dErrorSquaredTotalPrevious-dErrorSquaredTotal ) )
                {
                    //ds compute average error
                    const double dErrorAverage = dErrorSquaredTotal/vecMatches->size( );
                    const uint32_t uInliers    = vecMatches->size( )-uOutliers;

                    //ds if the solution is acceptable
                    if( dMaximumErrorAverageForClosure > dErrorAverage && uMinimumInliers < uInliers )
                    {
                        std::printf( "%4.1f %4.1f %4.1f | ", matTransformationToCLOSURE.translation( ).x( ), matTransformationToCLOSURE.translation( ).y( ), matTransformationToCLOSURE.translation( ).z( ) );
                        std::printf( "system converged in %2u iterations, average error: %5.3f (outliers: %2u) - MATCH\n", uLS, dErrorAverage, uOutliers );
                        break;
                    }
                    else
                    {
                        //ds keep looping through keyframes
                        std::printf( "%4.1f %4.1f %4.1f | ", matTransformationToCLOSURE.translation( ).x( ), matTransformationToCLOSURE.translation( ).y( ), matTransformationToCLOSURE.translation( ).z( ) );
                        std::printf( "system converged in %2u iterations, average error: %5.3f (outliers: %2u) - discarded\n", uLS, dErrorAverage, uOutliers );
                        break;
                    }
                }
                else
                {
                    dErrorSquaredTotalPrevious = dErrorSquaredTotal;
                }

                //ds if not converged
                if( 99 == uLS )
                {
                    std::printf( "system did not converge\n" );
                }
            }




*/

            //ds transformation between this keyframe and the loop closure one (take current measurement as prior)
            Eigen::Isometry3d matTransformationToCLOSURE( cCloudReference.matTransformationLEFTtoWORLD.inverse( )*cCloudQuery.matTransformationLEFTtoWORLD );
            //Eigen::Isometry3d matTransformationToCLOSURE( Eigen::Matrix4d::Identity( ) );
            matTransformationToCLOSURE.translation( ) = Eigen::Vector3d::Zero( );
            const Eigen::Isometry3d matTransformationToClosureInitial( matTransformationToCLOSURE );

            //ds 1mm for convergence
            const double dErrorDeltaForConvergence      = 1e-5;
            double dErrorSquaredTotalPrevious           = 0.0;
            const double dMaximumErrorForInlier         = 0.25;
            const double dMaximumErrorAverageForClosure = 0.2;

            //ds LS setup
            Eigen::Matrix< double, 6, 6 > matH;
            Eigen::Matrix< double, 6, 1 > vecB;
            Eigen::Matrix3d matOmega( Eigen::Matrix3d::Identity( ) );

            std::printf( "t: %4.1f %4.1f %4.1f > ", matTransformationToCLOSURE.translation( ).x( ), matTransformationToCLOSURE.translation( ).y( ), matTransformationToCLOSURE.translation( ).z( ) );

            //ds run least-squares maximum 100 times
            for( uint8_t uLS = 0; uLS < 100; ++uLS )
            {
                //ds error
                double dErrorSquaredTotal = 0.0;
                uint32_t uInliers         = 0;

                //ds initialize setup
                matH.setZero( );
                vecB.setZero( );

                //ds for all the points
                for( const CMatchCloud& cMatch: *vecMatches )
                {
                    //ds compute projection into closure
                    const CPoint3DCAMERA vecPointXYZQuery( matTransformationToCLOSURE*cMatch.cPointQuery.vecPointXYZCAMERA );
                    if( 0.0 < vecPointXYZQuery.z( ) )
                    {
                        assert( 0.0 < cMatch.cPointReference.vecPointXYZCAMERA.z( ) );

                        //ds adjust omega to inverse depth value (the further away the point, the less weight)
                        matOmega(2,2) = 1.0/( cMatch.cPointReference.vecPointXYZCAMERA.z( ) );

                        //ds compute error
                        const Eigen::Vector3d vecError( vecPointXYZQuery-cMatch.cPointReference.vecPointXYZCAMERA );

                        //ds update chi
                        const double dErrorSquared = vecError.transpose( )*matOmega*vecError;

                        //ds check if outlier
                        double dWeight = 1.0;
                        if( dMaximumErrorForInlier < dErrorSquared )
                        {
                            dWeight = dMaximumErrorForInlier/dErrorSquared;
                        }
                        else
                        {
                            ++uInliers;
                        }
                        dErrorSquaredTotal += dWeight*dErrorSquared;

                        //ds get the jacobian of the transform part = [I 2*skew(T*modelPoint)]
                        Eigen::Matrix< double, 3, 6 > matJacobianTransform;
                        matJacobianTransform.setZero( );
                        matJacobianTransform.block<3,3>(0,0).setIdentity( );
                        matJacobianTransform.block<3,3>(0,3) = -2*CMiniVisionToolbox::getSkew( vecPointXYZQuery );

                        //ds precompute transposed
                        const Eigen::Matrix< double, 6, 3 > matJacobianTransformTransposed( matJacobianTransform.transpose( ) );

                        //ds accumulate
                        matH += dWeight*matJacobianTransformTransposed*matOmega*matJacobianTransform;
                        vecB += dWeight*matJacobianTransformTransposed*matOmega*vecError;
                    }
                }

                //ds solve the system and update the estimate
                matTransformationToCLOSURE = CMiniVisionToolbox::getTransformationFromVector( matH.ldlt( ).solve( -vecB ) )*matTransformationToCLOSURE;

                //ds enforce rotation symmetry
                const Eigen::Matrix3d matRotation        = matTransformationToCLOSURE.linear( );
                Eigen::Matrix3d matRotationSquared       = matRotation.transpose( )*matRotation;
                matRotationSquared.diagonal( ).array( ) -= 1;
                matTransformationToCLOSURE.linear( )    -= 0.5*matRotation*matRotationSquared;

                //ds check if converged (no descent required)
                if( dErrorDeltaForConvergence > std::fabs( dErrorSquaredTotalPrevious-dErrorSquaredTotal ) )
                {
                    //ds compute average error
                    const double dErrorAverage = dErrorSquaredTotal/(1+uInliers);

                    //ds if the solution is acceptable
                    if( dMaximumErrorAverageForClosure > dErrorAverage && uMinimumInliers < uInliers )
                    {
                        std::printf( "%4.1f %4.1f %4.1f | ", matTransformationToCLOSURE.translation( ).x( ), matTransformationToCLOSURE.translation( ).y( ), matTransformationToCLOSURE.translation( ).z( ) );
                        std::printf( "system converged in %2u iterations, average error: %5.3f (inliers: %2u) - MATCH\n", uLS, dErrorAverage, uInliers );

                        //ds display result
                        CViewerCloud cViewer( vecMatches, matTransformationToClosureInitial, matTransformationToCLOSURE );
                        cApplicationQT.exec( );
                        break;
                    }
                    else
                    {
                        //ds keep looping through keyframes
                        std::printf( "%4.1f %4.1f %4.1f | ", matTransformationToCLOSURE.translation( ).x( ), matTransformationToCLOSURE.translation( ).y( ), matTransformationToCLOSURE.translation( ).z( ) );
                        std::printf( "system converged in %2u iterations, average error: %5.3f (inliers: %2u) - discarded\n", uLS, dErrorAverage, uInliers );

                        //ds display result
                        CViewerCloud cViewer( vecMatches, matTransformationToClosureInitial, matTransformationToCLOSURE );
                        cApplicationQT.exec( );
                        break;
                    }
                }
                else
                {
                    dErrorSquaredTotalPrevious = dErrorSquaredTotal;
                }

                //ds if not converged
                if( 99 == uLS )
                {
                    std::printf( "system did not converge\n" );
                }
            }
        }
        else
        {
            std::printf( "NO CLOSURE\n" );
        }
    }



    /*ds against one training cloud each
    for( const CDescriptorPointCloud& cCloudTrain: vecClouds )
    {
        //ds if not the same cloud
        if( cCloudTrain.uID != cCloudQuery.uID )
        {
            //ds count matches
            UIDLandmark uMatches = 0;

            //ds try to match the query into the training
            for( const CDescriptorVectorPoint3DWORLD& cPointQuery: cCloudQuery.vecPoints )
            {
                //ds current matching distance
                double dMatchingDistanceBest           = 1e6;
                //UIDDescriptorPoint3D uIDQueryBest      = 0;
                //UIDDescriptor uDescriptorMatchingsBest = 0;

                //ds match this point against all training points
                for( const CDescriptorVectorPoint3DWORLD cPointTrain: cCloudTrain.vecPoints )
                {
                    //ds compute current matching distance against training point - EUCLIDIAN
                    double dMatchingDistance = dWeightEuclidian*( cPointQuery.vecPointXYZWORLD-cPointTrain.vecPointXYZWORLD ).squaredNorm( );

                    //ds get start and end iterators of training descriptors
                    std::vector< CDescriptor >::const_iterator itDescriptorTrainStart( cPointTrain.vecDescriptors.begin( ) );
                    std::vector< CDescriptor >::const_iterator itDescriptorTrainEnd( cPointTrain.vecDescriptors.end( )-1 );

                    double dMatchingDistanceHammingForward   = 0.0;
                    double dMatchingDistanceHammingBackwards = 0.0;
                    UIDDescriptor uDescriptorMatchings       = 0;

                    //ds compute descriptor matching in ascending versus descending order
                    for( const CDescriptor cDescriptorQuery: cPointQuery.vecDescriptors )
                    {
                        dMatchingDistanceHammingForward   += cv::norm( cDescriptorQuery, *itDescriptorTrainStart, cv::NORM_HAMMING );
                        dMatchingDistanceHammingBackwards += cv::norm( cDescriptorQuery, *itDescriptorTrainEnd, cv::NORM_HAMMING );
                        ++uDescriptorMatchings;

                        //ds move forward respective backwards
                        ++itDescriptorTrainStart;

                        //ds check if descriptor arrays overlap
                        if( cPointTrain.vecDescriptors.end( ) == itDescriptorTrainStart || cPointTrain.vecDescriptors.begin( ) == itDescriptorTrainEnd )
                        {
                            break;
                        }

                        //ds since begin returns still a valid element whereas end does not
                        --itDescriptorTrainEnd;
                    }

                    //ds average matchings
                    dMatchingDistanceHammingForward /= uDescriptorMatchings;
                    dMatchingDistanceHammingBackwards /= uDescriptorMatchings;

                    //std::printf( "matching forward: %f backwards: %f (matchings: %u)\n", dMatchingDistanceHammingForward, dMatchingDistanceHammingBackwards, uDescriptorMatchings );

                    //ds take the best run
                    if( dMatchingDistanceHammingForward < dMatchingDistanceHammingBackwards )
                    {
                        dMatchingDistance += dMatchingDistanceHammingForward;
                    }
                    else
                    {
                        dMatchingDistance += dMatchingDistanceHammingBackwards;
                    }

                    //ds check if better
                    if( dMatchingDistanceBest > dMatchingDistance )
                    {
                        dMatchingDistanceBest    = dMatchingDistance;
                        //uIDQueryBest             = cPointTrain.uID;
                        //uDescriptorMatchingsBest = uDescriptorMatchings;
                    }
                }

                //ds check if match
                if( dMatchingDistanceCutoff > dMatchingDistanceBest )
                {
                    ++uMatches;
                }

                //std::printf( "(main) best match for query: %03lu -> train: %03lu (matching distance: %9.4f, descriptor matchings: %03lu)\n", cPointQuery.uID, uIDQueryBest, dMatchingDistanceBest, uDescriptorMatchingsBest );
            }

            std::printf( "(main) cloud [%02lu] -> [%02lu] - matches: %06lu/%06lu\n", cCloudQuery.uID, cCloudTrain.uID, uMatches, cCloudQuery.vecPoints.size( ) );
        }
    }*/

    std::printf( "(main) terminated: %s\n", argv[0] );
    std::fflush( stdout);
    return 0;
}
