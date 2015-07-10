#include <iostream>
#include <unistd.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Cholesky>
#include <Eigen/StdVector>
#include <vector>
#include <cstdio>
#include <fstream>

namespace gtools
{

using namespace std;

typedef std::vector< Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d > > Vector3dVector;
typedef std::vector< Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d > > Vector2dVector;
typedef Eigen::Matrix< double, 6, 6 > Matrix6d;
typedef Eigen::Matrix< double, 6, 1 > Vector6d;
typedef Eigen::Matrix< double, 2, 6 > Matrix2_6d;
typedef Eigen::Matrix< double, 4, 6 > Matrix4x6d;

struct CPositSolverStereo
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    CPositSolverStereo( const shared_ptr< CStereoCamera > p_pCameraSTEREO ): uImageWidth( p_pCameraSTEREO->m_uPixelWidth ),
                        uImageHeight( p_pCameraSTEREO->m_uPixelHeight ),
                        T( Eigen::Matrix4d::Identity( ) ),
                        matProjectionLEFT( p_pCameraSTEREO->m_pCameraLEFT->m_matProjection ),
                        matProjectionRIGHT( p_pCameraSTEREO->m_pCameraRIGHT->m_matProjection ),
                        uNumberOfInliers( 0 ),
                        uNumberOfReprojections( 0 ),
                        dMaximumError( 1e6 ),
                        dInlierThreshold( 10.0 )

    {
        //ds nothing to do
    }

    inline Eigen::Isometry3d v2t( const Vector6d& t )
    {
        Eigen::Isometry3d T;
        T.setIdentity();
        T.translation()=t.head<3>();
        float w=t.block<3,1>(3,0).squaredNorm();
        if (w<1)
        {
            w=sqrt(1-w);
            T.linear()=Eigen::Quaterniond(w, t(3), t(4), t(5)).toRotationMatrix();
        }
        else
        {
            T.linear().setIdentity();
        }
        return T;
    }

    inline Eigen::Matrix3d skew( const Eigen::Vector3d& p )
    {
        Eigen::Matrix3d s;
            s <<
            0,  -p.z(), p.y(),
            p.z(), 0,  -p.x(),
            -p.y(), p.x(), 0;
        return s;
    }


  Vector3dVector model_points;
  Vector2dVector vecProjectionsUVLEFT;
  Vector2dVector vecProjectionsUVRIGHT;
  
  float uImageWidth, uImageHeight;
  Eigen::Isometry3d T;
  Eigen::Matrix< double, 3, 4 > matProjectionLEFT;
  Eigen::Matrix< double, 3, 4 > matProjectionRIGHT;
  int uNumberOfInliers;
  int uNumberOfReprojections;
  float dMaximumError;
  float dInlierThreshold;
  std::vector<bool> inliers;

  inline bool errorAndJacobian(Eigen::Vector4d&  error, Matrix4x6d&  J, const Eigen::Vector3d& modelPoint, const Eigen::Vector2d& vecUVLEFT, const Eigen::Vector2d& vecUVRIGHT){
    J.setZero();
    // apply the transform to the point
    Eigen::Vector3d tp=T*modelPoint;

    // if the points is behind the camera, drop it;
    if (tp.z()<0)
      return false;

    float z = tp.z();

    // apply the projection to the transformed point
    Eigen::Vector4d vecPointHomogeneous( tp.x( ), tp.y( ), tp.z( ), 1.0 );
    Eigen::Vector3d pp1LEFT = matProjectionLEFT*vecPointHomogeneous;
    Eigen::Vector3d pp1RIGHT = matProjectionRIGHT*vecPointHomogeneous;
    Eigen::Vector2d ppLEFT (pp1LEFT.x()/pp1LEFT.z(), pp1LEFT.y()/pp1LEFT.z());
    Eigen::Vector2d ppRIGHT (pp1RIGHT.x()/pp1RIGHT.z(), pp1RIGHT.y()/pp1RIGHT.z());

    // if the projected point is outside the image, drop it
    if (ppLEFT.x()<0 || ppLEFT.x()>uImageWidth ||
	ppLEFT.y()<0 || ppLEFT.y()>uImageHeight)
      return false;

    // if the projected point is outside the image, drop it
    if (ppRIGHT.x()<0 || ppRIGHT.x()>uImageWidth ||
    ppRIGHT.y()<0 || ppRIGHT.y()>uImageHeight)
      return false;

    // compute the error of given the projection
    error = Eigen::Vector4d( ppLEFT.x( )-vecUVLEFT.x( ),
                             ppLEFT.y( )-vecUVLEFT.y( ),
                             ppRIGHT.x( )-vecUVRIGHT.x( ),
                             ppRIGHT.y( )-vecUVRIGHT.y( ) );

    // jacobian of the transform part = [ I 2*skew(T*modelPoint) ]
    Eigen::Matrix<double, 4, 6> Jt;
    Jt.setZero();
    Jt.block<3,3>(0,0).setIdentity();
    Jt.block<3,3>(0,3)=-2*skew(tp);

    // jacobian of the homogeneous division
    // 1/z   0    -x/z^2
    // 0     1/z  -y/z^2
    Eigen::Matrix<double, 2, 3> JpLEFT;
    JpLEFT << 
      1/z, 0,   -pp1LEFT.x()/(z*z),
      0,   1/z, -pp1LEFT.y()/(z*z);

    Eigen::Matrix<double, 2, 3> JpRIGHT;
    JpRIGHT <<
      1/z, 0,   -pp1RIGHT.x()/(z*z),
      0,   1/z, -pp1RIGHT.y()/(z*z);

    // apply the chain rule and get the damn jacobian
    J.block< 2,6 >(0,0) = JpLEFT*matProjectionLEFT*Jt;
    J.block< 2,6 >(2,0) = JpRIGHT*matProjectionRIGHT*Jt;
    return true;
  }

    void init( )
    {
        inliers.resize( model_points.size( ), true );
    }

  double oneRound( ){
    Matrix6d H;
    Vector6d b;
    H.setZero();
    b.setZero();
    double chi2 = 0;

    uNumberOfReprojections = 0;
    uNumberOfInliers = 0;
    for (size_t i = 0; i<model_points.size(); i++){
      Eigen::Vector4d e;
      Matrix4x6d J;
      if (errorAndJacobian(e,J,model_points[i], vecProjectionsUVLEFT[i], vecProjectionsUVRIGHT[i])) {
        double en = e.squaredNorm();
	if (en<dInlierThreshold) {
	  uNumberOfInliers++;
	  inliers[i]=true;
	} else
	  inliers[i]=false;

        double scale = 1;
        if (en>dMaximumError) {
	  scale  = dMaximumError/en;
        }
        chi2 += e.transpose()*e;
        H+=J.transpose()*J;
        b+=J.transpose()*e*scale;
        uNumberOfReprojections++;
      }
      
    }
    //H += sqrt(chi2)*Matrix6f::Identity();
    //cerr << "inliers: " << inliers << endl;
    //cerr << H << endl;
    // add damping?
    Vector6d dt = H.ldlt().solve(-b);
    T = v2t(dt)*T;

    Eigen::Matrix3d R = T.linear();
    Eigen::Matrix3d E = R.transpose() * R;
    E.diagonal().array() -= 1;
    T.linear() -= 0.5 * R * E;

    return chi2;
  }

  double oneRoundInliersOnly( ){
      Matrix6d H;
      Vector6d b;
      H.setZero();
      b.setZero();
      double chi2 = 0;

      uNumberOfReprojections = 0;
      uNumberOfInliers = 0;
      for (size_t i = 0; i<model_points.size(); i++){
        if (!inliers[i])
      continue;
        Eigen::Vector4d e;
        Matrix4x6d J;
        if (errorAndJacobian(e,J,model_points[i], vecProjectionsUVLEFT[i], vecProjectionsUVRIGHT[i])) {
          double en = e.squaredNorm();
      if (en<dInlierThreshold) {
        uNumberOfInliers++;
        inliers[i]=true;
      } else
        inliers[i]=false;

          double scale = 1;
          if (en>dMaximumError) {
        scale  = dMaximumError/en;
          }
          chi2 += e.transpose()*e;
          H+=J.transpose()*J;
          b+=J.transpose()*e*scale;
          uNumberOfReprojections++;
        }

      }
      //H += sqrt(chi2)*Matrix6f::Identity();
      //cerr << "inliers: " << inliers << endl;
      //cerr << H << endl;
      // add damping?
      Vector6d dt = H.ldlt().solve(-b);
      T = v2t(dt)*T;

      Eigen::Matrix3d R = T.linear();
      Eigen::Matrix3d E = R.transpose() * R;
      E.diagonal().array() -= 1;
      T.linear() -= 0.5 * R * E;

      return chi2;
    }
};

} //ds namespace gtools
