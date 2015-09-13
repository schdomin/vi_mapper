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

typedef std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> > Vector3dVector;
typedef std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d> > Vector2dVector;
typedef Eigen::Matrix<double, 6, 6> Matrix6d;
typedef Eigen::Matrix<double, 6, 1> Vector6d;
typedef Eigen::Matrix<double, 2, 6> Matrix2_6d;

struct CPositSolverProjection{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  CPositSolverProjection( const Eigen::Matrix< double, 3, 4>& p_matProjection ){
    T.setIdentity();
    P = p_matProjection;
    num_inliers=0;
    num_reprojected_points=0;
    image_cols=752;
    image_rows=480;
    maxError = 1e6;
    inlier_error_threshold = 4;
  }

  inline Eigen::Isometry3d v2t(const Vector6d& t){
    Eigen::Isometry3d T;
    T.setIdentity();
    T.translation()=t.head<3>();
    float w=t.block<3,1>(3,0).squaredNorm();
    if (w<1) {
      w=sqrt(1-w);
      T.linear()=Eigen::Quaterniond(w, t(3), t(4), t(5)).toRotationMatrix();
    } else {
      T.linear().setIdentity();
    }
    return T;
  }

  inline Eigen::Matrix3d skew(const Eigen::Vector3d& p){
    Eigen::Matrix3d s;
    s <<
      0,  -p.z(), p.y(),
      p.z(), 0,  -p.x(),
      -p.y(), p.x(), 0;
    return s;
  }


  Vector3dVector model_points;
  Vector2dVector image_points;
  
  Eigen::Isometry3d T;
  Eigen::Matrix< double, 3, 4 > P;
  int num_reprojected_points;
  int num_inliers;
  float image_rows, image_cols;
  float maxError;
  float inlier_error_threshold;
  std::vector<bool> inliers;
  
  inline void project(Vector2dVector& dest){
    dest.resize(model_points.size());
    for (size_t i = 0; i<model_points.size(); i++){
      const Eigen::Vector3d& p=model_points[i];
      Eigen::Vector3d pp=P*T*p;
      if (pp.z()<0)
	pp << 0,0,-1;
      pp *= (1./pp.z());
      if (pp.x()<0 || pp.x()>image_cols ||
	  pp.y()<0 || pp.y()>image_rows)
	pp << 0,0,-1;
      dest[i].x()=pp.x();
      dest[i].y()=pp.y();
    }
  }

  inline bool errorAndJacobian(Eigen::Vector2d&  error, Matrix2_6d&  J, const Eigen::Vector3d& modelPoint, const Eigen::Vector2d& imagePoint){
    J.setZero();
    // apply the transform to the point
    Eigen::Vector3d tp=T*modelPoint;

    // if the points is behind the camera, drop it;
    if (tp.z()<0)
      return false;

    float z = tp.z();

    // apply the projection to the transformed point
    Eigen::Vector4d vecPointHomogeneous( tp.x( ), tp.y( ), tp.z( ), 1.0 );
    Eigen::Vector3d pp1 = P*vecPointHomogeneous;
    Eigen::Vector2d pp (pp1.x()/pp1.z(), pp1.y()/pp1.z());

    // if the projected point is outside the image, drop it
    if (pp.x()<0 || pp.x()>image_cols ||
	pp.y()<0 || pp.y()>image_rows)
      return false;

    // compute the error of given the projection
    error = pp - imagePoint;

    // jacobian of the transform part = [ I 2*skew(T*modelPoint) ]
    Eigen::Matrix<double, 4, 6> Jt;
    Jt.setZero();
    Jt.block<3,3>(0,0).setIdentity();
    Jt.block<3,3>(0,3)=-2*skew(tp);

    // jacobian of the homogeneous division
    // 1/z   0    -x/z^2
    // 0     1/z  -y/z^2
    Eigen::Matrix<double, 2, 3> Jp;
    Jp << 
      1/z, 0,   -pp1.x()/(z*z),
      0,   1/z, -pp1.y()/(z*z);

    // apply the chain rule and get the damn jacobian
    J=Jp*P*Jt;
    return true;
  }

  void init() {
    inliers.resize(model_points.size());
    std::fill(inliers.begin(), inliers.end(), true);
  }

  double oneRound( ){
    Matrix6d H;
    Vector6d b;
    H.setZero();
    b.setZero();
    double chi2 = 0;

    num_reprojected_points = 0;
    num_inliers = 0;
    for (size_t i = 0; i<model_points.size(); i++){
      Eigen::Vector2d e;
      Matrix2_6d J;
      if (errorAndJacobian(e,J,model_points[i], image_points[i])) {
        double en = e.squaredNorm();
	if (en<inlier_error_threshold) {
	  num_inliers++;
	  inliers[i]=true;
	} else
	  inliers[i]=false;

        double scale = 1;
        if (en>maxError) {
	  scale  = maxError/en;
        }
        chi2 += e.transpose()*e;
        H+=J.transpose()*J;
        b+=J.transpose()*e*scale;
        num_reprojected_points++;
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

      num_reprojected_points = 0;
      num_inliers = 0;
      for (size_t i = 0; i<model_points.size(); i++){
        if (!inliers[i])
      continue;
        Eigen::Vector2d e;
        Matrix2_6d J;
        if (errorAndJacobian(e,J,model_points[i], image_points[i])) {
          double en = e.squaredNorm();
      if (en<inlier_error_threshold) {
        num_inliers++;
        inliers[i]=true;
      } else
        inliers[i]=false;

          double scale = 1;
          if (en>maxError) {
        scale  = maxError/en;
          }
          chi2 += e.transpose()*e;
          H+=J.transpose()*J;
          b+=J.transpose()*e*scale;
          num_reprojected_points++;
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
