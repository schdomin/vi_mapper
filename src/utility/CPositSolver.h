#include <iostream>
#include <unistd.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Cholesky>
#include <Eigen/StdVector>
#include <vector>
#include <cstdio>
#include <fstream>

using namespace std;

namespace gtools
{

typedef std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> > Vector3dVector;
typedef std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d> > Vector2dVector;
typedef Eigen::Matrix<double, 6, 6> Matrix6d;
typedef Eigen::Matrix<double, 6, 1> Vector6d;
typedef Eigen::Matrix<double, 2, 6> Matrix2_6d;

struct CPositSolver{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  CPositSolver(){
    T.setIdentity();
    K << 468.2793078854663, 0.0,               368.7120388971904,
         0.0,               466.4527618561632, 215.186116509721,
         0.0,               0.0,               1.0;
    imageCols=752;
    imageRows=480;
    maxError = 1e6;
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


  Vector3dVector modelPoints;
  Vector2dVector imagePoints;
  
  Eigen::Isometry3d T;
  Eigen::Matrix3d K;
  float imageRows, imageCols;
  float maxError;
 
  
  inline void project(Vector2dVector& dest){
    Eigen::Matrix3d KT = K*T.linear();
    Eigen::Vector3d Kt = K*T.translation();
    dest.resize(modelPoints.size());
    for (size_t i = 0; i<modelPoints.size(); i++){
      const Eigen::Vector3d& p=modelPoints[i];
      Eigen::Vector3d pp=KT*p + Kt;
      if (pp.z()<0)
	pp << 0,0,-1;
      pp *= (1./pp.z());
      if (pp.x()<0 || pp.x()>imageCols ||
	  pp.y()<0 || pp.y()>imageRows)
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
    Eigen::Vector3d pp1 = K*tp;
    Eigen::Vector2d pp (pp1.x()/pp1.z(), pp1.y()/pp1.z());

    // if the projected point is outside the image, drop it
    if (pp.x()<0 || pp.x()>imageCols ||
	pp.y()<0 || pp.y()>imageRows)
      return false;

    // compute the error of given the projection
    error = pp - imagePoint;

    // jacobian of the transform part = [ I 2*skew(T*modelPoint) ]
    Eigen::Matrix<double, 3, 6> Jt;
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
    J=Jp*K*Jt;
    return true;
  }

  float oneRound(){
    Matrix6d H;
    Vector6d b;
    H.setZero();
    b.setZero();
    float chi2 = 0;

    int inliers = 0;
    for (size_t i = 0; i<modelPoints.size(); i++){
      Eigen::Vector2d e;
      Matrix2_6d J;
      if (errorAndJacobian(e,J,modelPoints[i], imagePoints[i])) {
	float en = e.squaredNorm();
	float scale = 1;
	if (en>maxError){
	  scale  = maxError/en;
	}
	chi2 += e.transpose()*e;
	H+=J.transpose()*J;
	b+=J.transpose()*e*scale;
	inliers ++;
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



/*int main(){

  int numPoints = 50;
  float cx=0, cy=0, cz=1;
  float xspread = 1, yspread = 1, zspread = 0.1;
  
  // sample randomly model points
  Vector3fVector modelPoints;
  modelPoints.resize(numPoints);
  for (size_t i = 0; i<numPoints; i++){
    float x = (drand48()-0.5)*xspread + cx;
    float y = (drand48()-0.5)*yspread + cy;
    float z = (drand48()-0.5)*zspread + cz;
    modelPoints[i] = Eigen::Vector3f(x,y,z);
  }
  
  // constructs a solver
  PositSolver solver;

  // feed the solver with the 3D points in the camera frame
  solver.modelPoints = modelPoints;

  // projects the model to image points 
  // this is a simulation, in real you get the points from your matching
  Vector2fVector imagePoints;
  solver.project(imagePoints);


  solver.imagePoints = imagePoints;

  // sets a wrong initial guess to the solver and checks how it behaves
  solver.T.translation() << 0.2, 0.2, 0.2;
  solver.T.linear()=Eigen::AngleAxisf(M_PI/8, Eigen::Vector3f(0,1,1)).toRotationMatrix();

  

  for (int iteration = 0; iteration< 10; iteration++){
    cerr << "iteration "<< iteration << endl;
    cerr << " error: " << solver.oneRound() << endl;


    char filename[1024];
    sprintf (filename, "iteration-%05d.dat", iteration);
    Vector2fVector currentPoints;
    solver.project(currentPoints);
    ofstream os(filename);
    for (size_t j = 0; j<imagePoints.size(); j++){
      Eigen::Vector2f ip = imagePoints[j];
      Eigen::Vector2f cp = currentPoints[j];
      os << ip.x() << " " << ip.y() << endl;
      os << cp.x() << " " << cp.y() << endl;
      os << endl;
    }


  }
  
  // put a certain number of points in the scene;
  // translate and project them
  
  // eliminate the points which are non visible
  
  
}*/

