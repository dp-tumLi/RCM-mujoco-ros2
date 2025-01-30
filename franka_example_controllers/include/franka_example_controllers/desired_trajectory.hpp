#ifndef DESIRED_TRAJECTORY_H
#define DESIRED_TRAJECTORY_H

#include <Eigen/Dense>
#include <iostream>

void spiralTrajectory(double param1, double param2, Eigen::Matrix<double, 3, 1>& matrix1, Eigen::Matrix<double, 9, 1>& matrix2);
Eigen::Matrix3d skewSymmetric(const Eigen::Vector3d& v);
Eigen::Vector3d unskewSymmetric(const Eigen::Matrix3d& m);
#endif // DESIRED_TRAJECTORY_H