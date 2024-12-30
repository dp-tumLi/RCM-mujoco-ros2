// Recorder.hpp

#pragma once

#include <Eigen/Dense>
#include <fstream>
#include <iostream>
#include <array>
#include <string>

class Recorder {
 public:
  Recorder(double t_rec, double sampleTime, int NoDataRec = 10, std::string name = "DATA");
  ~Recorder();

  void addToRec(int value);
  void addToRec(double value);
  void addToRec(double array[], int sizeofarray);
  void addToRec(std::array<double, 3> array);
  void addToRec(std::array<double, 6> array);
  void addToRec(std::array<double, 7> array);

  void addToRec(Eigen::Vector3d& vector);
  void saveData();
  void next();

  void getDAT(Eigen::Matrix<double, Eigen::Dynamic, 1>& _Buffer, int rowNum);

 private:
  int _index;
  int _columnindex;
  int _rowindex;
  double _t_rec;
  int _NoDataRec;
  std::string _name;
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> _DAT;
};
