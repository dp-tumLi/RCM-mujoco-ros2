#include "franka_example_controllers/recorder.hpp"

Recorder::Recorder(double t_rec, double sampleTime, int NoDataRec, std::string name) {
  _DAT.resize(static_cast<int>(t_rec / sampleTime) + 2, NoDataRec);
  _DAT.setZero();
  _rowindex = 0;
  _columnindex = 0;
  _t_rec = t_rec;
  _name = name;
  _NoDataRec = NoDataRec;
}

Recorder::~Recorder() {
  saveData();
}

void Recorder::getDAT(Eigen::Matrix<double, Eigen::Dynamic, 1>& _Buffer, int rowNum) {
  _Buffer = _DAT.row(rowNum);
}

void Recorder::addToRec(int value) {
  if (_rowindex < _DAT.rows() && _columnindex < _DAT.cols()) {
    _DAT(_rowindex, _columnindex) = value;
    _columnindex++;
  }
}

void Recorder::addToRec(double value) {
  if (_rowindex < _DAT.rows() && _columnindex < _DAT.cols()) {
    _DAT(_rowindex, _columnindex) = value;
    _columnindex++;
  }
}

void Recorder::addToRec(double array[], int sizeofarray) {
  for (int i = 0; i < sizeofarray; i++) {
    if (_rowindex < _DAT.rows() && _columnindex < _DAT.cols()) {
      _DAT(_rowindex, _columnindex) = array[i];
      _columnindex++;
    }
  }
}

void Recorder::addToRec(std::array<double, 3> array) {
  for (double val : array) {
    if (_rowindex < _DAT.rows() && _columnindex < _DAT.cols()) {
      _DAT(_rowindex, _columnindex) = val;
      _columnindex++;
    }
  }
}

void Recorder::addToRec(std::array<double, 6> array) {
  for (double val : array) {
    if (_rowindex < _DAT.rows() && _columnindex < _DAT.cols()) {
      _DAT(_rowindex, _columnindex) = val;
      _columnindex++;
    }
  }
}

void Recorder::addToRec(std::array<double, 7> array) {
  for (double val : array) {
    if (_rowindex < _DAT.rows() && _columnindex < _DAT.cols()) {
      _DAT(_rowindex, _columnindex) = val;
      _columnindex++;
    }
  }
}

void Recorder::addToRec(Eigen::Vector3d& vector) {
  for (int i = 0; i < vector.size(); i++) {
    if (_rowindex < _DAT.rows() && _columnindex < _DAT.cols()) {
      _DAT(_rowindex, _columnindex) = vector[i];
      _columnindex++;
    }
  }
}

void Recorder::saveData() {
  std::ofstream myfile;
  myfile.open(_name + ".m");
  myfile << _name << "m" << "=[" << _DAT << "];\n";
  myfile.close();
  std::cout << "\n\n\t************ Data was written successfully ************\n";
}

void Recorder::next() {
  if (_rowindex < _DAT.rows()) {
    _rowindex++;
    _columnindex = 0;
  }
}