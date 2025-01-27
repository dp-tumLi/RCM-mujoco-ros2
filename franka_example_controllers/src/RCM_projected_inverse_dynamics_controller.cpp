// Copyright (c) 2021 Franka Emika GmbH
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <franka_example_controllers/RCM_projected_inverse_dynamics_controller.hpp>
#include <franka_example_controllers/desired_trajectory.hpp>
#include <cassert>
#include <cmath>
#include <exception>
#include <string>
#include <franka/model.h>

#include <Eigen/SVD>  // Include Eigen's SVD module
#include <filesystem>  // Include filesystem for path manipulation
#include <ament_index_cpp/get_package_share_directory.hpp>  // Include ament_index_cpp for package share directory

inline void pseudoInverse(const Eigen::MatrixXd& M_, Eigen::MatrixXd& M_pinv_, bool damped = true) {
    double lambda_ = damped ? 0.2 : 0.0;

    Eigen::JacobiSVD<Eigen::MatrixXd> svd(M_, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::JacobiSVD<Eigen::MatrixXd>::SingularValuesType sing_vals_ = svd.singularValues();
    Eigen::MatrixXd S_ = M_;  // copying the dimensions of M_, its content is not needed.
    S_.setZero();

    for (int i = 0; i < sing_vals_.size(); i++)
        S_(i, i) = (sing_vals_(i)) / (sing_vals_(i) * sing_vals_(i) + lambda_ * lambda_);

    M_pinv_ = Eigen::MatrixXd(svd.matrixV() * S_.transpose() * svd.matrixU().transpose());
}


namespace franka_example_controllers {

// Initialize the static pointer
Recorder* ProjectedInverseDynamicsController::static_recorder_ = nullptr;

controller_interface::InterfaceConfiguration
ProjectedInverseDynamicsController::command_interface_configuration() const {
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;

  for (int i = 1; i <= num_joints; ++i) {
    config.names.push_back(arm_id_ + "_joint" + std::to_string(i) + "/effort");
  }
  return config;
}

controller_interface::InterfaceConfiguration
ProjectedInverseDynamicsController::state_interface_configuration() const {
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;
  // should be model interface
  for (const auto& franka_robot_model_name : franka_robot_model_->get_state_interface_names()) {
    config.names.push_back(franka_robot_model_name);
  }
  return config;
}

ProjectedInverseDynamicsController::ProjectedInverseDynamicsController() {
  // Set up signal handling
  signal(SIGSEGV, ProjectedInverseDynamicsController::signalHandler);
  signal(SIGINT, ProjectedInverseDynamicsController::signalHandler);
  signal(SIGTERM, ProjectedInverseDynamicsController::signalHandler);
  
  // Load configuration from YAML file
  std::string package_share_directory = ament_index_cpp::get_package_share_directory("franka_example_controllers");
  std::filesystem::path config_path = std::filesystem::path(package_share_directory) / "config" / "recorder_config.yaml";
  std::cout << "config_path: " << config_path.string() << std::endl;
  loadConfig(config_path.string());
}

void ProjectedInverseDynamicsController::loadConfig(const std::string& config_file) {
  YAML::Node config = YAML::LoadFile(config_file);
  double t_rec = config["recorder"]["t_rec"].as<double>();
  double sample_time = config["recorder"]["sample_time"].as<double>();
  int no_data_rec = config["recorder"]["no_data_rec"].as<int>();
  std::string name = config["recorder"]["name"].as<std::string>();
  // Initialize the Recorder object with the parameters
  recorder_ = std::make_unique<Recorder>(t_rec, sample_time, no_data_rec, name);
  static_recorder_ = recorder_.get();  
}

void ProjectedInverseDynamicsController::signalHandler(int signum) {
  std::cout << "Interrupt signal (" << signum << ") received.\n";
  // Save data before exiting
  if (static_recorder_) {
    static_recorder_->saveData();
  }
  // Terminate program
  exit(signum);
}

controller_interface::return_type ProjectedInverseDynamicsController::update(
    const rclcpp::Time& /*time*/,
    const rclcpp::Duration& /*period*/) {
  try{
    Eigen::Map<const Matrix4d> current(franka_robot_model_->getPoseMatrix(franka::Frame::kFlange).data());
    Eigen::Vector3d current_position(current.block<3,1>(0,3));
    Eigen::Quaterniond current_orientation(current.block<3,3>(0,0));
    Eigen::Map<const Matrix7d> inertia(franka_robot_model_->getMassMatrix().data());
    Eigen::Map<const Vector7d> coriolis(franka_robot_model_->getCoriolisForceVector().data());
    Eigen::Matrix<double, 6, 7> jacobian(
        franka_robot_model_->getZeroJacobian(franka::Frame::kFlange).data());
    Eigen::Map<const Vector7d> qD(franka_robot_model_->getRobotState()->dq.data());
    Eigen::Map<const Vector7d> q(franka_robot_model_->getRobotState()->q.data());
    Vector6d error;

    // Extract the flange pose
    Eigen::Map<const Eigen::Matrix4d> flange_pose(franka_robot_model_->getPoseMatrix(franka::Frame::kFlange).data());
    Eigen::Vector3d flange_position = flange_pose.block<3,1>(0,3); // Flange position
    Eigen::Matrix3d flange_rotation = flange_pose.block<3,3>(0,0); // Flange orientation

    // Define the endowrist offset and orientation relative to the flange
    Eigen::Vector3d endowrist_offset(-0.035, 0.61, 0.08); // Endowrist displacement w.r.t. flange
    Eigen::Quaterniond endowrist_orientation(-0.5, 0.5, -0.5, 0.5); // Endowrist orientation w.r.t. flange

    // Compute the endowrist pose relative to the flange
    Eigen::Matrix4d endowrist_pose_in_flange = Eigen::Matrix4d::Identity();
    endowrist_pose_in_flange.block<3,3>(0,0) = endowrist_orientation.toRotationMatrix();
    endowrist_pose_in_flange.block<3,1>(0,3) = endowrist_offset;

    // Compute the endowrist pose in the base frame
    Eigen::Matrix4d endowrist_pose_in_base = flange_pose * endowrist_pose_in_flange;
    Eigen::Vector3d endowrist_position_in_base = endowrist_pose_in_base.block<3,1>(0,3);
    Eigen::Matrix3d endowrist_rotation_in_base = endowrist_pose_in_base.block<3,3>(0,0);

    // Transform the Jacobian from flange to endowrist
    Eigen::Matrix<double, 6, 7> jacobian_flange(franka_robot_model_->getZeroJacobian(franka::Frame::kFlange).data());
    Eigen::Matrix<double, 6, 6> adjoint_endowrist_inv = Eigen::Matrix<double, 6, 6>::Zero();
    adjoint_endowrist_inv.block<3,3>(0,0) = endowrist_orientation.toRotationMatrix().transpose();
    adjoint_endowrist_inv.block<3,3>(0,3) = endowrist_orientation.toRotationMatrix().transpose() * skewSymmetric(endowrist_offset);
    adjoint_endowrist_inv.block<3,3>(3,3) = endowrist_orientation.toRotationMatrix().transpose();
    Eigen::Matrix<double, 6, 7> jacobian_endowrist_virtual = adjoint_endowrist_inv * jacobian_flange;

    // Transform the Jacobian from endowrist to TCP
    Eigen::Vector3d tcp_offset(0.3, 0.0175, 0.04);
    Eigen::Matrix<double, 6, 6> adjoint_endo2tcp_inv = Eigen::Matrix<double, 6, 6>::Zero();
    adjoint_endo2tcp_inv.block<3,3>(0,0) = Eigen::Matrix3d::Identity();
    adjoint_endo2tcp_inv.block<3,3>(0,3) = skewSymmetric(-tcp_offset);
    adjoint_endo2tcp_inv.block<3,3>(3,3) = Eigen::Matrix3d::Identity();
    Eigen::Matrix<double, 6, 7> jacobian_tcp = adjoint_endo2tcp_inv * jacobian_endowrist_virtual;


// Print the results
// std::cout << "Flange Position: " << flange_position.transpose() << std::endl;
// // std::cout << "Flange Orientation: " << Eigen::Quaterniond(flange_rotation).coeffs().transpose() << std::endl;
// // std::cout << "Flange Jacobian: \n" << jacobian_flange << std::endl;

// std::cout << "Endowrist Position: " << endowrist_position.transpose() << std::endl;
// // std::cout << "Endowrist Orientation: " << Eigen::Quaterniond(endowrist_rotation).coeffs().transpose() << std::endl;

// std::cout << "TCP Position: " << tcp_position.transpose() << std::endl;
// // std::cout << "TCP Jacobian: \n" << jacobian_tcp << std::endl;

// std::cout << "ETA Position: " << eta_position.transpose() << std::endl;
// // std::cout << "ETA Jacobian: \n" << jacobian_eta << std::endl;

    current_position = endowrist_position_in_base;
    jacobian = jacobian_endowrist_virtual;


    Eigen::Matrix<double, 6, 6> AdT_cr = Eigen::Matrix<double, 6, 6>::Zero();
    AdT_cr.block<3,3>(0,0) = Eigen::Matrix3d::Identity();
    AdT_cr.block<3,3>(0,3) = skewSymmetric(flange_position - pc_)*Eigen::Matrix3d::Identity();
    AdT_cr.block<3,3>(3,3) = Eigen::Matrix3d::Identity();

    Eigen::Matrix<double, 6, 6> R_block_Nr = Eigen::Matrix<double, 6, 6>::Zero();
    R_block_Nr.block<3,3>(0,0) = flange_rotation;
    R_block_Nr.block<3,3>(3,3) = flange_rotation;

    Eigen::Matrix<double, 6, 6> H = Eigen::Matrix<double, 6, 6>::Zero();
    H.block<2,2>(0,0) = Eigen::Matrix2d::Identity();
    Eigen::Matrix<double, 6, 7> J_c = H*R_block_Nr.transpose()*AdT_cr*jacobian_flange;
    Eigen::Matrix<double, 2, 7> A = J_c.block<2,7>(0,0);

    Eigen::MatrixXd Apinv;
    pseudoInverse(A, Apinv);
    Eigen::Matrix<double, 7, 7> P = Eigen::Matrix<double, 7, 7>::Identity() - Apinv*A;
    Eigen::MatrixXd Ppinv;
    pseudoInverse(P, Ppinv);

    Eigen::Matrix<double, 9, 1> desired;
    auto time = this->get_node()->now() - start_time_;

    spiralTrajectory(20, time.seconds(), desired_position, desired);    

    Vector6d Xdes = Vector6d::Zero(); Xdes.block<3,1>(0,0) = desired.block<3,1>(0,0);
    Vector6d Xdes_dot = Vector6d::Zero(); Xdes_dot.block<3,1>(0,0) = desired.block<3,1>(3,0);
    Vector6d Xdes_ddot = Vector6d::Zero(); Xdes_ddot.block<3,1>(0,0) = desired.block<3,1>(6,0);
    Vector6d X = Vector6d::Zero(); X.block<3,1>(0,0) = current_position;
    Eigen::Matrix<double, 6, 6> Lambda_x_inv = Eigen::Matrix<double, 6, 6>::Zero();
    Lambda_x_inv = jacobian*inertia.inverse()*jacobian.transpose();
    Eigen::Matrix<double, 6, 6> D_x = Eigen::Matrix<double, 6, 6>::Zero();
    Eigen::Matrix<double, 6, 6> K_x = Eigen::Matrix<double, 6, 6>::Zero();

    // here change the stiffness and damping matrices
    D_x.block<3,3>(0,0) = 10*Matrix3d::Identity();
    K_x.block<3,3>(0,0) = 100*Matrix3d::Identity();


    Vector6d Xcmd_ddot = Xdes_ddot + Lambda_x_inv * (D_x * (Xdes_dot - jacobian * qD) + K_x * (Xdes - X));

    Vector7d qcmd_ddot; qcmd_ddot.setZero();
    Eigen::MatrixXd J_pinv;
    pseudoInverse(jacobian, J_pinv);
    Eigen::Matrix<double, 6, 7> jacobian_dot = (jacobian - previous_jacobian_)/1e-3;
    qcmd_ddot = J_pinv * (Xcmd_ddot - jacobian_dot*qD);

    Vector7d tauf, tauc, tau_d;
    tauf.setZero(); tauc.setZero(); tau_d.setZero();
    tauf = Ppinv*P*(inertia*qcmd_ddot + coriolis);
    tauc = (Eigen::Matrix<double, 7, 7>::Identity() - Ppinv*P)*(Vector7d::Zero());

    Vector7d tau_arbitary; tau_arbitary.setZero();
    tau_arbitary = inertia*qcmd_ddot + coriolis;
    tau_d <<  tau_arbitary;
    for (int i = 0; i < num_joints; ++i) {
      command_interfaces_[i].set_value(tau_d(i));
    }
    // std::cout<<"time: "<<time.seconds()<<" tau_d: "<<tau_d.transpose()<<std::endl;

    // Check if Jacobian drops rank
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(jacobian, Eigen::ComputeThinU | Eigen::ComputeThinV);
    double tolerance = 1e-6;  // Define a tolerance for singular values
    Eigen::VectorXd singular_values = svd.singularValues();
    // std::cout << "Singular values of the Jacobian: " << singular_values.transpose() << std::endl;
    if (singular_values.minCoeff() < tolerance) {
      std::cout << "Jacobian has dropped rank!" << std::endl;
    }
    
    recorder_->addToRec(time.seconds());
    recorder_->addToRec(current_position);
    Vector3d xdes = desired.head<3>();
    recorder_->addToRec(xdes);
    recorder_->addToRec(flange_position);
    // recorder_->addToRec(endowrist_position);
    // recorder_->addToRec(flange_position);
    // recorder_->addToRec(eta_position);
    recorder_->next();


    previous_jacobian_ = jacobian;
    return controller_interface::return_type::OK;
  } catch (const std::exception& e) {
    std::cerr << "Exception caught in update: " << e.what() << std::endl;
    if (static_recorder_) {
      static_recorder_->saveData();
    }
    return controller_interface::return_type::ERROR;
  }
}

CallbackReturn ProjectedInverseDynamicsController::on_init() {
  try {
    auto_declare<std::string>("arm_id", "panda");
  } catch (const std::exception& e) {
    fprintf(stderr, "Exception thrown during init stage with message: %s \n", e.what());
    return CallbackReturn::ERROR;
  }
  return CallbackReturn::SUCCESS;
}

CallbackReturn ProjectedInverseDynamicsController::on_configure(
    const rclcpp_lifecycle::State& /*previous_state*/) {
  arm_id_ = get_node()->get_parameter("arm_id").as_string();
  franka_robot_model_ = std::make_unique<franka_semantic_components::FrankaRobotModel>(
      franka_semantic_components::FrankaRobotModel(arm_id_ + "/robot_model",
                                                   arm_id_));
        
  return CallbackReturn::SUCCESS;
}

CallbackReturn ProjectedInverseDynamicsController::on_activate(
    const rclcpp_lifecycle::State& /*previous_state*/) {
  franka_robot_model_->assign_loaned_state_interfaces(state_interfaces_);
  start_time_ = this->get_node()->now();
  // desired = Matrix4d(franka_robot_model_->getPoseMatrix(franka::Frame::kFlange).data());
  // desired_position = Vector3d(desired.block<3,1>(0,3));
  // desired_orientation = Quaterniond(desired.block<3,3>(0,0));
  // desired_qn = Vector7d(franka_robot_model_->getRobotState()->q.data());
  // desired_position = Eigen::Vector3d(0.4, 0.45, 0.6);  // Set your initial Cartesian position
  // desired_orientation = Eigen::Quaterniond::Identity(); // Set your initial orientation

  double roll = -M_PI / 2;   // 45 degrees
  double pitch = 0.0;  // 30 degrees
  double yaw = 0.0;    // 60 degrees
  Eigen::Matrix3d rotation_matrix = 
      Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()).toRotationMatrix() *
      Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY()).toRotationMatrix() *
      Eigen::AngleAxisd(roll, Eigen::Vector3d::UnitX()).toRotationMatrix();
  desired_orientation = Eigen::Quaterniond(rotation_matrix);
  desired_qn = Vector7d(franka_robot_model_->getRobotState()->q.data());
  // Extract the flange pose
  Eigen::Map<const Eigen::Matrix4d> flange_pose(franka_robot_model_->getPoseMatrix(franka::Frame::kFlange).data());
  Eigen::Vector3d flange_position = flange_pose.block<3,1>(0,3); // Flange position
  Eigen::Matrix3d flange_rotation = flange_pose.block<3,3>(0,0); // Flange orientation

  Eigen::Vector3d endowrist_offset(-0.035, 0.61, 0.08); // Endowrist displacement w.r.t. flange
  Eigen::Quaterniond endowrist_orientation(-0.5, 0.5, -0.5, 0.5); // Endowrist orientation w.r.t. flange

  Eigen::Vector3d endowrist_position = flange_position + flange_rotation * endowrist_offset; // Endowrist position
  Eigen::Matrix3d endowrist_rotation = flange_rotation * endowrist_orientation.toRotationMatrix(); // Endowrist orientation

  Eigen::Vector3d tcp_offset(0.0, 0.0175, 0.04); // TCP displacement w.r.t. endowrist
  Eigen::Vector3d tcp_position = endowrist_position + endowrist_rotation * tcp_offset; // TCP position

  Eigen::Vector3d eta_offset(0.3, 0.0175, 0.04); // ETA displacement w.r.t. endowrist
  Eigen::Vector3d eta_position = endowrist_position + endowrist_rotation * eta_offset; // ETA position
    
  // desired_orientation = Quaterniond(endowrist_rotation); // Endowrist orientation
  desired_position = tcp_position; // TCP position
  pc_ = eta_position; // ETA position
  // desired_position(0) += 0.4; // Modify the x-coordinate
  // desired_position(1) += 0.45; // Modify the y-coordinate
  // desired_position(2) -= 0.3; // Set the z-coordinate to a specific value

  double pos_stiff = 1000.0;
  double rot_stiff = 200.0;
  stiffness.setIdentity();
  stiffness.topLeftCorner(3, 3) << pos_stiff * Matrix3d::Identity();
  stiffness.bottomRightCorner(3, 3) << rot_stiff * Matrix3d::Identity();
  // Simple critical damping
  damping.setIdentity();
  damping.topLeftCorner(3,3) << 2 * sqrt(pos_stiff) * Matrix3d::Identity();
  damping.bottomRightCorner(3, 3) << 0.717 * 2 * sqrt(rot_stiff) * Matrix3d::Identity();
  n_stiffness = 10.0;

  previous_jacobian_ = Eigen::Matrix<double, 6, 7>::Zero();


  

  return CallbackReturn::SUCCESS;
}

CallbackReturn ProjectedInverseDynamicsController::on_deactivate(
    const rclcpp_lifecycle::State& /*previous_state*/){
  franka_robot_model_->release_interfaces();
  recorder_->saveData();
  return CallbackReturn::SUCCESS;
}

}  // namespace franka_example_controllers
#include "pluginlib/class_list_macros.hpp"
// NOLINTNEXTLINE
PLUGINLIB_EXPORT_CLASS(franka_example_controllers::ProjectedInverseDynamicsController,
                       controller_interface::ControllerInterface)