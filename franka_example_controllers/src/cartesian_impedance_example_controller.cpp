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

#include <franka_example_controllers/cartesian_impedance_example_controller.hpp>
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

// void spiralTrajectory(double T, double t, Eigen::Matrix<double, 3, 1> &p0, Eigen::Matrix<double, 9, 1> &des)
// {
//     double xi[3];
//     double L, v_max, a_max, Tm, S, Sd, Sdd;

//     xi[0] = p0[0];
//     xi[1] = p0[1];
//     xi[2] = p0[2];


//     double pitch = 0.015; //0.015
//     double R = 0.05; //0.015
//     L = 8 * M_PI;

//     v_max = 1.25 * L / T; // TODO

//     a_max = v_max * v_max / (T * v_max - L);
//     Tm = v_max / a_max;

//     if (Tm < T / 5 || Tm > T / 2.1)
//     {
//         std::cout << "HS: ERROR in trajectory planning timing law" << std::endl;
//         exit(1);
//     }

//     if (t >= 0 && t <= Tm)
//     {
//         S = a_max * t * t / 2;
//         Sd = a_max * t;
//         Sdd = a_max;
//     }
//     else if (t >= Tm && t <= (T - Tm))
//     {
//         S = v_max * t - v_max * v_max / (2 * a_max);
//         Sd = v_max;
//         Sdd = 0;
//     }
//     else if (t >= (T - Tm) && t <= T)
//     {
//         S = -a_max * (t - T) * (t - T) / 2 + v_max * T - v_max * v_max / a_max;
//         Sd = -a_max * (t - T);
//         Sdd = -a_max;
//     }
//     else
//     {
//         S = L;
//         Sd = 0;
//         Sdd = 0;
//     }
//     // Geometric path
//     //  spiral with z axis
//     // p_des
//     des[0] = (xi[0] - R) + R * cos(S);
//     des[1] = xi[1] + R * sin(S);
//     des[2] = xi[2] + S * pitch / (2 * M_PI);
//     // pd_des
//     des[3] = -R * Sd * sin(S);
//     des[4] = R * Sd * cos(S);
//     des[5] = pitch * Sd / (2 * M_PI);
//     // pdd_des
//     des[6] = -R * Sdd * sin(S) - R * Sd * Sd * cos(S);
//     des[7] = R * Sdd * cos(S) - R * Sd * Sd * sin(S);
//     des[8] = pitch * Sdd / (2 * M_PI);
// }


namespace franka_example_controllers {

// Initialize the static pointer
Recorder* CartesianImpedanceExampleController::static_recorder_ = nullptr;

controller_interface::InterfaceConfiguration
CartesianImpedanceExampleController::command_interface_configuration() const {
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;

  for (int i = 1; i <= num_joints; ++i) {
    config.names.push_back(arm_id_ + "_joint" + std::to_string(i) + "/effort");
  }
  return config;
}

controller_interface::InterfaceConfiguration
CartesianImpedanceExampleController::state_interface_configuration() const {
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;
  // should be model interface
  for (const auto& franka_robot_model_name : franka_robot_model_->get_state_interface_names()) {
    config.names.push_back(franka_robot_model_name);
  }
  return config;
}

CartesianImpedanceExampleController::CartesianImpedanceExampleController() {
  // Set up signal handling
  signal(SIGSEGV, CartesianImpedanceExampleController::signalHandler);
  signal(SIGINT, CartesianImpedanceExampleController::signalHandler);
  signal(SIGTERM, CartesianImpedanceExampleController::signalHandler);
  
  // Load configuration from YAML file
  std::string package_share_directory = ament_index_cpp::get_package_share_directory("franka_example_controllers");
  std::filesystem::path config_path = std::filesystem::path(package_share_directory) / "config" / "recorder_config.yaml";
  std::cout << "config_path: " << config_path.string() << std::endl;
  loadConfig(config_path.string());
}

void CartesianImpedanceExampleController::loadConfig(const std::string& config_file) {
  YAML::Node config = YAML::LoadFile(config_file);
  double t_rec = config["recorder"]["t_rec"].as<double>();
  double sample_time = config["recorder"]["sample_time"].as<double>();
  int no_data_rec = config["recorder"]["no_data_rec"].as<int>();
  std::string name = config["recorder"]["name"].as<std::string>();
  // Initialize the Recorder object with the parameters
  recorder_ = std::make_unique<Recorder>(t_rec, sample_time, no_data_rec, name);
  static_recorder_ = recorder_.get();  
}

void CartesianImpedanceExampleController::signalHandler(int signum) {
  std::cout << "Interrupt signal (" << signum << ") received.\n";
  // Save data before exiting
  if (static_recorder_) {
    static_recorder_->saveData();
  }
  // Terminate program
  exit(signum);
}

controller_interface::return_type CartesianImpedanceExampleController::update(
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


    // Eigen::Map<const Matrix4d> flange_pose(franka_robot_model_->getPoseMatrix(franka::Frame::kFlange).data());
    // Eigen::Vector3d flange_position(flange_pose.block<3,1>(0,3));
    // Eigen::Quaterniond flange_orientation(flange_pose.block<3,3>(0,0));
    // Eigen::Vector3d visual_tcp_position = flange_position + flange_orientation * Eigen::Vector3d(0.0, 0, 0.0);
    // Eigen::Matrix<double, 6, 7> jacobian_flange(
    //     franka_robot_model_->getZeroJacobian(franka::Frame::kFlange).data());
    // Eigen::Matrix<double, 6, 7> jacobian_visual_tcp = jacobian_flange;
    // jacobian_visual_tcp.block<3,7>(0,0) += flange_orientation.toRotationMatrix() * Eigen::Matrix<double, 3, 7>::Zero();

    // std::cout << "Visual TCP Position: " << visual_tcp_position.transpose() << std::endl;
    // std::cout << "Visual TCP Jacobian: " << jacobian_visual_tcp << std::endl;

    // Extract the flange pose
    Eigen::Map<const Matrix4d> flange_pose(franka_robot_model_->getPoseMatrix(franka::Frame::kFlange).data());
    Eigen::Vector3d flange_position(flange_pose.block<3,1>(0,3));
    Eigen::Quaterniond flange_orientation(flange_pose.block<3,3>(0,0));
    Eigen::Vector3d endowrist_offset(-0.035, 0.61, 0.08);
    Eigen::Quaterniond endowrist_orientation(-0.5, 0.5, -0.5, 0.5);
    Eigen::Vector3d endowrist_position = flange_position + flange_orientation * endowrist_offset;
    Eigen::Quaterniond endowrist_full_orientation = flange_orientation * endowrist_orientation;
    Eigen::Vector3d tcp_offset(0.3, 0.0175, 0.04);
    Eigen::Vector3d tcp_position = endowrist_position + endowrist_full_orientation * tcp_offset;
    Eigen::Matrix<double, 6, 7> jacobian_flange(
        franka_robot_model_->getZeroJacobian(franka::Frame::kFlange).data());
    Eigen::Matrix<double, 6, 7> jacobian_tcp = jacobian_flange;
    jacobian_tcp.block<3,7>(0,0) += endowrist_full_orientation.toRotationMatrix() * Eigen::Matrix<double, 3, 7>::Zero();

    // Print the TCP position and Jacobian
    // std::cout << "TCP Position: " << tcp_position.transpose() << std::endl;
    // std::cout << "TCP Jacobian: " << jacobian_tcp << std::endl;
    current_position = tcp_position;
    jacobian = jacobian_tcp;


    // Define the desired orientation using Euler angles (roll, pitch, yaw)
    double roll = -M_PI / 2;  // 45 degrees
    double pitch = 0.0; // 30 degrees
    double yaw = 0.0;   // 60 degrees

    desired_orientation = Eigen::AngleAxisd(roll, Eigen::Vector3d::UnitX()) *
                          Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY()) *
                          Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ());

    auto time = this->get_node()->now() - start_time_;
    // auto desired_position_cur = desired_position;
    // desired_position_cur[0] += 0.1*sin(time.seconds());
    // desired_position_cur[1] += 0.1*sin(time.seconds());
    Eigen::Matrix<double, 9, 1> desired;
    spiralTrajectory(20, time.seconds(), desired_position, desired);
    auto desired_position_cur = desired.head<3>();
    // std::cout<<"time: "<<time.seconds()<<" pos_des: "<<desired_position_cur.transpose()<<std::endl;
    // std::cout<<"pos_cur: "<<current_position.transpose()<<" error: "<<(current_position - desired_position_cur).transpose()<<std::endl;
    error.head(3) << current_position - desired_position_cur;
    if (desired_orientation.coeffs().dot(current_orientation.coeffs()) < 0.0) {
      current_orientation.coeffs() << -current_orientation.coeffs();
    }
    Eigen::Quaterniond rot_error(
        current_orientation * desired_orientation.inverse());
    Eigen::AngleAxisd rot_error_aa(rot_error);
    error.tail(3) << rot_error_aa.axis() * rot_error_aa.angle();
    Vector7d tau_task, tau_nullspace, tau_d;
    tau_task.setZero();
    tau_nullspace.setZero();
    tau_d.setZero();
    
    tau_task << jacobian.transpose() * (-stiffness*error - damping*(jacobian*qD));

    Eigen::MatrixXd jacobian_transpose_pinv;
      pseudoInverse(jacobian.transpose(), jacobian_transpose_pinv);
    tau_nullspace << (Eigen::MatrixXd::Identity(7, 7) -
                        jacobian.transpose() * jacobian_transpose_pinv) *
                          (n_stiffness * (desired_qn - q) -
                            (2.0 * sqrt(n_stiffness)) * qD);

    tau_d <<  tau_task + coriolis + tau_nullspace;
    for (int i = 0; i < num_joints; ++i) {
      command_interfaces_[i].set_value(tau_d(i));
    }
    std::cout<<"tau_d: "<<tau_d.transpose()<<std::endl;
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
    Vector3d X_des = desired.head<3>();
    recorder_->addToRec(X_des);
    recorder_->next();
    return controller_interface::return_type::OK;
  } catch (const std::exception& e) {
    std::cerr << "Exception caught in update: " << e.what() << std::endl;
    if (static_recorder_) {
      static_recorder_->saveData();
    }
    return controller_interface::return_type::ERROR;
  }
}

CallbackReturn CartesianImpedanceExampleController::on_init() {
  try {
    auto_declare<std::string>("arm_id", "panda");
  } catch (const std::exception& e) {
    fprintf(stderr, "Exception thrown during init stage with message: %s \n", e.what());
    return CallbackReturn::ERROR;
  }
  return CallbackReturn::SUCCESS;
}

CallbackReturn CartesianImpedanceExampleController::on_configure(
    const rclcpp_lifecycle::State& /*previous_state*/) {
  arm_id_ = get_node()->get_parameter("arm_id").as_string();
  franka_robot_model_ = std::make_unique<franka_semantic_components::FrankaRobotModel>(
      franka_semantic_components::FrankaRobotModel(arm_id_ + "/robot_model",
                                                   arm_id_));
        
  return CallbackReturn::SUCCESS;
}

CallbackReturn CartesianImpedanceExampleController::on_activate(
    const rclcpp_lifecycle::State& /*previous_state*/) {
  franka_robot_model_->assign_loaned_state_interfaces(state_interfaces_);
  start_time_ = this->get_node()->now();
  desired = Matrix4d(franka_robot_model_->getPoseMatrix(franka::Frame::kFlange).data());
  desired_position = Vector3d(desired.block<3,1>(0,3));
  desired_orientation = Quaterniond(desired.block<3,3>(0,0));
  desired_qn = Vector7d(franka_robot_model_->getRobotState()->q.data());

  double pos_stiff = 400.0;
  double rot_stiff = 20.0;
  stiffness.setIdentity();
  stiffness.topLeftCorner(3, 3) << pos_stiff * Matrix3d::Identity();
  stiffness.bottomRightCorner(3, 3) << rot_stiff * Matrix3d::Identity();
  // Simple critical damping
  damping.setIdentity();
  damping.topLeftCorner(3,3) << 2 * sqrt(pos_stiff) * Matrix3d::Identity();
  damping.bottomRightCorner(3, 3) << 0.8 * 2 * sqrt(rot_stiff) * Matrix3d::Identity();
  n_stiffness = 10.0;

  return CallbackReturn::SUCCESS;
}

CallbackReturn CartesianImpedanceExampleController::on_deactivate(
    const rclcpp_lifecycle::State& /*previous_state*/){
  franka_robot_model_->release_interfaces();
  recorder_->saveData();
  return CallbackReturn::SUCCESS;
}

}  // namespace franka_example_controllers
#include "pluginlib/class_list_macros.hpp"
// NOLINTNEXTLINE
PLUGINLIB_EXPORT_CLASS(franka_example_controllers::CartesianImpedanceExampleController,
                       controller_interface::ControllerInterface)