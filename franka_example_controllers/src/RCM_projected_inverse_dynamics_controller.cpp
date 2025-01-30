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


    Eigen::Vector3d endowrist_offset(-0.035, 0.61, 0.08); // Endowrist displacement w.r.t. flange
    Eigen::Quaterniond endowrist_orientation(-0.5, 0.5, -0.5, 0.5); // Endowrist orientation w.r.t. flange
    Eigen::Vector3d tcp_offset(0.0, 0.0175, 0.04);

    // Eigen::Map<const Eigen::Matrix4d> flange_pose(franka_robot_model_->getPoseMatrix(franka::Frame::kFlange).data());
    std::array<double, 16> raw_pose = franka_robot_model_->getPoseMatrix(franka::Frame::kFlange);
    Eigen::Matrix4d flange_pose = Eigen::Map<const Eigen::Matrix4d>(raw_pose.data());
    Eigen::Vector3d flange_position = flange_pose.block<3,1>(0,3); // Flange position
    Eigen::Matrix3d flange_rotation = flange_pose.block<3,3>(0,0); // Flange orientation

    Eigen::Matrix4d T7endo = Eigen::Matrix4d::Identity();
    T7endo.block<3,3>(0,0) = endowrist_orientation.toRotationMatrix();
    T7endo.block<3,1>(0,3) = endowrist_offset;

    Eigen::Matrix4d Tendo_tcp = Eigen::Matrix4d::Identity();
    Tendo_tcp.block<3,1>(0,3) = tcp_offset;

    Eigen::Matrix4d T0endo = flange_pose*T7endo;
    Eigen::Matrix4d T7tcp = T7endo * Tendo_tcp;
    Eigen::Matrix4d T0tcp = flange_pose*T7tcp;

    Eigen::Vector3d p0endo = T0endo.block<3,1>(0,3);
    Eigen::Matrix3d R0endo = T0endo.block<3,3>(0,0);
    Eigen::Vector3d p0tcp = T0tcp.block<3,1>(0,3);
    Eigen::Matrix3d R0tcp = T0tcp.block<3,3>(0,0);

    Eigen::Matrix3d R7tcp = T7tcp.block<3,3>(0,0);
    Eigen::Vector3d p7tcp = T7tcp.block<3,1>(0,3);
    Eigen::Matrix<double, 6, 7> jacobian_flange(franka_robot_model_->getZeroJacobian(franka::Frame::kFlange).data());
    Eigen::Matrix<double, 6, 6> adTtcp7 = Eigen::Matrix<double, 6, 6>::Zero();
    Eigen::Matrix4d Ttcp7 = T7tcp.inverse();
    Eigen::Matrix3d Rtcp7 = Ttcp7.block<3,3>(0,0); 
    Eigen::Vector3d ptcp7 = Ttcp7.block<3,1>(0,3);
    adTtcp7.block<3,3>(0,0) = Rtcp7;
    adTtcp7.block<3,3>(3,3) = Rtcp7;
    adTtcp7.block<3,3>(0,3) = skewSymmetric(ptcp7) * Rtcp7;


Eigen::Vector3d endowrist_position = flange_position + flange_rotation * endowrist_offset; // Endowrist position
Eigen::Matrix3d endowrist_rotation = flange_rotation * endowrist_orientation.toRotationMatrix(); // Endowrist orientation
Eigen::Vector3d tcp_position = endowrist_position + endowrist_rotation * tcp_offset; // TCP position

Eigen::Matrix<double, 6, 6> adTflange_tcp = Eigen::Matrix<double, 6, 6>::Zero();
adTflange_tcp.block<3,3>(0,0) = Eigen::Matrix3d::Identity();
adTflange_tcp.block<3,3>(3,3) = Eigen::Matrix3d::Identity();
adTflange_tcp.block<3,3>(0,3) = skewSymmetric(Vector3d(0.005,0.61,0.0625));
Eigen::Matrix<double, 6, 7> J_tcp = adTflange_tcp.inverse() * jacobian_flange;

// std::cout<<"tcp_position: "<<tcp_position.transpose()<<" p0tcp:"<<p0tcp.transpose()<<std::endl;

std::vector<franka::Frame> joint_frames = {
    franka::Frame::kJoint1,
    franka::Frame::kJoint2,
    franka::Frame::kJoint3,
    franka::Frame::kJoint4,
    franka::Frame::kJoint5,
    franka::Frame::kJoint6,
    franka::Frame::kJoint7,
};

std::vector<Eigen::Matrix4d> joint_poses;
for (const auto& joint_frame : joint_frames) {
    Eigen::Map<const Eigen::Matrix4d> joint_pose(
        franka_robot_model_->getPoseMatrix(joint_frame).data());
    joint_poses.push_back(joint_pose);
}

Eigen::Vector3d p_flange = flange_position; // flange position
Eigen::Vector3d p_tcp = tcp_position; // TCP position
Eigen::Vector3d p_endo = endowrist_position; // Endowrist position
Eigen::Matrix<double, 6, 7> geometric_jacobian_flange = Eigen::Matrix<double, 6, 7>::Zero();
Eigen::Matrix<double, 6, 7> geometric_jacobian_tcp = Eigen::Matrix<double, 6, 7>::Zero();
Eigen::Matrix<double, 6, 7> geometric_jacobian_endo = Eigen::Matrix<double, 6, 7>::Zero();
for (size_t i = 0; i < joint_frames.size(); ++i) {
    Eigen::Vector3d z_i = joint_poses[i].block<3,3>(0,0).col(2); // Joint axis
    Eigen::Vector3d p_i = joint_poses[i].block<3,1>(0,3); // Joint position
    Eigen::Vector3d J_linear = z_i.cross(p_flange - p_i); // Linear velocity contribution
    Eigen::Vector3d J_angular = z_i; // Angular velocity contribution

    geometric_jacobian_flange.block<3,1>(0,i) = J_linear;
    geometric_jacobian_flange.block<3,1>(3,i) = J_angular;
}

for (size_t i = 0; i < joint_frames.size(); ++i) {
    Eigen::Vector3d z_i = joint_poses[i].block<3,3>(0,0).col(2); // Joint axis
    Eigen::Vector3d p_i = joint_poses[i].block<3,1>(0,3); // Joint position
    Eigen::Vector3d J_linear = z_i.cross(p_tcp - p_i); // Linear velocity contribution
    Eigen::Vector3d J_angular = z_i; // Angular velocity contribution

    geometric_jacobian_tcp.block<3,1>(0,i) = J_linear;
    geometric_jacobian_tcp.block<3,1>(3,i) = J_angular;
}

for (size_t i = 0; i < joint_frames.size(); ++i) {
    Eigen::Vector3d z_i = joint_poses[i].block<3,3>(0,0).col(2); // Joint axis
    Eigen::Vector3d p_i = joint_poses[i].block<3,1>(0,3); // Joint position
    Eigen::Vector3d J_linear = z_i.cross(p_endo - p_i); // Linear velocity contribution
    Eigen::Vector3d J_angular = z_i; // Angular velocity contribution

    geometric_jacobian_endo.block<3,1>(0,i) = J_linear;
    geometric_jacobian_endo.block<3,1>(3,i) = J_angular;
}

Eigen::Matrix<double, 6, 7> jacobian_tcp = adTtcp7 * geometric_jacobian_flange;

Eigen::Matrix<double, 6, 6> adT7endo = Eigen::Matrix<double, 6, 6>::Zero();
adT7endo.block<3,3>(0,0) = endowrist_orientation.toRotationMatrix();
adT7endo.block<3,3>(3,3) = endowrist_orientation.toRotationMatrix();
adT7endo.block<3,3>(0,3) = skewSymmetric(endowrist_offset) * endowrist_orientation.toRotationMatrix();

Eigen::Matrix<double, 6, 7> jacobian_endo = adT7endo * geometric_jacobian_flange;


// Twist adjoint transformation validation
Vector6d V_0flange = geometric_jacobian_flange * qD;
Vector6d V_0endo = geometric_jacobian_endo * qD;
Vector6d V_0tcp = geometric_jacobian_tcp * qD;

Eigen::Matrix<double, 6, 6> adT87_computed 
      = geometric_jacobian_endo*geometric_jacobian_flange.transpose()
      *(geometric_jacobian_flange*geometric_jacobian_flange.transpose()).inverse();
Vector6d V_0endo_adj_computed = Vector6d::Zero();
Vector6d V_0endo_adj = Vector6d::Zero();
V_0endo_adj_computed = adT87_computed * V_0flange;
V_0endo_adj = adT7endo * V_0flange;


// std::cout<< "T7tcp "<<T7tcp<<std::endl;

// std::cout<<"adT87_computed: \n"<<adT87_computed<<std::endl;
// std::cout<<"adT87: \n"<<adT7endo.inverse()<<std::endl;

// auto T78 = flange_pose.inverse()*T0endo;
// std::cout<<"T78: \n"<<T78<<std::endl;
// auto p87_computed = adT87_computed.block<3,3>(0,3)*adT87_computed.block<3,3>(0,0).transpose();
// std::cout<<"p87_computed: \n"<<p87_computed<<std::endl;

// Eigen::Matrix<double, 4, 4> Vhat_0endo = Eigen::Matrix<double, 4, 4>::Zero();
// Vhat_0endo.block<3,3>(0,0) = skewSymmetric(V_0endo.block<3,1>(0,0));
// Vhat_0endo.block<3,1>(0,3) = V_0endo.block<3,1>(0,0);
// Eigen::Matrix<double, 4, 4> Vhat_0flange = Eigen::Matrix<double, 4, 4>::Zero();
// Vhat_0flange.block<3,3>(0,0) = skewSymmetric(V_0flange.block<3,1>(0,0));
// Vhat_0flange.block<3,1>(0,3) = V_0flange.block<3,1>(0,0);
// Eigen::Matrix<double, 4, 4> Vhat_0endo_adj = T7endo * Vhat_0flange * T7endo.inverse();

// std::cout<<"Vhat_0endo: \n"<<Vhat_0endo<<"\nVhat_0endo_adj: \n"<<Vhat_0endo_adj<<std::endl;


// std::cout<<"V_0endo: "<<V_0endo.transpose()<<" V_0endo_adj_computed: "<<V_0endo_adj_computed.transpose()<<" V_0endo_adj: "<<V_0endo_adj.transpose()<<std::endl;
// std::cout<<"V_0tcp: "<<V_0tcp.transpose()<<" V_0tcp_adj: "<<V_0tcp_adj.transpose()<<std::endl;



// Print the manually computed Jacobian
// std::cout << "Geometric Jacobian: \n" << geometric_jacobian_tcp << std::endl;
// std::cout << "Jacobian TCP: \n" << J_tcp << std::endl;

// std::cout << "Geometric Jacobian_endo: \n" << geometric_jacobian_endo << std::endl;
// std::cout << "Jacobian Endo: \n" << jacobian_endo << std::endl;

// std::cout<<"jacobian_flange: \n"<<jacobian_flange<<std::endl;
// std::cout<<"geometric_jacobian_flange: \n"<<geometric_jacobian_flange<<std::endl;

// std::cout << "J_tcp: \n" << J_tcp << std::endl;
// std::cout << "ERROR: \n" << (geometric_jacobian_tcp - jacobian_tcp).norm() << std::endl;

// std::cout<<"R7tcp\n"<<T7tcp.block<3,3>(0,0)<<std::endl;
// std::cout<<"R7tcp_1\n"<<endowrist_orientation.toRotationMatrix()<<std::endl;
// std::cout<<"p7tcp"<<T7tcp.block<3,1>(0,3).transpose()<<std::endl;
// std::cout<<"p7tcp_1"<<tcp_offset.transpose()<<std::endl;


    current_position = tcp_position;
    jacobian = geometric_jacobian_tcp;


    Eigen::Matrix<double, 6, 7> J_c_N = Eigen::Matrix<double, 6, 7>::Zero();
    for (size_t i = 0; i < joint_frames.size(); ++i) {
        Eigen::Vector3d z_i = joint_poses[i].block<3,3>(0,0).col(2); // Joint axis
        Eigen::Vector3d p_i = joint_poses[i].block<3,1>(0,3); // Joint position
        Eigen::Vector3d J_linear = z_i.cross(pc_ - p_i); // Linear velocity contribution
        Eigen::Vector3d J_angular = z_i; // Angular velocity contribution

        J_c_N.block<3,1>(0,i) = J_linear;
        J_c_N.block<3,1>(3,i) = J_angular;
    }
    Eigen::Matrix<double, 6, 6> AdT_c_flange = J_c_N*geometric_jacobian_flange.transpose()
      *((geometric_jacobian_flange*geometric_jacobian_flange.transpose()).inverse());
    
    Eigen::Matrix<double, 6, 6> R_block_N_flange = Eigen::Matrix<double, 6, 6>::Zero();
    R_block_N_flange.block<3,3>(0,0) = flange_rotation;
    R_block_N_flange.block<3,3>(3,3) = flange_rotation;

    Eigen::Matrix<double, 6, 6> H = Eigen::Matrix<double, 6, 6>::Zero();
    H.block<2,2>(0,0) = Eigen::Matrix2d::Identity();
    Eigen::Matrix<double, 6, 7> J_c =
       H*R_block_N_flange.transpose()*AdT_c_flange*geometric_jacobian_flange;
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
    tau_d <<  tauf;
    for (int i = 0; i < num_joints; ++i) {
      command_interfaces_[i].set_value(tau_d(i));
    }
    std::cout<<"time: "<<time.seconds()<<" tau_d: "<<tau_d.transpose()<<std::endl;

    // Check if Jacobian drops rank
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(jacobian, Eigen::ComputeThinU | Eigen::ComputeThinV);
    double tolerance = 1e-6;  // Define a tolerance for singular values
    Eigen::VectorXd singular_values = svd.singularValues();
    // std::cout << "Singular values of the Jacobian: " << singular_values.transpose() << std::endl;
    if (singular_values.minCoeff() < tolerance) {
      std::cout << "Jacobian has dropped rank!" << std::endl;
    }
    
    recorder_->addToRec(time.seconds());
    recorder_->addToRec(p0tcp);
    Vector3d xdes = desired.head<3>();
    recorder_->addToRec(xdes);
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
  prev_R78 = endowrist_orientation.toRotationMatrix(); // Previous endowrist orientation
  // desired_position(0) += 0.4; // Modify the x-coordinate
  // desired_position(1) += 0.45; // Modify the y-coordinate
  // desired_position(2) -= 0.3; // Set the z-coordinate to a specific value
  // pc_(0) += 0.4; // Modify the x-coordinate
  // pc_(1) += 0.45; // Modify the y-coordinate
  // pc_(2) -= 0.3; // Set the z-coordinate to a specific value

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