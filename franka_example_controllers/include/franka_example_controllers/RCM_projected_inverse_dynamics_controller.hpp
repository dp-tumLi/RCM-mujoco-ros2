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

#pragma once

#include <string>

#include <Eigen/Eigen>
#include <controller_interface/controller_interface.hpp>
#include "franka_semantic_components/franka_robot_model.hpp"
#include <rclcpp/rclcpp.hpp>
#include <Eigen/Dense>
#include "franka_example_controllers/recorder.hpp"
#include <yaml-cpp/yaml.h>  // Include the yaml-cpp library

using CallbackReturn = rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn;

namespace franka_example_controllers {
  using Eigen::Matrix3d;
  using Matrix4d = Eigen::Matrix<double, 4, 4>;
  using Matrix6d = Eigen::Matrix<double, 6, 6>;
  using Matrix7d = Eigen::Matrix<double, 7, 7>;

  using Vector3d = Eigen::Matrix<double, 3, 1>;
  using Vector6d = Eigen::Matrix<double, 6, 1>;
  using Vector7d = Eigen::Matrix<double, 7, 1>;
  
  using Eigen::Quaterniond;

/**
 * The cartesian impedance example controller implements the Hogan formulation.
 */
class ProjectedInverseDynamicsController : public controller_interface::ControllerInterface {
 public:
  using Vector7d = Eigen::Matrix<double, 7, 1>;
  controller_interface::InterfaceConfiguration command_interface_configuration() const override;
  controller_interface::InterfaceConfiguration state_interface_configuration() const override;
  controller_interface::return_type update(const rclcpp::Time& time,
                                           const rclcpp::Duration& period) override;
  CallbackReturn on_init() override;
  CallbackReturn on_configure(const rclcpp_lifecycle::State& previous_state) override;
  CallbackReturn on_activate(const rclcpp_lifecycle::State& previous_state) override;
  CallbackReturn on_deactivate(const rclcpp_lifecycle::State& previous_state) override;
  ProjectedInverseDynamicsController();

 private:
  std::string arm_id_;
  const int num_joints = 7;
  std::unique_ptr<franka_semantic_components::FrankaRobotModel> franka_robot_model_;
  rclcpp::Time start_time_;
  Quaterniond desired_orientation;
  Vector3d desired_position;
  Vector7d desired_qn;
  Matrix4d desired;
  Eigen::Matrix<double, 6, 7> previous_jacobian_;
  Eigen::Matrix<double, 3, 3> prev_R78;
  Vector3d pc_;
  Matrix6d stiffness;
  Matrix6d damping;
  double n_stiffness;
  std::unique_ptr<Recorder> recorder_;
  void loadConfig(const std::string& config_file);
  static void signalHandler(int signum);
  static Recorder* static_recorder_;
};

}  // namespace franka_example_controllers