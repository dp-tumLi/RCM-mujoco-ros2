#include <multi_mode_controller/controllers/panda_joint_impedance_controller.h>

#include <cmath>

#include <multi_mode_controller/utils/controller_factory.h>

using namespace panda_controllers;
using Vector7d = Eigen::Matrix<double, 7, 1>;
using Pose = PandaJointImpedanceControllerPose;
using Params = double;
using GoalMsg = multi_mode_control_msgs::msg::JointGoal;
using ConfigRequest = multi_mode_control_msgs::srv::SetJointImpedance::Request;
using ConfigResponse = multi_mode_control_msgs::srv::SetJointImpedance::Response;
using Controller = PandaJointImpedanceController;

static auto registration = ControllerFactory::registerClass<Controller>(
    "panda_joint_impedance_controller");

bool Controller::desiredPoseCallbackImpl(Pose& p_d, 
                                         const Pose& p,
                                         const GoalMsg& msg) {
  p_d.q = Vector7d(msg.q.data());
  for (int i=0;i<7;++i) {
    if (std::abs(p_d.q[i]+this->getOffset().q[i]-p.q[i]) > 0.1) {
      auto& clk = *this->node_->get_clock();
      RCLCPP_WARN_THROTTLE(this->node_->get_logger(), clk, 1000, "panda_joint_impedance_controller: Discarding "
          "target pose that is too far away from current pose (%f rad, allowed "
          "maximum is 0.1 rad) in joint %i.",
          std::abs(p_d.q[i]+this->getOffset().q[i]-p.q[i]), i+1);
      return false;
    }
  }
  p_d.qD = Vector7d(msg.qd.data());
  return true;
}

bool Controller::setParametersCallbackImpl(Params& p_d, const Params& p,
    const ConfigRequest::SharedPtr& request, const ConfigResponse::SharedPtr& response) {
  p_d = request->stiffness_scale;
  return true;
}
