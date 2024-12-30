#include <franka_msgs/msg/franka_state.hpp>
#include <franka/robot_state.h>

using franka_msgs::msg::FrankaState;
using franka_msgs::msg::Errors;
using franka::RobotState;

// A utility file for getting a franka_msg::msg::FrankaState given a franka::RobotState

namespace{

franka_msgs::msg::Errors errorsToMessage(const franka::Errors& error) {
  franka_msgs::msg::Errors message;
  message.joint_position_limits_violation =
      static_cast<decltype(message.joint_position_limits_violation)>(
          error.joint_position_limits_violation);
  message.cartesian_position_limits_violation =
      static_cast<decltype(message.cartesian_position_limits_violation)>(
          error.cartesian_position_limits_violation);
  message.self_collision_avoidance_violation =
      static_cast<decltype(message.self_collision_avoidance_violation)>(
          error.self_collision_avoidance_violation);
  message.joint_velocity_violation =
      static_cast<decltype(message.joint_velocity_violation)>(error.joint_velocity_violation);
  message.cartesian_velocity_violation =
      static_cast<decltype(message.cartesian_velocity_violation)>(
          error.cartesian_velocity_violation);
  message.force_control_safety_violation =
      static_cast<decltype(message.force_control_safety_violation)>(
          error.force_control_safety_violation);
  message.joint_reflex = static_cast<decltype(message.joint_reflex)>(error.joint_reflex);
  message.cartesian_reflex =
      static_cast<decltype(message.cartesian_reflex)>(error.cartesian_reflex);
  message.max_goal_pose_deviation_violation =
      static_cast<decltype(message.max_goal_pose_deviation_violation)>(
          error.max_goal_pose_deviation_violation);
  message.max_path_pose_deviation_violation =
      static_cast<decltype(message.max_path_pose_deviation_violation)>(
          error.max_path_pose_deviation_violation);
  message.cartesian_velocity_profile_safety_violation =
      static_cast<decltype(message.cartesian_velocity_profile_safety_violation)>(
          error.cartesian_velocity_profile_safety_violation);
  message.joint_position_motion_generator_start_pose_invalid =
      static_cast<decltype(message.joint_position_motion_generator_start_pose_invalid)>(
          error.joint_position_motion_generator_start_pose_invalid);
  message.joint_motion_generator_position_limits_violation =
      static_cast<decltype(message.joint_motion_generator_position_limits_violation)>(
          error.joint_motion_generator_position_limits_violation);
  message.joint_motion_generator_velocity_limits_violation =
      static_cast<decltype(message.joint_motion_generator_velocity_limits_violation)>(
          error.joint_motion_generator_velocity_limits_violation);
  message.joint_motion_generator_velocity_discontinuity =
      static_cast<decltype(message.joint_motion_generator_velocity_discontinuity)>(
          error.joint_motion_generator_velocity_discontinuity);
  message.joint_motion_generator_acceleration_discontinuity =
      static_cast<decltype(message.joint_motion_generator_acceleration_discontinuity)>(
          error.joint_motion_generator_acceleration_discontinuity);
  message.cartesian_position_motion_generator_start_pose_invalid =
      static_cast<decltype(message.cartesian_position_motion_generator_start_pose_invalid)>(
          error.cartesian_position_motion_generator_start_pose_invalid);
  message.cartesian_motion_generator_elbow_limit_violation =
      static_cast<decltype(message.cartesian_motion_generator_elbow_limit_violation)>(
          error.cartesian_motion_generator_elbow_limit_violation);
  message.cartesian_motion_generator_velocity_limits_violation =
      static_cast<decltype(message.cartesian_motion_generator_velocity_limits_violation)>(
          error.cartesian_motion_generator_velocity_limits_violation);
  message.cartesian_motion_generator_velocity_discontinuity =
      static_cast<decltype(message.cartesian_motion_generator_velocity_discontinuity)>(
          error.cartesian_motion_generator_velocity_discontinuity);
  message.cartesian_motion_generator_acceleration_discontinuity =
      static_cast<decltype(message.cartesian_motion_generator_acceleration_discontinuity)>(
          error.cartesian_motion_generator_acceleration_discontinuity);
  message.cartesian_motion_generator_elbow_sign_inconsistent =
      static_cast<decltype(message.cartesian_motion_generator_elbow_sign_inconsistent)>(
          error.cartesian_motion_generator_elbow_sign_inconsistent);
  message.cartesian_motion_generator_start_elbow_invalid =
      static_cast<decltype(message.cartesian_motion_generator_start_elbow_invalid)>(
          error.cartesian_motion_generator_start_elbow_invalid);
  message.cartesian_motion_generator_joint_position_limits_violation =
      static_cast<decltype(message.cartesian_motion_generator_joint_position_limits_violation)>(
          error.cartesian_motion_generator_joint_position_limits_violation);
  message.cartesian_motion_generator_joint_velocity_limits_violation =
      static_cast<decltype(message.cartesian_motion_generator_joint_velocity_limits_violation)>(
          error.cartesian_motion_generator_joint_velocity_limits_violation);
  message.cartesian_motion_generator_joint_velocity_discontinuity =
      static_cast<decltype(message.cartesian_motion_generator_joint_velocity_discontinuity)>(
          error.cartesian_motion_generator_joint_velocity_discontinuity);
  message.cartesian_motion_generator_joint_acceleration_discontinuity =
      static_cast<decltype(message.cartesian_motion_generator_joint_acceleration_discontinuity)>(
          error.cartesian_motion_generator_joint_acceleration_discontinuity);
  message.cartesian_position_motion_generator_invalid_frame =
      static_cast<decltype(message.cartesian_position_motion_generator_invalid_frame)>(
          error.cartesian_position_motion_generator_invalid_frame);
  message.force_controller_desired_force_tolerance_violation =
      static_cast<decltype(message.force_controller_desired_force_tolerance_violation)>(
          error.force_controller_desired_force_tolerance_violation);
  message.controller_torque_discontinuity =
      static_cast<decltype(message.controller_torque_discontinuity)>(
          error.controller_torque_discontinuity);
  message.start_elbow_sign_inconsistent =
      static_cast<decltype(message.start_elbow_sign_inconsistent)>(
          error.start_elbow_sign_inconsistent);
  message.communication_constraints_violation =
      static_cast<decltype(message.communication_constraints_violation)>(
          error.communication_constraints_violation);
  message.power_limit_violation =
      static_cast<decltype(message.power_limit_violation)>(error.power_limit_violation);
  message.joint_p2p_insufficient_torque_for_planning =
      static_cast<decltype(message.joint_p2p_insufficient_torque_for_planning)>(
          error.joint_p2p_insufficient_torque_for_planning);
  message.tau_j_range_violation =
      static_cast<decltype(message.tau_j_range_violation)>(error.tau_j_range_violation);
  message.instability_detected =
      static_cast<decltype(message.instability_detected)>(error.instability_detected);
  message.joint_move_in_wrong_direction =
      static_cast<decltype(message.joint_move_in_wrong_direction)>(error.joint_move_in_wrong_direction);
  
  // base acceleration tag needs to be added; seems like that was removed for FR3
  return message;
}

FrankaState getRobotStateMsg(const RobotState& robot_state) {
    FrankaState message;
    
  static_assert(
      sizeof(robot_state.cartesian_collision) == sizeof(robot_state.cartesian_contact),
      "Robot state Cartesian members do not have same size");
  static_assert(
      sizeof(robot_state.cartesian_collision) == sizeof(robot_state.K_F_ext_hat_K),
      "Robot state Cartesian members do not have same size");
  static_assert(
      sizeof(robot_state.cartesian_collision) == sizeof(robot_state.O_F_ext_hat_K),
      "Robot state Cartesian members do not have same size");
  static_assert(sizeof(robot_state.cartesian_collision) == sizeof(robot_state.O_dP_EE_d),
                "Robot state Cartesian members do not have same size");
  static_assert(sizeof(robot_state.cartesian_collision) == sizeof(robot_state.O_dP_EE_c),
                "Robot state Cartesian members do not have same size");
  static_assert(sizeof(robot_state.cartesian_collision) == sizeof(robot_state.O_ddP_EE_c),
                "Robot state Cartesian members do not have same size");
  for (size_t i = 0; i < robot_state.cartesian_collision.size(); i++) {
    message.cartesian_collision[i] = robot_state.cartesian_collision[i];
    message.cartesian_contact[i] = robot_state.cartesian_contact[i];
    message.k_f_ext_hat_k[i] = robot_state.K_F_ext_hat_K[i];
    message.o_f_ext_hat_k[i] = robot_state.O_F_ext_hat_K[i];
    message.o_dp_ee_d[i] = robot_state.O_dP_EE_d[i];
    message.o_dp_ee_c[i] = robot_state.O_dP_EE_c[i];
    message.o_ddp_ee_c[i] = robot_state.O_ddP_EE_c[i];
  }

  static_assert(sizeof(robot_state.q) == sizeof(robot_state.q_d),
                "Robot state joint members do not have same size");
  static_assert(sizeof(robot_state.q) == sizeof(robot_state.dq),
                "Robot state joint members do not have same size");
  static_assert(sizeof(robot_state.q) == sizeof(robot_state.dq_d),
                "Robot state joint members do not have same size");
  static_assert(sizeof(robot_state.q) == sizeof(robot_state.ddq_d),
                "Robot state joint members do not have same size");
  static_assert(sizeof(robot_state.q) == sizeof(robot_state.tau_J),
                "Robot state joint members do not have same size");
  static_assert(sizeof(robot_state.q) == sizeof(robot_state.dtau_J),
                "Robot state joint members do not have same size");
  static_assert(sizeof(robot_state.q) == sizeof(robot_state.tau_J_d),
                "Robot state joint members do not have same size");
  static_assert(sizeof(robot_state.q) == sizeof(robot_state.theta),
                "Robot state joint members do not have same size");
  static_assert(sizeof(robot_state.q) == sizeof(robot_state.dtheta),
                "Robot state joint members do not have same size");
  static_assert(sizeof(robot_state.q) == sizeof(robot_state.joint_collision),
                "Robot state joint members do not have same size");
  static_assert(sizeof(robot_state.q) == sizeof(robot_state.joint_contact),
                "Robot state joint members do not have same size");
  static_assert(sizeof(robot_state.q) == sizeof(robot_state.tau_ext_hat_filtered),
                "Robot state joint members do not have same size");
  for (size_t i = 0; i < robot_state.q.size(); i++) {
    message.q[i] = robot_state.q[i];
    message.q_d[i] = robot_state.q_d[i];
    message.dq[i] = robot_state.dq[i];
    message.dq_d[i] = robot_state.dq_d[i];
    message.ddq_d[i] = robot_state.ddq_d[i];
    message.tau_j[i] = robot_state.tau_J[i];
    message.dtau_j[i] = robot_state.dtau_J[i];
    message.tau_j_d[i] = robot_state.tau_J_d[i];
    message.theta[i] = robot_state.theta[i];
    message.dtheta[i] = robot_state.dtheta[i];
    message.joint_collision[i] = robot_state.joint_collision[i];
    message.joint_contact[i] = robot_state.joint_contact[i];
    message.tau_ext_hat_filtered[i] = robot_state.tau_ext_hat_filtered[i];
  }

  static_assert(sizeof(robot_state.elbow) == sizeof(robot_state.elbow_d),
                "Robot state elbow configuration members do not have same size");
  static_assert(sizeof(robot_state.elbow) == sizeof(robot_state.elbow_c),
                "Robot state elbow configuration members do not have same size");
  static_assert(sizeof(robot_state.elbow) == sizeof(robot_state.delbow_c),
                "Robot state elbow configuration members do not have same size");
  static_assert(sizeof(robot_state.elbow) == sizeof(robot_state.ddelbow_c),
                "Robot state elbow configuration members do not have same size");

  for (size_t i = 0; i < robot_state.elbow.size(); i++) {
    message.elbow[i] = robot_state.elbow[i];
    message.elbow_d[i] = robot_state.elbow_d[i];
    message.elbow_c[i] = robot_state.elbow_c[i];
    message.delbow_c[i] = robot_state.delbow_c[i];
    message.ddelbow_c[i] = robot_state.ddelbow_c[i];
  }

  static_assert(sizeof(robot_state.O_T_EE) == sizeof(robot_state.F_T_EE),
                "Robot state transforms do not have same size");
  static_assert(sizeof(robot_state.O_T_EE) == sizeof(robot_state.F_T_NE),
                  "Robot state transforms do not have same size");
  static_assert(sizeof(robot_state.O_T_EE) == sizeof(robot_state.NE_T_EE),
                  "Robot state transforms do not have same size");
  static_assert(sizeof(robot_state.O_T_EE) == sizeof(robot_state.EE_T_K),
                "Robot state transforms do not have same size");
  static_assert(sizeof(robot_state.O_T_EE) == sizeof(robot_state.O_T_EE_d),
                "Robot state transforms do not have same size");
  static_assert(sizeof(robot_state.O_T_EE) == sizeof(robot_state.O_T_EE_c),
                "Robot state transforms do not have same size");
  for (size_t i = 0; i < robot_state.O_T_EE.size(); i++) {
    message.o_t_ee[i] = robot_state.O_T_EE[i];
    message.f_t_ee[i] = robot_state.F_T_EE[i];
    message.f_t_ne[i] = robot_state.F_T_NE[i];
    message.ne_t_ee[i] = robot_state.NE_T_EE[i];
    message.ee_t_k[i] = robot_state.EE_T_K[i];
    message.o_t_ee_d[i] = robot_state.O_T_EE_d[i];
    message.o_t_ee_c[i] = robot_state.O_T_EE_c[i];
  }
  message.m_ee = robot_state.m_ee;
  message.m_load = robot_state.m_load;
  message.m_total = robot_state.m_total;

  for (size_t i = 0; i < robot_state.I_load.size(); i++) {
    message.i_ee[i] = robot_state.I_ee[i];
    message.i_load[i] = robot_state.I_load[i];
    message.i_total[i] = robot_state.I_total[i];
  }

  for (size_t i = 0; i < robot_state.F_x_Cload.size(); i++) {
    message.f_x_cee[i] = robot_state.F_x_Cee[i];
    message.f_x_cload[i] = robot_state.F_x_Cload[i];
    message.f_x_ctotal[i] = robot_state.F_x_Ctotal[i];
  }

  message.time = robot_state.time.toSec();
  message.control_command_success_rate = robot_state.control_command_success_rate;
  message.current_errors = errorsToMessage(robot_state.current_errors);
  message.last_motion_errors = errorsToMessage(robot_state.last_motion_errors);

  switch (robot_state.robot_mode) {
    case franka::RobotMode::kOther:
      message.robot_mode = franka_msgs::msg::FrankaState::ROBOT_MODE_OTHER;
      break;

    case franka::RobotMode::kIdle:
      message.robot_mode = franka_msgs::msg::FrankaState::ROBOT_MODE_IDLE;
      break;

    case franka::RobotMode::kMove:
      message.robot_mode = franka_msgs::msg::FrankaState::ROBOT_MODE_MOVE;
      break;

    case franka::RobotMode::kGuiding:
      message.robot_mode = franka_msgs::msg::FrankaState::ROBOT_MODE_GUIDING;
      break;

    case franka::RobotMode::kReflex:
      message.robot_mode = franka_msgs::msg::FrankaState::ROBOT_MODE_REFLEX;
      break;

    case franka::RobotMode::kUserStopped:
      message.robot_mode = franka_msgs::msg::FrankaState::ROBOT_MODE_USER_STOPPED;
      break;

    case franka::RobotMode::kAutomaticErrorRecovery:
      message.robot_mode = franka_msgs::msg::FrankaState::ROBOT_MODE_AUTOMATIC_ERROR_RECOVERY;
      break;
  }
  return message;
}

}