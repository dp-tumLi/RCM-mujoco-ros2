#include <franka_example_controllers/desired_trajectory.hpp>


void spiralTrajectory(double T, double t, Eigen::Matrix<double, 3, 1> &p0, Eigen::Matrix<double, 9, 1> &des)
{
    double xi[3];
    double L, v_max, a_max, Tm, S, Sd, Sdd;

    xi[0] = p0[0];
    xi[1] = p0[1];
    xi[2] = p0[2];


    double pitch = 0.015; //0.015
    double R = 0.1; //0.015
    L = 8 * M_PI;

    v_max = 1.25 * L / T; // TODO

    a_max = v_max * v_max / (T * v_max - L);
    Tm = v_max / a_max;

    if (Tm < T / 5 || Tm > T / 2.1)
    {
        std::cout << "HS: ERROR in trajectory planning timing law" << std::endl;
        exit(1);
    }

    if (t >= 0 && t <= Tm)
    {
        S = a_max * t * t / 2;
        Sd = a_max * t;
        Sdd = a_max;
    }
    else if (t >= Tm && t <= (T - Tm))
    {
        S = v_max * t - v_max * v_max / (2 * a_max);
        Sd = v_max;
        Sdd = 0;
    }
    else if (t >= (T - Tm) && t <= T)
    {
        S = -a_max * (t - T) * (t - T) / 2 + v_max * T - v_max * v_max / a_max;
        Sd = -a_max * (t - T);
        Sdd = -a_max;
    }
    else
    {
        S = L;
        Sd = 0;
        Sdd = 0;
    }
    // Geometric path
    //  spiral with z axis
    // p_des
    des[0] = (xi[0] - R) + R * cos(S);
    des[1] = xi[1] + R * sin(S);
    des[2] = xi[2] + S * pitch / (2 * M_PI);
    // pd_des
    des[3] = -R * Sd * sin(S);
    des[4] = R * Sd * cos(S);
    des[5] = pitch * Sd / (2 * M_PI);
    // pdd_des
    des[6] = -R * Sdd * sin(S) - R * Sd * Sd * cos(S);
    des[7] = R * Sdd * cos(S) - R * Sd * Sd * sin(S);
    des[8] = pitch * Sdd / (2 * M_PI);
}


Eigen::Matrix3d skewSymmetric(const Eigen::Vector3d& v) {
    Eigen::Matrix3d skew;
    skew <<   0, -v(2),  v(1),
            v(2),    0, -v(0),
           -v(1),  v(0),    0;
    return skew;
}

Eigen::Vector3d unskewSymmetric(const Eigen::Matrix3d& m) {
    Eigen::Vector3d v;
    v << m(2,1), m(0,2), m(1,0);
    return v;
}