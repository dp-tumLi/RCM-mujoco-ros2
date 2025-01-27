% Parameters
T = 20;            % Total trajectory duration (seconds)
dt = 0.001;         % Time step
time = 0:dt:T;     % Time vector
p0 = [0; 0; 0];    % Initial position

% Initialize storage
trajectory = zeros(9, length(time));

% Compute the trajectory
for i = 1:length(time)
    trajectory(:, i) = spiralTrajectory(T, time(i), p0);
end

% Extract position
position = trajectory(1:3, :);

% Plot the 3D spiral trajectory
figure;
plot3(position(1, :), position(2, :), position(3, :), 'LineWidth', 1.5);
xlabel('X (m)');
ylabel('Y (m)');
zlabel('Z (m)');
grid on;
title('3D Spiral Trajectory');


function des = spiralTrajectory(T, t, p0)
    % Inputs:
    % T  - Total trajectory duration
    % t  - Current time
    % p0 - Initial position as a 3x1 vector [x0; y0; z0]
    %
    % Outputs:
    % des - Desired position, velocity, and acceleration as a 9x1 vector
    %       des(1:3) = position
    %       des(4:6) = velocity
    %       des(7:9) = acceleration

    % Parameters
    pitch = 0.015; % Spiral pitch
    R = 0.1;      % Spiral radius
    L = 8 * pi;    % Total path length

    % Timing law parameters
    v_max = 1.25 * L / T;  % Maximum velocity
    a_max = v_max^2 / (T * v_max - L); % Maximum acceleration
    Tm = v_max / a_max;    % Time to reach max velocity

    % Error handling for invalid trajectory timing
    if Tm < T / 5 || Tm > T / 2.1
        error('HS: ERROR in trajectory planning timing law');
    end

    % Compute S, Sd, and Sdd based on timing law
    if t >= 0 && t <= Tm
        S = a_max * t^2 / 2;
        Sd = a_max * t;
        Sdd = a_max;
    elseif t > Tm && t <= (T - Tm)
        S = v_max * t - v_max^2 / (2 * a_max);
        Sd = v_max;
        Sdd = 0;
    elseif t > (T - Tm) && t <= T
        S = -a_max * (t - T)^2 / 2 + v_max * T - v_max^2 / a_max;
        Sd = -a_max * (t - T);
        Sdd = -a_max;
    else
        S = L;
        Sd = 0;
        Sdd = 0;
    end

    % Geometric path: 3D spiral with z-axis displacement
    des = zeros(9, 1);
    des(1) = (p0(1) - R) + R * cos(S);                  % Position x
    des(2) = p0(2) + R * sin(S);                       % Position y
    des(3) = p0(3) + S * pitch / (2 * pi);             % Position z
    des(4) = -R * Sd * sin(S);                         % Velocity x
    des(5) = R * Sd * cos(S);                          % Velocity y
    des(6) = pitch * Sd / (2 * pi);                    % Velocity z
    des(7) = -R * Sdd * sin(S) - R * Sd^2 * cos(S);    % Acceleration x
    des(8) = R * Sdd * cos(S) - R * Sd^2 * sin(S);     % Acceleration y
    des(9) = pitch * Sdd / (2 * pi);                   % Acceleration z
end
