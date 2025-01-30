% run("LICENSE.m");
clc
close all
clear
%%
run("DATA.m")
% Data = DATAm;
Data=DATAm(any(DATAm,2),:);  % to remove zero rows

time = Data(:,1);
ptip_x = Data(:,2);
ptip_y = Data(:,3);
ptip_z = Data(:,4);

ptip_x_des = Data(:,5);
ptip_y_des = Data(:,6);
ptip_z_des = Data(:,7);

% pc_x = Data(:,8);
% pc_y = Data(:,9);
% pc_z = Data(:,10);
%%
figure
start_t = 1;
plot3(ptip_x(start_t:end),ptip_y(start_t:end),ptip_z(start_t:end),'--',"LineWidth",2)
hold on
plot3(ptip_x_des(1:end),ptip_y_des(1:end),ptip_z_des(1:end),'.')
hold on
% plot3(pc_x,pc_y,pc_z, '+k', 'MarkerSize', 1);


xlabel("x")
ylabel("y")
zlabel("z")