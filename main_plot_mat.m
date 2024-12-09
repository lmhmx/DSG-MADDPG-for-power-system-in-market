close all
clear

record_version = "v_no_rl_001";
integral_recorder = load("./useful/"+record_version+"_integral_Controller.mat");
GNE_recorder_no_constraint = load("./useful/"+record_version+"_linear_Controller.mat");
GNE_recorder_constraint = load("./useful/"+record_version+"_linear_Constrained_Controller.mat");
GNE_recorder_constraint_with_hard = load("./useful/"+record_version+"_saturated_linear_Constrained_Controller.mat");

dt = 0.005;
figure(1)
set(gcf,'position',[100,100,780,320])

tmp_plot = tiledlayout(2,4,'TileSpacing','compact','Padding','tight');

nexttile(1,[2,2])
box on
grid on
hold on
set(gca,'FontSize',20)
plot((1:length(integral_recorder.omega))*dt, 60+60*integral_recorder.omega(:,53), "--",LineWidth=0.9, Color="r")
plot((1:length(GNE_recorder_no_constraint.omega))*dt, 60+60*GNE_recorder_no_constraint.omega(:,53), "-",LineWidth=0.9,Color="b")
axis([0,20-0.1,59.85,60.05])
xticks(linspace(0,20,6))
yticks(linspace(59.85,60.05,5))
yticklabels(split(sprintf("%4.2f ",linspace(59.85,60.05,5))))
ylabel("Frequency (Hz)")
xlabel("Time (s)")
legend(["Integral Controller", "Proposed Algorithm"],FontSize=18,Location="southeast")
title("Bus 53")

nexttile(3,[2,2])
set(gca,'FontSize',20)
box on
grid on
hold on
plot((1:length(integral_recorder.omega))*dt, 60+60*integral_recorder.omega(:,54), "--",LineWidth=0.9, Color="r")
plot((1:length(GNE_recorder_no_constraint.omega))*dt, 60+60*GNE_recorder_no_constraint.omega(:,54), "-",LineWidth=0.9,Color="b")
axis([0,20,59.85,60.05])
xticks(linspace(0,20,6))
yticks(linspace(59.85,60.05,5))
yticklabels([])
xlabel("Time (s)")
legend(["Integral Controller", "Proposed Algorithm"],FontSize=18,Location="southeast")
title("Bus 54")

% 
% set(gcf,'position',[100,100,800,450])
% set(gca,'FontSize',24)
% box on
% grid on
% hold on
% colororder([[1,0,0];[0,1,1];[0,0,1]])
% plot((1:length(integral_recorder.omega))*dt, 60+60*integral_recorder.omega(:,[53 54 55]), "--",LineWidth=0.9)
% plot((1:length(GNE_recorder_no_constraint.omega))*dt, 60+60*GNE_recorder_no_constraint.omega(:,[53 54 55]), "-",LineWidth=0.9)
% legend(["Bus 53 under Integral Controller","Bus 54 under Integral Controller","Bus 55 under Integral Controller", ...
%     "Bus 53 under Proposed Algorithm","Bus 54 under Proposed Algorithm","Bus 55 under Proposed Algorithm"],"NumColumns",1,"Location","best","FontSize",20)
% xlabel("Time (s)")
% ylabel("Frequency (Hz)")
% axis([0,20,59.8,60.04])
% xticks(linspace(0,20,11))
% yticks(linspace(59.8,60.04,7))
% yticklabels(split(sprintf("%4.2f ",linspace(59.8,60.04,7))))


figure(2)
set(gcf,'position',[100,100,1300,500])
set(gca,'FontSize',28)
box on
grid on
hold on
colororder([[1,0,0];
            [0.8,0.2,0];
            [0.6,0.4,0];
            [0,0,1.0];
            [0,0.2,0.8];
            [0,0.4,0.6]]);
plot((1:length(integral_recorder.u))*dt, integral_recorder.u(:,[1 2 3 53 54 55]), "--",LineWidth=0.9)
plot((1:length(GNE_recorder_no_constraint.u))*dt, GNE_recorder_no_constraint.u(:,[1 2 3 53 54 55]), "-",LineWidth=0.9)
legend(["Bus 1 under Integral Controller","Bus 2 under Integral Controller","Bus 3 under Integral Controller", ...
    "Bus 53 under Integral Controller","Bus 54 under Integral Controller","Bus 55 under Integral Controller", ...
    "Bus 1 under Proposed Algorithm","Bus 2 under Proposed Algorithm","Bus 3 under Proposed Algorithm", ...
    "Bus 53 under Proposed Algorithm","Bus 54 under Proposed Algorithm","Bus 55 under Proposed Algorithm"],"NumColumns",2,"Location","southeast",FontSize=20)
xlabel("Time (s)",FontSize=25)
ylabel("Controllable Power u (p.u.)",FontSize=25)
axis([0,20,-0.,0.6])
xticks(linspace(0,20,11))
yticks(linspace(-0.,0.6,5))
yticklabels(split(sprintf("%2.2f ",linspace(0.,0.6,5))))


figure(3)
set(gcf,'position',[100,100,900,450])
set(gca,'FontSize',24)
box on
grid on
hold on
colororder([[1,0,0];[0,1,1];[0,0,1];[0,0,0];[1,0,0];[0,1,1];[0,0,1]])
plot((1:length(GNE_recorder_no_constraint.u))*dt, GNE_recorder_no_constraint.u(:,[1 2 3]), "--",LineWidth=0.9)
plot((1:length(GNE_recorder_constraint.u))*dt, 0.5+0*(1:length(GNE_recorder_constraint.u)),"--",Color="black", LineWidth=0.7)

plot((1:length(GNE_recorder_constraint.u))*dt, GNE_recorder_constraint.u(:,[1 2 3]), "-",LineWidth=0.9)
plot((1:length(GNE_recorder_constraint.u))*dt, -[-0.4,-0.3]'+0*(1:length(GNE_recorder_constraint.u)),"--",Color="black", LineWidth=0.7)
legend(["Bus 1 without constraints","Bus 2 without constraints","Bus 3 without constraints","Constraints",...
    "Bus 1 with constraints","Bus 2 with constraints","Bus 3 with constraints"],"NumColumns",2, "Location","best",FontSize=20)
xlabel("Time (s)",FontSize=22)
ylabel("Controllable Power u (p.u.)",FontSize=22)
axis([0,20,-0.15,0.6])
xticks(linspace(0,20,11))
yticks(linspace(-0.15,0.6,6))
yticklabels(split(sprintf("%3.2f ",linspace(-0.15,0.6,6))))


figure(4)
set(gcf,'position',[100,100,1000,450])
set(gca,'FontSize',24)
box on
grid on
hold on
colororder([[1,0,0];[0,1,1];[0,0,1];[0,0,0];[1,0,0];[0,1,1];[0,0,1]])
plot((1:length(GNE_recorder_constraint.u))*dt, GNE_recorder_constraint.u(:,[1 2 3]), "--",LineWidth=0.9)
plot((1:length(GNE_recorder_constraint.u))*dt, 0.5+0*(1:length(GNE_recorder_constraint.u)),"--",Color="black", LineWidth=0.7)
plot((1:length(GNE_recorder_constraint_with_hard.u))*dt, GNE_recorder_constraint_with_hard.u(:,[1 2 3]), "-",LineWidth=0.9)

plot((1:length(GNE_recorder_constraint.u))*dt, -[-0.55,-0.45,-0.35]'+0*(1:length(GNE_recorder_constraint.u)),"-",Color="black", LineWidth=0.7)
plot((1:length(GNE_recorder_constraint.u))*dt, -[-0.4,-0.3]'+0*(1:length(GNE_recorder_constraint.u)),"--",Color="black", LineWidth=0.7)

legend(["Bus 1 without hard constraints","Bus 2 without hard constraints","Bus 3 without hard constraints","Soft constraints", ...
    "Bus 1 with hard constraints","Bus 2 with hard constraints","Bus 3 with hard constraints", "Hard constraints"],"NumColumns",2,"Location","best",FontSize=18)
xlabel("Time (s)",FontSize=22)
ylabel("Controllable Power u (p.u.)",FontSize=22)
axis([0,20,-0.15,0.6])
xticks(linspace(0,20,11))
yticks(linspace(-0.15,0.6,6))
yticklabels(split(sprintf("%3.2f ",linspace(-0.15,0.6,6))))


figure(5)

plot_saturated_function()

figure(6)
box on
grid on
hold on
set(gcf,'position',[100,100,800,450])
set(gca,'FontSize',24)
maddpg_results = load("./useful/v_maddpg_001.mat");

maddpg_returns = maddpg_results.test_return;

blues_color=[
[0.7161860822760477,0.8332026143790849,0.916155324875048];
[0.548235294117647,0.7529719338715878,0.866958861976163];
[0.366643598615917,0.6461822376009227,0.8185467128027681];
[0.22380622837370245,0.5375317185697808,0.7584313725490196];
[0.10557477893118032,0.41262591311034214,0.6859669357939254];
[0.03137254901960784,0.2897347174163783,0.570319108035371];];


% blues_color = flip(blues_color);
% colororder([[1,0,0];[0,0,0.4];[0,0,0.5];[0,0,0.6];[0,0,0.7];[0,0,0.8];[0,0,1]])
colororder([[1,0,0];blues_color])
% colororder(map)
% monotone_colors = [[0,0,0.2];[0,0,0.4];[0,0,0.6];[0,0,0.7];[0,0,0.8];[0,0,1]]
plot((0:20)*10, maddpg_returns(1,:), LineWidth=0.9)
plot((0:20)*10, maddpg_returns(2,:), LineWidth=0.9)
plot((0:20)*10, maddpg_returns(3,:), LineWidth=0.9)
plot((0:20)*10, maddpg_returns(4,:), LineWidth=0.9)
plot((0:20)*10, maddpg_returns(5,:), LineWidth=0.9)
plot((0:20)*10, maddpg_returns(6,:), LineWidth=0.9)
plot((0:20)*10, maddpg_returns(7,:), LineWidth=0.9)
axis([0,200, -0.41,-0.33])
xticks(linspace(0,200,6))
% yticks(linspace(-0.40,-0.30,5))
legend(["Linear function", "Monotonic function with d=2", ...
    "Monotonic function with d=4","Monotonic function with d=8", ...
    "Monotonic function with d=12", "Monotonic function with d=16",...
    "Monotonic function with d=20",...
%     "Monotonic function with d=10", "Monotonic function with d=12" ...
    ],"Location","best",FontSize=18)
xlabel("Episode")
ylabel("Return")


figure(7)
% box on
% grid on
hold on

k_p = maddpg_results.k_p';
k_m = maddpg_results.k_m'
b_p = maddpg_results.b_p'
b_m = maddpg_results.b_m'

set(gcf,'position',[100,100,800,450])
set(gca,'FontSize',24)
x = (-2:0.01:2)';
y=f_monotone(x, k_p, k_m, b_p, b_m);

plot(x, y, Color="r", LineWidth=0.9)
plot(x, maddpg_results.K_linear*x, Color="b", LineWidth=0.9)

plot(b_p(2)+0*linspace(-1,0,100), linspace(0, f_monotone(b_p(2), k_p, k_m, b_p, b_m), 100), "--", Color=[0.5, 0.5, 0.5])
plot(b_p(3)+0*linspace(-1,0,100), linspace(0, f_monotone(b_p(3), k_p, k_m, b_p, b_m), 100), "--", Color=[0.5, 0.5, 0.5])
plot(b_p(4)+0*linspace(-1,0,100), linspace(0, f_monotone(b_p(4), k_p, k_m, b_p, b_m), 100), "--", Color=[0.5, 0.5, 0.5])

plot(b_m(2)+0*linspace(-1,0,100), linspace(0, f_monotone(b_m(2), k_p, k_m, b_p, b_m), 100), "--", Color=[0.5, 0.5, 0.5])
plot(b_m(3)+0*linspace(-1,0,100), linspace(0, f_monotone(b_m(3), k_p, k_m, b_p, b_m), 100), "--", Color=[0.5, 0.5, 0.5])
plot(b_m(4)+0*linspace(-1,0,100), linspace(0, f_monotone(b_m(4), k_p, k_m, b_p, b_m), 100), "--", Color=[0.5, 0.5, 0.5])

plot(linspace(0, b_p(2), 100), f_monotone(b_p(2), k_p, k_m, b_p, b_m)+0*linspace(-1, 0, 100), "--", Color=[0.5, 0.5, 0.5])
plot(linspace(0, b_p(3), 100), f_monotone(b_p(3), k_p, k_m, b_p, b_m)+0*linspace(-1, 0, 100), "--", Color=[0.5, 0.5, 0.5])
plot(linspace(0, b_p(4), 100), f_monotone(b_p(4), k_p, k_m, b_p, b_m)+0*linspace(-1, 0, 100), "--", Color=[0.5, 0.5, 0.5])

plot(linspace(0, b_m(2), 100), f_monotone(b_m(2), k_p, k_m, b_p, b_m)+0*linspace(-1, 0, 100), "--", Color=[0.5, 0.5, 0.5])
plot(linspace(0, b_m(3), 100), f_monotone(b_m(3), k_p, k_m, b_p, b_m)+0*linspace(-1, 0, 100), "--", Color=[0.5, 0.5, 0.5])
plot(linspace(0, b_m(4), 100), f_monotone(b_m(4), k_p, k_m, b_p, b_m)+0*linspace(-1, 0, 100), "--", Color=[0.5, 0.5, 0.5])


axis([-1.2, 1.3,-2.7,3.2])
% Function arrow: This needs a toolbox in matlab called "Arrow"
arrow([-1.2,0],[1.3,0],'width',0.4)
arrow([0, -2.7],[0, 3.2],'width',0.4)
ax = gca;
ax.XAxisLocation = 'origin';
ax.YAxisLocation = 'origin';
ax.TickDir = 'none';

x_turning_point = [b_m(4) b_m(3) b_m(2) b_p(2) b_p(3) b_p(4)]';
y_at_turning_point = f_monotone(x_turning_point, k_p, k_m, b_p, b_m);

x_turning_point = round(x_turning_point*100)/100;
y_at_turning_point = round(y_at_turning_point*100)/100;

x_turning_point_dev = [-0.15 -0.0 0.05 -0.16 -0.16 -0.08]'-0.1;
% xticks(x_turning_point + x_turning_point_dev)
% xticklabels(string(x_turning_point'));
xticklabels([]);

for i=1:length(x_turning_point)
    text(x_turning_point(i)+x_turning_point_dev(i), 0.3, string(x_turning_point(i)), FontSize=24)
end


yticks(y_at_turning_point + [-0.1 0.18 0.18 0 0 0.1]')
yticklabels(string(y_at_turning_point'))


text(1.15, -0.3, "s_1", FontSize=24)
text(0.05, 2.9, "U_1", FontSize=24)

legend("Monotonic function", "Linear function", "Location","southeast")

figure(8)
box on
grid on
hold on
set(gcf,'position',[100,100,800,450])
set(gca,'FontSize',24)

pso_returns = 1:40;
ga_returns = 1:40;
maddpg_returns_d_4 = maddpg_returns(3,:);
pso_ga_version = "v_pso_ga_001"
pso_results = load("./useful/"+pso_ga_version+"_pso.mat");
ga_results = load("./useful/"+pso_ga_version+"_ga.mat")
pso_returns = pso_retults.test_scores;
ga_returns = ga_results.test_scores;

colororder([[1,0,0];[0,1,0],[0,0,1]])

plot((0:20)*10, maddpg_returns_d_4, LineWidth=0.9)
plot((0:40)*10, pso_returns, LineWidth=0.9)
plot((0:40)*10, ga_returns, LineWidth=0.9)

axis([0,400, -0.45,-0.33])
xticks(linspace(0,400,6))
% yticks(linspace(-0.40,-0.30,5))
legend(["DSG-MADDPG", "PSO", "GA"],"Location","best",FontSize=18)
xlabel("Episode")
ylabel("Return")


function y=f_monotone(x, k_p, k_m, b_p, b_m)
% x: n*1
% k_p, k_m, b_p, b_m: d*1
% y: n*1
    g_p = relu(x-b_p')*k_p-relu(x-b_p(2:end)')*k_p(1:end-1);
    g_m = -relu(b_m'-x)*k_m+relu(b_m(2:end)'-x)*k_m(1:end-1);
    y = g_p+g_m;
end

function y=relu(x)
    y = max(0, x);
end

function plot_saturated_function()
s=-4:0.03:4;

u_o = 1;
u_oo = 1.5;
u_l = -1;
u_ll = -1.5;
% 收敛太慢，换成指数
% u=(s<u_l).*(u_ll-(u_l-u_ll)^2./(s+u_ll-2*u_l))...
%    +(s>=u_l).*(s<=u_o).*(s)...
%    +(s>u_o).*(u_oo-(u_oo-u_o)^2./(s+u_oo-2*u_o));

u=(s<u_l).*(u_ll-(u_ll-u_l)*exp(-(s-u_l)/(u_ll-u_l)))...
   +(s>=u_l).*(s<=u_o).*(s)...
   +(s>u_o).*(u_oo-(u_oo-u_o)*exp(-(s-u_o)/(u_oo-u_o)));

set(gcf,'position',[100,100,950,430])
set(gca,'FontSize',32)
set(gca,'TickLabelInterpreter','latex');

hold on
set(gca,'GridLineStyle',':','GridColor','k','GridAlpha',1);
plot(u_ll-0.4:0.03:u_oo+0.4,u_ll-0.4:0.03:u_oo+0.4,'LineWidth',1.3,"LineStyle","--","Color","r")
plot(s,u,'LineWidth',1.3,"LineStyle","-","Color","b")

plot(s,u*0+u_ll, 'LineWidth',0.5,"LineStyle","--","Color","black")
plot(s,u*0+u_oo, 'LineWidth',0.5,"LineStyle","--","Color","black")
plot(u_o+0*(0:0.03:u_o), 0:0.03:u_o, 'LineWidth',0.9,"LineStyle","-","Color",[0.5,0.5,0.5])
plot(0:0.03:u_o, u_o+0*(0:0.03:u_o), 'LineWidth',0.9,"LineStyle","-","Color",[0.5,0.5,0.5])
plot(u_l+0*(0:-0.03:u_l), 0:-0.03:u_l, 'LineWidth',0.9,"LineStyle","-","Color",[0.5,0.5,0.5])
plot(0:-0.03:u_l, u_l+0*(0:-0.03:u_l), 'LineWidth',0.9,"LineStyle","-","Color",[0.5,0.5,0.5])
% xlabel("s_i")
% ylabel("U_i")
axis([-3,3,-2.2,2.2])
% Function arrow: This needs a toolbox in matlab called "Arrow"
arrow([-3,0],[3,0],'width',0.4)
arrow([0, -2.2],[0, 2.2],'width',0.4)
legend("Linear function","Saturated linear function", ...
    "Location","best",Fontsize=20)
ax = gca;
ax.XAxisLocation = 'origin';
ax.YAxisLocation = 'origin';
% ax.XAxis. = 'off';
% ax.YAxis.Visible = 'off';
xticks([u_l-0.15, 0+0.15, u_o])
% xticklabels(["$$\underline{u}_i$$","0","$$\overline{u}_i$$"])
xticklabels([])

yticks([u_ll-0.35, u_l-0.15, u_o, u_oo+0.31])
yticklabels(["$${\underline{u}}^h_i$$","$$\underline{u}_i$$","$$\overline{u}_i$$", "$${\overline{u}}^h_i$$"])

text(u_l-0.3, -0.2, "$\underline{u}_i$",FontSize=32,Interpreter="latex")
text(0.1, -0.3, "$0$",FontSize=32,Interpreter="latex")
text(u_o-0.1, -0.25, "$\overline{u}_i$",FontSize=32,Interpreter="latex")

ax.TickDir = 'none';
text(2.2, -0.3, "$K_is_i$", FontSize=28,Interpreter="latex")
text(0.1, 1.85, "${U}^h_i / U_i$", FontSize=28, Interpreter="latex")
end

