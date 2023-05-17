% Script to generate average trajectories with standard deviation (Fig 2.report) for a particular direction
load monkeydata_training.mat;

max_val=1000; % arbitrary value
x=zeros(100,max_val); % stores x-coordinate for each trial
y=zeros(100,max_val); % stores y-coordinate for each trial 
l=[];% array to store the total duration of each trial
dir=3; % arbitrary direction 

for tr=1:100
    X=trial(tr,dir).handPos(1,:);
    l_x=length(X);
    Y=trial(tr,dir).handPos(2,:);
    x(tr,1:l_x)=X;
    y(tr,1:l_x)=Y;
    l=[l,l_x];
end

l=min(l); % average trajectories on the minimum common duration for all trials
x=x(:,1:l); % reduce x coordinates from t=0 to t=l 
y=y(:,1:l); % reduce y coordinates from t=0 to t=l 

x_mean=mean(x);
y_mean=mean(y);

var_x=std(x);
var_y=std(y);

y1 = y_mean; 
x1=1:numel(y1);

curve1X = y1 + var_y;
curve2X = y1 - var_y;

x2 = [x1, fliplr(x1)];
inBetweenY = [curve1X, fliplr(curve2X)];


y3 = x_mean; 
x3=1:numel(y3);

curve1X = y3 + var_x;
curve2X = y3 - var_x;

x4 = [x3, fliplr(x3)];
inBetweenX = [curve1X, fliplr(curve2X)];



plot(x1, y1, 'b', 'LineWidth', 0.5);
set(gca,'ylim',[-150,150])
hold on;
plot(x3, y3, 'r', 'LineWidth', 0.5);
hold on;
line([300, 300], ylim, 'Color', 'k', 'LineWidth', 0.5);
hold on;
fill(x2, inBetweenY, 'b','FaceAlpha',0.05);
hold on;
fill(x4, inBetweenX, 'r','FaceAlpha',0.05);
set(gca,'ylim',[-150,150])% start motion 
lg2=legend('Mean y trajectory','Mean x trajectory','Reaction time', 'Location','east');
set(lg2, 'Interpreter','latex')
grid('minor')
set(gca,'TickLabelInterpreter','latex')
xlabel('Times [ms]','Interpreter','latex','FontSize',13)
ylabel('Position [cm]','Interpreter','latex','FontSize',13)


