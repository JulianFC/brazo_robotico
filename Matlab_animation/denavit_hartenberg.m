% Denavit Hartenberg Plot.
function P = denavit_hartenberg( theta1, theta2, theta3, theta4 )

L1 = 1; L2 = 1; L3 = 1; L4 = 1; N = 4;

QT = [ 0, 0, 0, 0, 0; 0, 0, 0, 0, 0; 0, 0, 0, 0, 0; 1, 1, 1, 1, 1];

R1 = [cosd(theta1), 0, sind(theta1), 0; sind(theta1), 0, -cosd(theta1), 0; 0, 1, 0, L1; 0, 0, 0, 1];
R2 = [cosd(theta2), -sind(theta2), 0, L2*cosd(theta2); sind(theta2), cosd(theta2), 0, L2*sind(theta2); 0, 0, 1, 0; 0, 0, 0, 1];
R3 = [cosd(theta3), -sind(theta3), 0, L3*cosd(theta3); sind(theta3), cosd(theta3), 0, L3*sind(theta3); 0, 0, 1, 0; 0, 0, 0, 1];
R4 = [cosd(theta4), -sind(theta4), 0, L4*cosd(theta4); sind(theta4), cosd(theta4), 0, L4*sind(theta4); 0, 0, 1, 0; 0, 0, 0, 1];

Q1 = R1*QT(:,2);
Q2 = R1*R2*QT(:,3);
Q3 = R1*R2*R3*QT(:,4);
Q4 = R1*R2*R3*R4*QT(:,5);

QT(:,2) = Q1;
QT(:,3) = Q2;
QT(:,4) = Q3;
QT(:,5) = Q4;

P = [QT(1,5), QT(2,5), QT(3,5)];
hold on
for i = 1:N
    plot3(QT(1,i:i+1),QT(2,i:i+1),QT(3,i:i+1),'LineWidth',5);
end

grid on; xlim([-2 4]); ylim([-2 4]); zlim([0 4]); view(40,4); hold off;

end

