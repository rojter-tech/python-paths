using PyPlot
fig1 = gcf(); close(fig1); 
plot(rand(5,1)); fig1 = gcf();
display(fig1);

fig2 = gcf(); close(fig2); 
plot(rand(5,2)); fig2 = gcf();
display(fig2);

fig3 = gcf(); close(fig3); 
x = range(0,stop=2*pi,length=1000); 
y = sin.(3*x + 4*cos.(2*x));
plot(x, y, color="red", linewidth=2.0, linestyle="--"); fig3 = gcf();
display(fig3);

fig4 = gcf(); close(fig3); 
hist([1,1,1,2,2,3,3,3,3,3,5,5]); fig4 = gcf();
display(fig4);
