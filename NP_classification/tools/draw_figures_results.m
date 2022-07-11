function  draw_figures_results(dataset_name,test_problem,results,nr_test,obj_bench)



nr_algors           = length(results);
temp_results        = cell(nr_algors,1);
for i = 1:nr_algors
   temp_results{i}  = results{i}; 
end


res_name            = dataset_name;
res_prob            = test_problem;


y_min_tobj   = 0.5*obj_bench;     y_max_tobj   = 1;
y_min_eobj   = 0.5*obj_bench;     y_max_eobj   = 1;

y_min_t  = 0;       y_max_tcons  = 1;
y_min_e  = 0;       y_max_econs  = 1;


x_max_e = 0;   x_max_t = 0;
for i = 1:nr_algors
    x_max_e = max(x_max_e,max(temp_results{i}.epoch));
    x_max_t = max(x_max_t,max(temp_results{i}.time)); 
    y_min_e = min(y_min_e,min(temp_results{i}.cons));
    y_min_t = min(y_min_t,min(temp_results{i}.cons));     
end


x_max_eobj   = 1.1*x_max_e;   x_max_econs  = x_max_eobj;
x_max_tobj   = 1.1*x_max_t;   x_max_tcons  = x_max_tobj;

y_min_econs   = 0.5*y_min_e;   
y_min_tcons   = 0.5*y_min_t;   
        
intervals     = 200*ones(1,nr_algors);




names = cell(nr_algors,1);

for i = 1:nr_algors
    names{i}    = temp_results{i}.name;
end


markers_t     = {'.-','d-','o-','s-','^-','>-','<-','v-'};
colors_t = {       [119,55,0]/255,...  
          [0,0,0]/255,...      
          [0,101,189]/255,...  
          [17,140,17]/255,... 
          [255,71,71]/255,...     
          [67,20,97]/255, ...               
          [0.9,0.7,0.0], ...        
          [218,215,203]/255};        
markersize_t = {6,6,6,6,6};   
linewidth_t = {0.5,1.5,1.5,1.5,1.5};
colors   = cell(1,nr_algors); markers   = cell(nr_algors,1); markersizes   = cell(nr_algors,1); linewidths   = cell(nr_algors,1);
if nr_algors > 8; error('The number of algorithms is too many!'); end
for i = 1:nr_algors
   colors{i} = colors_t{i}; markers{i} = markers_t{i}; markersizes{i} = markersize_t{i}; linewidths{i} = linewidth_t{i};
end

%%{
%% print seperate pics


    
%--------------------------------------------------------------------------
% plot: epoch // obj
%--------------------------------------------------------------------------
    
figure;
clf
for i = 1:nr_algors

    semilogy_marker(temp_results{i}.epoch,temp_results{i}.obj,markers{i},intervals(i),markersizes{i},colors{i},linewidths{i});
end


ylim([y_min_eobj y_max_eobj]);
xlim([0 x_max_eobj]);

xlabel('epochs');
ylabel('objective');
set(gca,'FontSize',20);
grid on;
set(gca,'YMinorGrid','off','YMinorTick','off');

h = refline([0,obj_bench]); 
%h.DisplayName = 'appr optim'; 
h.Color = [0.4940 0.1840 0.5560]; h.LineWidth = 1.0; h.LineStyle = '--';

legend(names);

print('-depsc',strcat('./results/results_',res_prob,'_',res_name,'_epoch_obj_',num2str(nr_test),'_runs','.eps'));
%--------------------------------------------------------------------------
% plot: epoch // cons
%--------------------------------------------------------------------------
figure;
clf
for i = 1:nr_algors
    semilogy_marker(temp_results{i}.epoch,temp_results{i}.cons,markers{i},intervals(i),markersizes{i},colors{i},linewidths{i});
end


ylim([y_min_econs y_max_econs]);
xlim([0 x_max_econs]);


xlabel('epochs');
ylabel('constraint violation');
set(gca,'FontSize',20);
grid on;
set(gca,'YMinorGrid','off','YMinorTick','off','YScale','linear');

 h = refline([0,0]); 
%h.DisplayName = 'zero'; 
h.Color = [0.4940 0.1840 0.5560]; h.LineWidth = 1.0; h.LineStyle = '--';

legend(names);

print('-depsc',strcat('./results/results_',res_prob,'_',res_name,'_epoch_cons_',num2str(nr_test),'_runs','.eps'));


%--------------------------------------------------------------------------
% plot: time // obj
%--------------------------------------------------------------------------
figure;
clf
for i = 1:nr_algors

    semilogy_marker(temp_results{i}.time,temp_results{i}.obj,markers{i},intervals(i),markersizes{i},colors{i},linewidths{i});
end


ylim([y_min_tobj y_max_tobj]);
xlim([0 x_max_tobj])

xlabel('time elapsed (sec)');
ylabel('objective');
%ylabel('$\|F^I(x)\|$','Interpreter','latex');
set(gca,'FontSize',20);
grid on;
set(gca,'YMinorGrid','off','YMinorTick','off');

h = refline([0,obj_bench]); 
%h.DisplayName = 'appr optim'; 
h.Color = [0.4940 0.1840 0.5560]; h.LineWidth = 1.0; h.LineStyle = '--';

legend(names);

print('-depsc',strcat('./results/results_',res_prob,'_',res_name,'_time_obj_',num2str(nr_test),'_runs','.eps'));

%--------------------------------------------------------------------------
% plot: time // cons
%--------------------------------------------------------------------------
figure;
clf    

for i = 1:nr_algors

    semilogy_marker(temp_results{i}.time,temp_results{i}.cons,markers{i},intervals(i),markersizes{i},colors{i},linewidths{i});
end


ylim([y_min_tcons y_max_tcons]);
xlim([0 x_max_tcons])

xlabel('time elapsed (sec)');
ylabel('constraint violation');
%ylabel('$\|F^I(x)\|$','Interpreter','latex');
set(gca,'FontSize',20);
grid on;
set(gca,'YMinorGrid','off','YMinorTick','off','YScale','linear');

h = refline([0,0]); 
%h.DisplayName = 'zero'; 
h.Color = [0.4940 0.1840 0.5560]; h.LineWidth = 1.0; h.LineStyle = '--';

legend(names);

print('-depsc',strcat('./results/results_',res_prob,'_',res_name,'_time_cons_',num2str(nr_test),'_runs','.eps'));


%}






