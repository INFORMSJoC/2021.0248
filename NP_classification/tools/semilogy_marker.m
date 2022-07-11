function semilogy_marker(x, y, marker, interval, markersize,color,linewidth)

if strcmp(marker,'--')
    semilogy(x, y, marker, 'color', color, 'linewidth',linewidth);
else
semilogy(x, y, 'HandleVisibility', 'off','color',color,'linewidth',linewidth);	%plots the main curve
hold on
semilogy(x(1:interval:end), y(1:interval:end), marker(1),...
    'HandleVisibility', 'off','markerfacecolor','auto','markeredgecolor',color,...
    'markersize',markersize,'linewidth',linewidth);	%plots the markers
semilogy(x(1), y(1), marker,'markerfacecolor','auto',...
    'markersize',markersize,'markeredgecolor',color,'color',color,'linewidth',linewidth);	%plots a dummy point for legend
% hold off
end
end