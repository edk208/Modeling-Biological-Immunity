addpath('../../../mlab/util/');
im = imread('./ILSVRC2012_val_00000001.JPEG');
[data, HDR] = readpvpfile('./GanOnILSVRC2012_val_00000001.JPEG.pvp');
[data2, HDR] = readpvpfile('./GanOFFILSVRC2012_val_00000001.JPEG.pvp');
for k= 2:128
    %A = zeros(16384,1);
    %if ~isempty(data{k}.values)
    %    A(data{k}.values(:,1)+1) = data{k}.values(:,2);
    %end
    %B = reshape(A,[128 128]);
    imshow(im, 'InitialMagnification', 300);
    hold on;
    %for i = 1:128
    % for j= 1:128
    %     if B(i,j) == 1
    %           plot(i,j,'r.', 'MarkerSize',10);
    %     end
    % end
    %end
    
    if ~isempty(data{k}.values)
      val = data{k}.values(:,1)+1;
      xs = mod(val-1,128) + 1;
      ys = floor((val-1)/128);
      plot(xs,ys,'r.', 'MarkerSize',5);
    end
    if ~isempty(data2{k}.values)
      val = data2{k}.values(:,1)+1;
      xs = mod(val-1,128) + 1;
      ys = floor((val-1)/128);
      plot(xs,ys,'b.', 'MarkerSize',5);
    end
    
    pause(0.01);
    hold off;
    set(gca,'units','pixels') % set the axes units to pixels
    x = get(gca,'position') % get the position of the axes
    set(gcf,'units','pixels') % set the figure units to pixels
    y = get(gcf,'position') % get the figure position
    set(gcf,'position',[y(1) y(2) x(3) x(4)])% set the position of the figure to the length and width of the axes
    set(gca,'units','normalized','position',[0 0 1 1]) % set the axes units to pixels
    saveas(gcf,sprintf('movies/0001_%04d.png',k));
end

