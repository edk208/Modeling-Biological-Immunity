addpath('../../../mlab/util/');
im = imread('./ILSVRC2012_val_00000001.JPEG');
[data, HDR] = readpvpfile('./GanOnILSVRC2012_val_00000001.JPEG.pvp');
[data2, HDR] = readpvpfile('./GanOFFILSVRC2012_val_00000001.JPEG.pvp');
SumC = zeros(128,128);
for k= 1:128
    A = zeros(16384,1);
    Aoff = zeros(16384,1);
    if ~isempty(data{k}.values)
        A(data{k}.values(:,1)+1) = data{k}.values(:,2);
    end
    if ~isempty(data2{k}.values)
        Aoff(data2{k}.values(:,1)+1) = data2{k}.values(:,2);
    end
    B = reshape(A,[128 128]);
    Boff = reshape(Aoff,[128 128]);
    SumC = SumC + B'+ Boff'; 
end
sumofall = SumC;
sumofall(:,:,2) = SumC;
sumofall(:,:,3) = SumC;

newim = sumofall;%.*double(im);
maxval = max(max(max(newim)));
minval = min(min(min(newim)));
%imshow((newim+ abs(minval))/(maxval-minval),[])
imshow(im2uint8((sumofall.*im2double(im)/maxval)),[]);
