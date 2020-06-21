--local pvDir = os.getenv("HOME") .. "/OpenPV";
--package.path = package.path .. ";" .. pvDir .. "/parameterWrapper/?.lua"; 
--local pv = require "PVModule";

package.path = package.path .. ";" .. "../../../parameterWrapper/?.lua";
local pv = require "PVModule";

local pvpFileLocation = "Ganglion_pvp_file";
dofile("Retina_CIFAR.lua");
os.execute("mkdir -p " .. pvpFileLocation);
local file = io.open("/home/ek826/Documents/OpenPV/tutorials/Retina/img128.txt");
for i = 1,5 do
   local line = file:read();
   print(line);
   pvParams.Image.inputPath = line;
   local file_name = "temp";
   local params = io.open(file_name .. ".params","w");
   io.output(params);
   pv.printConsole(pvParams);
   io.close(params);
   os.execute("mpirun -np 4 ../../../build/tests/BasicSystemTest/Release/BasicSystemTest -p " .. file_name .. ".params -l ../GanLog.txt -t 2");
   local obj = line:match("([^/]+)$")
   local fname = obj .. ".pvp"
   print(fname);
   os.execute("mv /home/ek826/Documents/OpenPV/tutorials/Retina/output/Retina_CIFAR10/GanglionON.pvp " .. pvpFileLocation .. "/GanON" .. fname );
   os.execute("mv /home/ek826/Documents/OpenPV/tutorials/Retina/output/Retina_CIFAR10/GanglionOFF.pvp " .. pvpFileLocation .. "/GanOFF" .. fname );
end
io.close(file);
   
