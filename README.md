# demand-layaring
Darknet memory optimization

0-1. git clone https://github.com/AlexeyAB/darknet.git

0-2. git clone "This Repository"

0-3. Copy files and folders that are not in the "demand-layering" folder from "daknet" folder

> /backup, /build, /cfg, /data, /obj folders
> 
> .sh files





1. Download .weigts files (already apply the batch normalization)
>> https://drive.google.com/drive/folders/1gCFwwub2tJyNmBmxV1baVvYqYY6X3rKO?usp=sharing

2. Download .mp4 files
>> https://drive.google.com/file/d/1cVbsXqaTU8BIR3owLB2kgl4cGyvYwSt4/view?usp=sharing

3. Edit Make file (Choose one pipeline architecture, and change 0 to 1)
>> sequential = 0
>> sync = 0
>> async = 0
>> two_stage = 0

>> trade_off =0 (Only async and two_stage)

4. Excute .sh file





If you want to use video file, then V4L2 must be 0

If you want to use webcam, then V4L2 can be 0 or 1

