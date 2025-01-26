# Real Code
from openpyxl import Workbook
import os
from mpl_point_clicker import clicker
import matplotlib.pyplot as plt
import cv2
import matplotlib as mpl
import pandas as pd
mpl.use('QT5Agg')                                   
import numpy as np
from mpl_point_clicker import clicker 
wb = Workbook()
ws = wb.active
print("Video Name")
Video_Name = str(input())
print("Video Type , Check Video Type = MOV , MP4")
Video_type = str(input())
print("First Particle")
Particle_1 = str(input())
print("Second Particle")
Particle_2 = str(input())
print("Output Folder")
Image_Folder = str(input())
print("Total Liquid Level (cm)")
Liquid_Level = int(input())
print("Baffle Type , Supported Entry = Front , Mid , Back ")
Baffle = str(input())
print("Output Data Excel File")
excel_file = str(input())
video_path = "D:\\Video\\"+Video_Name+"."+Video_type
output_folder = "D:\\Images\\"+Particle_1+" + "+Particle_2+"\\"+Image_Folder+" "+Baffle+" Baffle\\"
output_upper_array = "D:\\Arrays\\"+Particle_1+" + "+Particle_2+"\\"+Image_Folder+" "+Baffle+" Baffle\\Upper Cloud"
output_lower_array = "D:\\Arrays\\"+Particle_1+" + "+Particle_2+"\\"+Image_Folder+" "+Baffle+" Baffle\\Lower Cloud"
image_path = os.makedirs(output_folder)
upper_folder = os.makedirs(output_upper_array)
lower_folder = os.makedirs(output_lower_array)
pixel_range = [0,0,1500,1700] 
cap = cv2.VideoCapture(video_path)
frame_count = 0
while True:
    success,frame = cap.read()
    x = pixel_range[0]
    y = pixel_range[1]
    w = pixel_range[2]
    h = pixel_range[3]
    if success:
        cropped_frame = frame[y:y+h, x:x + w]
        image_path = os.path.join(output_folder, f"Image_{frame_count:04d}.jpg")
        cv2.imwrite(image_path, cropped_frame)
        frame_count = frame_count + 1
    else:
        break
cap.release()
photo = np.arange(0,frame_count)
Photo = []
for i in photo:
    name1 = str(i)
    if i < 10:
        nf = "000"+name1
    elif i>=10 and i<100:
        nf = "00"+name1
    elif i>=100 and i<1000:
        nf = "0"+name1
    elif i>=1000:
        nf = name1
    Photo.append(nf)
print("Sample Photo , Available "+str(len(Photo)) + " Photo (Except "+str(len(Photo))+" th Image)")
sample_photo = int(input())
example = cv2.imread("D:\\Images\\"+Particle_1+" + "+Particle_2+"\\"+Image_Folder+" "+Baffle+" Baffle\\Image_"+str(Photo[sample_photo])+".jpg")
T2 = cv2.cvtColor(example,cv2.COLOR_BGR2RGB)
ex_image = cv2.flip(T2,0)
fig, ax = plt.subplots()
ax.imshow(ex_image, origin="lower")
ax.grid()
klicker = clicker(ax, ["Total Liquid Level","Vertical Scan Limit","Horizontal Scan Limit"], markers=["x","o","X"])
plt.show()
positions = klicker.get_positions()
Max_Level = positions["Total Liquid Level"]
Horizontal_Border = positions["Horizontal Scan Limit"]
Vertical_Border = positions["Vertical Scan Limit"]
Correction_Factor = Liquid_Level/(Max_Level[0][-1] - Max_Level[-1][-1])
xrange = np.arange(int(Horizontal_Border[0][0]),int(Horizontal_Border[-1][0]),1)
yrange = np.arange(int(Vertical_Border[-1][-1]),int(Vertical_Border[0][-1]),1)
Top_Clouds = 0
Bottom_Clouds = 0
for mi in Photo:
    t2 = cv2.imread("D:\\Images\\"+Particle_1+" + "+Particle_2+"\\"+Image_Folder+" "+Baffle+" Baffle\\Image_"+mi+".jpg")
    Upper_Path = "D:\\Arrays\\"+Particle_1+" + "+Particle_2+"\\"+Image_Folder+" "+Baffle+" Baffle\\Upper Cloud\\Image_"+mi+".jpg"
    Lower_Path ="D:\\Arrays\\"+Particle_1+" + "+Particle_2+"\\"+Image_Folder+" "+Baffle+" Baffle\\Lower Cloud\\Image_"+mi+".jpg"
    T2 = cv2.cvtColor(t2,cv2.COLOR_BGR2RGB)
    et2 = cv2.flip(T2,0)
    T4 = cv2.cvtColor(et2,cv2.COLOR_RGB2LAB)
    Upper = []
    Lower = []
    Border1 = []
    Border2 = []
    for A in xrange:
        d=[]
        for iz in yrange:
            pil = T4[iz][A]
            d.append(pil)
        x1 =[]
        p1 = []
        for p in d :
            x1.append(p[0])
        X1 = np.array(x1)
        border1 = np.where(X1 == np.max(X1))[0][0]
        slope1 = np.polyfit(yrange,X1,1)
        line = slope1[0]*yrange + slope1[1]
        diff = X1 - line
        sign = np.sign(diff)
        zero = np.where(np.diff(sign) != 0)[0] + 1
        border2 = 0
        diff_liquid = diff[0:border1+1]
        yrange_liquid = yrange[0:border1+1]
        sign_liquid = np.sign(diff_liquid)
        zero_liquid = np.where(np.diff(sign_liquid) != 0)[0] + 1
        zero_liquid = np.append(0,zero_liquid)
        zero_liquid = np.sort(np.append(-1,zero_liquid))
        center = np.where(diff_liquid == np.min(diff_liquid))[0][0]
        for i in range(0,len(zero_liquid)-1):
            interval = np.arange(yrange_liquid[zero_liquid[i]],yrange_liquid[zero_liquid[i+1]],1)
            if yrange_liquid[center] in interval:
                p1.append(yrange_liquid[zero_liquid[i]])
                p1.append(yrange_liquid[zero_liquid[i+1]])
        upper_part = diff_liquid[center:]
        h_extend = np.where(upper_part > -diff_liquid[center])[0]
        if len(h_extend) != 0:
            p1.append(yrange_liquid[center:][h_extend[-1]])
        if len(p1) == 0:
            p1.append(yrange_liquid[center])
            p1.append(yrange_liquid[center])
        Border1.append(yrange[border1])
        Border2.append(yrange[border2])
        Upper.append(max(p1))
        Lower.append(min(p1))
    Upper_cloud = np.array(Upper)*Correction_Factor
    Lower_cloud = np.array(Lower)*Correction_Factor
    np.savez(Upper_Path,maxs = Upper_cloud)
    np.savez(Lower_Path,mins = Lower_cloud)
    Top_Clouds = (Upper_cloud/len(Photo)) + Top_Clouds
    Bottom_Clouds = (Lower_cloud/len(Photo)) + Bottom_Clouds
Final_Top = Top_Clouds
Final_Bottom = Bottom_Clouds
if Baffle == "Front":
    xrange_f = xrange*Correction_Factor
    xrange_final = xrange_f - xrange_f[0]
if Baffle == "Mid":
    slice = int(len(xrange)/2) - 1
    mid_point = xrange[slice]
    xrange_final = Correction_Factor*(xrange - mid_point)
if Baffle == "Back":
    xrange_fi = xrange*Correction_Factor
    xrange_final = (xrange_fi[-1]-xrange_fi)
x_axis_path = "D:\\Arrays\\"+Particle_1+" + "+Particle_2+"\\"+Image_Folder+" "+Baffle+" Baffle\\X Axis Data"
cf_path = "D:\\Arrays\\"+Particle_1+" + "+Particle_2+"\\"+Image_Folder+" "+Baffle+" Baffle\\Correction Factor"
cf_save = np.array([Correction_Factor])
np.savez(x_axis_path, x_axis = xrange_final)
np.savez(cf_path, c_factor = Correction_Factor)
data_e= [["Distance along baffle (cm)","Upper Cloud (cm)","Lower Cloud (cm)"]]
for row in data_e:
    ws.append(row)
for n in range(0,len(xrange_final)):
    xm = [list([xrange_final[n],Final_Top[n],Final_Bottom[n]])]
    for v in xm:
        ws.append(v)
final_excel_path = "C:\\Users\\User\\Desktop\\"+excel_file+".xlsx"
wb.save(final_excel_path)
plt.plot(xrange_final,Final_Top,color="red",label="Upper Cloud , Max = "+str(round(max(Final_Top),2))+" cm")
plt.plot(xrange_final,Final_Bottom,color="blue",label="Lower Cloud , Min = "+str(round(min(Final_Bottom),2))+" cm")
plt.grid()
plt.xlabel("Distance From Baffle (cm)")
plt.ylabel("Cloud Height (cm)")
plt.legend()
plt.xlim(xrange_final[0],xrange_final[-1])
plt.ylim(0,24)
plt.savefig("D:\\Final Plots\\"+Particle_1+" + "+Particle_2+" Mixing "+ Baffle +" Baffle")
    
