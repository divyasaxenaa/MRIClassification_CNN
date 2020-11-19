import nibabel as nb
import numpy as np
import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt

def shw_img(Input_file,label):
    # Load data
    img=nb.load(Input_file)
    # 3D data
    if img.header['dim'][0]==3:
        data=img.get_data()
        # 4D data
    elif  img.header['dim'][0]==4:
        data=img.get_data()[:,:,:,0]
    # Header
    header=img.header
    # Set NAN to 0
    data[np.isnan(data)] = 0
    ### ORIENTATION ###
    orient_qform = img.get_qform()[0, 0]
    orient_sform = img.get_sform()[0, 0]
    if orient_qform < 0 and (orient_sform == 0 or orient_sform < 0):
        orientation_Left = 'R'
    elif orient_qform > 0 and (orient_sform == 0 or orient_sform > 0):
        orientation_Left = 'L'
    if orient_sform < 0 and (orient_qform == 0 or orient_qform < 0):
        orientation_Left = 'R'
    elif orient_sform > 0 and (orient_qform == 0 or orient_qform > 0):
        orientation_Left = 'L'

    # Size per slice
    SizeSlice_Xaxis = data.shape[0]
    SizeSlice_Yaxis = data.shape[1]
    SizeSlice_Zaxis = data.shape[2]
    # Middle slice number
    middleSlice_Xaxis = int(SizeSlice_Xaxis/2)
    middleSlice_Yaxis = int(SizeSlice_Yaxis/2)
    middleSlice_Zaxis = int(SizeSlice_Zaxis/2)
    # True middle point
    tmiddleSlice_Xaxis = SizeSlice_Xaxis/2.0
    tmiddleSlice_Yaxis = SizeSlice_Yaxis/2.0
    tmiddleSlice_Zaxis = SizeSlice_Zaxis/2.0
    # Spacing for Aspect Ratio
    spacing_Xaxis = header['pixdim'][1]
    spacing_Yaxis = header['pixdim'][2]
    spacing_Zaxis = header['pixdim'][3]
    # Plot main window
    fig = plt.figure(
        facecolor='black',
        figsize=(5,4),
        dpi=200
    )
    # Black background
    plt.style.use('dark_background')
    # Set title
    fig.canvas.set_window_title(label)

    # Coronal
    axis1=fig.add_subplot(2,2,1)
    imgplot = plt.imshow(
        np.rot90(data[:,middleSlice_Yaxis,:]),
        aspect=spacing_Zaxis/spacing_Xaxis,
    )
    imgplot.set_cmap('gray')
    axis1.hlines(tmiddleSlice_Zaxis, 0, SizeSlice_Xaxis, colors='red', linestyles='dotted', linewidth=.5)
    axis1.vlines(tmiddleSlice_Xaxis, 0, SizeSlice_Zaxis, colors='red', linestyles='dotted', linewidth=.5)
    plt.axis('off')
    # Sagittal
    axis2=fig.add_subplot(2,2,2)
    imgplot = plt.imshow(
        np.rot90(data[middleSlice_Xaxis,:,:]),
        aspect=spacing_Zaxis/spacing_Yaxis,
    )
    imgplot.set_cmap('gray')
    axis2.hlines(tmiddleSlice_Zaxis, 0, SizeSlice_Yaxis, colors='red', linestyles='dotted', linewidth=.5)
    axis2.vlines(tmiddleSlice_Yaxis, 0, SizeSlice_Zaxis, colors='red', linestyles='dotted', linewidth=.5)
    plt.axis('off')
    # Axial
    axis3=fig.add_subplot(2,2,3)
    imgplot = plt.imshow(
        np.rot90(data[:,:,middleSlice_Zaxis]),
        aspect=spacing_Yaxis/spacing_Xaxis
    )
    imgplot.set_cmap('gray')
    axis3.hlines(tmiddleSlice_Yaxis, 0, SizeSlice_Xaxis, colors='red', linestyles='dotted', linewidth=.5)
    axis3.vlines(tmiddleSlice_Xaxis, 0, SizeSlice_Yaxis, colors='red', linestyles='dotted', linewidth=.5)
    plt.axis('off')
    plt.text(-10, middleSlice_Yaxis+5, orientation_Left, fontsize=9, color='red') # Label on left side
    sform=np.round(img.get_sform(),decimals=2)
    sform_txt=str(sform).replace('[',' ').replace(']',' ').replace(' ','   ').replace('   -','  -')
    # qform code
    qform=np.round(img.get_qform(),decimals=2)
    qform_txt=str(qform).replace('[',' ').replace(']',' ').replace(' ','   ').replace('   -','  -')
    # Dimensions
    dims=str(data.shape).replace(', ',' x ').replace('(','').replace(')','')
    dim=("Dimensions: "+dims)
    # Spacing
    spacing=("Spacing: "
             +str(np.round(spacing_Xaxis, decimals=2))
             +" x "
             +str(np.round(spacing_Yaxis, decimals=2))
             +" x "
             +str(np.round(spacing_Zaxis, decimals=2))
             +" mm"
    )
    # Data type
    type=img.header.get_data_dtype()
    type_str=("Data type: "+str(type))
    # Volumes
    volumes=("Volumes: "+str(img.header['dim'][4]))
    # Range
    min=np.round(np.amin(data), decimals=2)
    max=np.round(np.amax(data), decimals=2)
    range=("Range: "+str(min)+" - "+str(max))
    text=(
        dim+"\n"
        +spacing+"\n"
        +volumes+"\n"
        +type_str+"\n"
        +range+"\n\n"
        +"sform code:\n"
        +sform_txt+"\n"
        +"\nqform code:\n"
        +qform_txt
    )

    # Plot text subplot
    ax4=fig.add_subplot(2,2,4)
    plt.text(
        0.15,
        0.95,
        text,
        horizontalalignment='left',
        verticalalignment='top',
        size=6,
        color='white',
    )
    plt.axis('off')


if __name__ == "__main__":
    plt.rcParams['toolbar'] = 'None'
    home = str(Path.home())
    Input_file_patient = "/home/divya/Downloads/Cnn_Mri_Classification/6389_project1/6389_project1/Testing/backup/health/sub1/T1_bet1.nii"
    Input_file_health = "/home/divya/Downloads/Cnn_Mri_Classification/6389_project1/6389_project1/Testing/backup/patient/sub1/T1_bet1.nii"
    np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})
    shw_img(Input_file_patient,"Patient Sample")
    shw_img(Input_file_health,"Healthy Sample")

# Adjust whitespace
plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)

# Display
plt.show()