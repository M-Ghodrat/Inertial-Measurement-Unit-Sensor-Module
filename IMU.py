import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
import pandas as pd
import csv
import control
import matplotlib.pyplot as plt
import io
from io import StringIO
import math

# uploaded_file = None

def ToEulerAngles(Quaternion):
    
    q = Quaternion
    
    # roll (x-axis rotation)
    sinr_cosp = 2 * (q[0]*q[1]+q[2]*q[3])
    cosr_cosp = 1 - 2 * (q[1]**2+q[2]**2)
    phi = math.atan2(sinr_cosp,cosr_cosp)
    
    # pitch (y-axis rotation)
    sinp = 2 * (q[0]*q[2]-q[3]*q[1])
    theta = np.arcsin(sinp)
    
    # yaw (z-axis rotation)
    siny_cosp = 2 * (q[0]*q[3]+q[1]*q[2])
    cosy_cosp = 1 - 2 * (q[2]**2+q[3]**2)
    psi = math.atan2(siny_cosp,cosy_cosp)
    
    return [phi, theta, psi]

def IMU_read(file, *nrows):
    
    text_io = io.TextIOWrapper(file)
    contents = text_io.read()
    text_io.seek(0)
    delimiter = text_io.readline().strip('\n')
    contentsList = contents.split(delimiter)
    text_io.seek(0)
    
    if not nrows:
        IMU = pd.read_csv(StringIO(contentsList[3])) 
    else:
        IMU = pd.read_csv(StringIO(contentsList[3]), nrows=nrows)
        
    info = pd.read_csv(StringIO(contentsList[2]), sep=':')
    dt = 1 / float(info.loc['Average sample rate', 'Additional Info'].strip('Hz'))
    
    # Delete last empty column
    if ' ' in IMU.columns:
        del IMU[' ']

    # remove extra space
    IMU.columns = IMU.columns.str.strip()
    
    #
    Quaternions = ['ANGULAR_POSITION_w', 'ANGULAR_POSITION_x', 'ANGULAR_POSITION_y', 'ANGULAR_POSITION_z']
    Quaternions_Cal = ['ANGULAR_POSITION_w_cal', 'ANGULAR_POSITION_x_cal', 'ANGULAR_POSITION_y_cal', 'ANGULAR_POSITION_z_cal']    
    EulerAngles = ['EulerAngle_x', 'EulerAngle_y', 'EulerAngle_z']
    
    # calibrate quarternions
    cal_period = 1
    initial_q = list(np.mean(IMU[Quaternions][:int(cal_period/dt)]))
    negative_initial_q = [initial_q[0],-initial_q[1],-initial_q[2],-initial_q[3]]
    
    # Quaternions Multiplication
    def Multi(p, q=negative_initial_q):
        output = [p[0]*q[0]-p[1]*q[1]-p[2]*q[2]-p[3]*q[3], 
                  p[1]*q[0]+p[0]*q[1]-p[3]*q[2]+p[2]*q[3],
                  p[2]*q[0]+p[3]*q[1]+p[0]*q[2]-p[1]*q[3],
                  p[3]*q[0]-p[2]*q[1]+p[1]*q[2]+p[0]*q[3]]
        return output

    for n, col in enumerate(Quaternions_Cal):
        IMU[col] = IMU[Quaternions].apply(Multi, axis=1).apply(lambda item: item[n])
    
    # adding Euler angles
    new_col_list = EulerAngles
    for n, col in enumerate(new_col_list):
        IMU[col] = IMU[Quaternions_Cal].apply(ToEulerAngles, axis=1).apply(lambda item: item[n])  
        
    return IMU, info, dt

def IMU_write(file, sep=',', index=False, accelerations=['LIN_ACC_NO_GRAVITY_x', 'LIN_ACC_NO_GRAVITY_y', 'LIN_ACC_NO_GRAVITY_z'],
             angularpositions=['EulerAngle_x', 'EulerAngle_y', 'EulerAngle_z']):
    
    Acceleration = IMU[accelerations]
    AngularPosition = IMU[angularpositions]
    
    return Acceleration, AngularPosition

pi = 3.14

def IMU_filter(Acceleration, AngularPosition, w_c=2*pi):
    
    g = -9.81
    BF = [None] * 3
    
    num_BF = [np.array([(w_c)**3]), 
               np.array([(w_c)**3]), 
               np.array([(w_c)**3])]

    den_BF = [np.array([1, 2*w_c, 2*(w_c)**2, (w_c)**3]), 
               np.array([1, 2*w_c, 2*(w_c)**2, (w_c)**3]), 
               np.array([1, 2*w_c, 2*(w_c)**2, (w_c)**3])]

    BF = [control.tf(num_BF[i], den_BF[i]) for i in range(3)]

    Acceleration_orig = [Acceleration.iloc[:,0].to_numpy(), Acceleration.iloc[:,1].to_numpy(), (Acceleration.iloc[:,2]-g).to_numpy()]
    AngularPosition_orig = [AngularPosition.iloc[:,0].to_numpy(), AngularPosition.iloc[:,1].to_numpy(), AngularPosition.iloc[:,2].to_numpy()]

    T = np.array(range(int(Acceleration.shape[0]))) * dt
    X0 = [0] * 3
    
    _, Acceleration_filtered = zip(*[control.forced_response(BF[i], T=T, U=Acceleration_orig[i], X0=X0[i]) for i in range(3)])
    _, AngularPosition_filtered = zip(*[control.forced_response(BF[i], T=T, U=AngularPosition_orig[i], X0=X0[i]) for i in range(3)])
    Acceleration_filtered = list(Acceleration_filtered)
    Acceleration_filtered[2] = Acceleration_filtered[2] + g
    
    return pd.DataFrame(np.array(Acceleration_filtered).T, columns=Acceleration.columns), pd.DataFrame(np.array(AngularPosition_filtered).T, columns=AngularPosition.columns)

def Calibration(Acceleration, AngularPosition, cal_period):
    
    g = -9.81
    
    Acceleration_calibrated = Acceleration[:]
    Acceleration_calibrated = Acceleration - np.mean(Acceleration[:int(cal_period/dt)])
    Acceleration_calibrated['LIN_ACC_NO_GRAVITY_z'] = Acceleration_calibrated['LIN_ACC_NO_GRAVITY_z'] + g
    
    AngularPosition_calibrated = AngularPosition[:]
    AngularPosition_calibrated = AngularPosition - np.mean(AngularPosition[:int(cal_period/dt)])
    
    return Acceleration_calibrated, AngularPosition_calibrated

def IMU_plot(Acceleration, AngularPosition,
             accelerations=['LIN_ACC_NO_GRAVITY_x', 'LIN_ACC_NO_GRAVITY_y', 'LIN_ACC_NO_GRAVITY_z'],
             angularpositions=['EulerAngle_x', 'EulerAngle_y', 'EulerAngle_z']):
    
    T = np.array(range(Acceleration.shape[0])) * dt
    
    fig, ax = plt.subplots(2, 3, figsize=(15,8))
    ax[0, 0].plot(T, Acceleration[accelerations[1]])
    ax[0, 0].set_title('Surge Acceleration')
#     ax[0, 0].set_ylim([Acceleration.min()[0]-0.1*np.abs(Acceleration.min()[0]),Acceleration.max()[0]+0.1*np.abs(Acceleration.max()[0])])
    ax[0, 1].plot(T, Acceleration[accelerations[0]])
    ax[0, 1].set_title('Sway Acceleration')
#     ax[0, 1].set_ylim([Acceleration.min()[1]-0.1*np.abs(Acceleration.min()[1]),Acceleration.max()[1]+0.1*np.abs(Acceleration.max()[1])])
    ax[0, 2].plot(T, Acceleration[accelerations[2]])
    ax[0, 2].set_title('Heave Acceleration')
#     ax[0, 2].set_ylim([Acceleration.min()[2]-0.1*np.abs(Acceleration.min()[2]),Acceleration.max()[2]+0.1*np.abs(Acceleration.max()[2])])
    ax[1, 0].plot(T, AngularPosition[angularpositions[1]])
    ax[1, 0].set_title('Roll Rotation')
#     ax[1, 0].set_ylim([AngularPosition.min()[0]-0.1*np.abs(AngularPosition.min()[0]),AngularPosition.max()[0]+0.1*np.abs(AngularPosition.max()[0])])
    ax[1, 1].plot(T, AngularPosition[angularpositions[0]])
    ax[1, 1].set_title('Pitch Rotation')
#     ax[1, 1].set_ylim([AngularPosition.min()[1]-0.1*np.abs(AngularPosition.min()[1]),AngularPosition.max()[1]+0.1*np.abs(AngularPosition.max()[1])])
    ax[1, 2].plot(T, AngularPosition[angularpositions[2]])
    ax[1, 2].set_title('Yaw Rotation')
#     ax[1, 2].set_ylim([AngularPosition.min()[2]-0.1*np.abs(AngularPosition.min()[2]),AngularPosition.max()[2]+0.1*np.abs(AngularPosition.max()[2])])
    
    ax[0, 0].set_ylim([-0.05+min(Acceleration['LIN_ACC_NO_GRAVITY_y']),0.05+max(Acceleration['LIN_ACC_NO_GRAVITY_y'])])
    ax[0, 1].set_ylim([-0.05+min(Acceleration['LIN_ACC_NO_GRAVITY_x']),0.05+max(Acceleration['LIN_ACC_NO_GRAVITY_x'])])
    ax[0, 2].set_ylim([-0.05+min(Acceleration['LIN_ACC_NO_GRAVITY_z']),0.05+max(Acceleration['LIN_ACC_NO_GRAVITY_z'])])
    ax[1, 0].set_ylim([-0.05+min(AngularPosition['EulerAngle_y']),0.05+max(AngularPosition['EulerAngle_y'])])
    ax[1, 1].set_ylim([-0.05+min(AngularPosition['EulerAngle_x']),0.05+max(AngularPosition['EulerAngle_x'])])
    ax[1, 2].set_ylim([-0.05+min(AngularPosition['EulerAngle_z']),0.05+max(AngularPosition['EulerAngle_z'])])
        
    st.pyplot(fig)

def Quaternion_plot(IMU,
             angularpositions=['ANGULAR_POSITION_w', 'ANGULAR_POSITION_x', 'ANGULAR_POSITION_y', 'ANGULAR_POSITION_z'],
             angularpositions_cal=['ANGULAR_POSITION_w_cal', 'ANGULAR_POSITION_x_cal', 'ANGULAR_POSITION_y_cal', 'ANGULAR_POSITION_z_cal']):
    
    T = np.array(range(IMU.shape[0])) * dt
    
    fig, ax = plt.subplots(2, 4, figsize=(15,8))
    ax[0, 0].plot(T, IMU[angularpositions[0]])
    ax[0, 0].set_title('ANGULAR_POSITION_w')
#     ax[0, 0].set_ylim([Acceleration.min()[0]-0.1*np.abs(Acceleration.min()[0]),Acceleration.max()[0]+0.1*np.abs(Acceleration.max()[0])])
    ax[0, 1].plot(T, IMU[angularpositions[1]])
    ax[0, 1].set_title('ANGULAR_POSITION_x')
#     ax[0, 1].set_ylim([Acceleration.min()[1]-0.1*np.abs(Acceleration.min()[1]),Acceleration.max()[1]+0.1*np.abs(Acceleration.max()[1])])
    ax[0, 2].plot(T, IMU[angularpositions[2]])
    ax[0, 2].set_title('ANGULAR_POSITION_y')
#     ax[0, 2].set_ylim([Acceleration.min()[2]-0.1*np.abs(Acceleration.min()[2]),Acceleration.max()[2]+0.1*np.abs(Acceleration.max()[2])])
    ax[0, 3].plot(T, IMU[angularpositions[3]])
    ax[0, 3].set_title('ANGULAR_POSITION_z')
#     ax[1, 0].set_ylim([AngularPosition.min()[0]-0.1*np.abs(AngularPosition.min()[0]),AngularPosition.max()[0]+0.1*np.abs(AngularPosition.max()[0])])
    ax[1, 0].plot(T, IMU[angularpositions_cal[0]])
    ax[1, 0].set_title('ANGULAR_POSITION_w_cal')
#     ax[0, 0].set_ylim([Acceleration.min()[0]-0.1*np.abs(Acceleration.min()[0]),Acceleration.max()[0]+0.1*np.abs(Acceleration.max()[0])])
    ax[1, 1].plot(T, IMU[angularpositions_cal[1]])
    ax[1, 1].set_title('ANGULAR_POSITION_x_cal')
#     ax[0, 1].set_ylim([Acceleration.min()[1]-0.1*np.abs(Acceleration.min()[1]),Acceleration.max()[1]+0.1*np.abs(Acceleration.max()[1])])
    ax[1, 2].plot(T, IMU[angularpositions_cal[2]])
    ax[1, 2].set_title('ANGULAR_POSITION_y_cal')
#     ax[0, 2].set_ylim([Acceleration.min()[2]-0.1*np.abs(Acceleration.min()[2]),Acceleration.max()[2]+0.1*np.abs(Acceleration.max()[2])])
    ax[1, 3].plot(T, IMU[angularpositions_cal[3]])
    ax[1, 3].set_title('ANGULAR_POSITION_z_cal')
#     ax[1, 0].set_ylim([AngularPosition.min()[0]-0.1*np.abs(AngularPosition.min()[0]),AngularPosition.max()[0]+0.1*np.abs(AngularPosition.max()[0])])

    ax[0, 0].set_ylim([-0.05+min(IMU['ANGULAR_POSITION_w']),0.05+max(IMU['ANGULAR_POSITION_w'])])
    ax[0, 1].set_ylim([-0.05+min(IMU['ANGULAR_POSITION_x']),0.05+max(IMU['ANGULAR_POSITION_x'])])
    ax[0, 2].set_ylim([-0.05+min(IMU['ANGULAR_POSITION_y']),0.05+max(IMU['ANGULAR_POSITION_y'])])
    ax[0, 3].set_ylim([-0.05+min(IMU['ANGULAR_POSITION_z']),0.05+max(IMU['ANGULAR_POSITION_z'])])
    ax[1, 0].set_ylim([-0.05+min(IMU['ANGULAR_POSITION_w_cal']),0.05+max(IMU['ANGULAR_POSITION_w_cal'])])
    ax[1, 1].set_ylim([-0.05+min(IMU['ANGULAR_POSITION_x_cal']),0.05+max(IMU['ANGULAR_POSITION_x_cal'])])
    ax[1, 2].set_ylim([-0.05+min(IMU['ANGULAR_POSITION_y_cal']),0.05+max(IMU['ANGULAR_POSITION_y_cal'])])
    ax[1, 3].set_ylim([-0.05+min(IMU['ANGULAR_POSITION_z_cal']),0.05+max(IMU['ANGULAR_POSITION_z_cal'])])
    
    st.pyplot(fig)
    
with st.sidebar:
    selected = option_menu("IMU Module", ["Home", 'Upload', 'Data', 'Info', 'Plot', 'Parameters', 'Settings'], 
           icons=['house', 'cloud-upload', 'table', 'info-circle', 'bar-chart-fill', 'sliders', 'gear'],  menu_icon="cast", default_index=0, styles={
        "container": {"padding": "0!important", "background-color": "#fafafa"},
        "icon": {"color": "orange", "font-size": "15px"}, 
        "nav-link": {"font-size": "15px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
        "nav-link-selected": {"background-color": "green"},
    })
    selected
    uploaded_file = st.file_uploader("Choose a file")
       
# if selected == "Upload":
#     uploaded_file = st.file_uploader("Choose a file")

# try:     
if uploaded_file is not None:
    IMU, info, dt = IMU_read(file=uploaded_file)

#     if selected == "Upload":        
#         uploaded_file = st.file_uploader("Choose a file")    

    if selected == "Home":
        st.header("IMU Module")
        st.write("Hello, welcome to this page!")    

    if selected == "Data":
        st.header("IMU Data")
        st.write(IMU)

    if selected == "Info":
        st.header("Summary of Data Info")
        st.write(info)

        if len(np.where(IMU.dtypes == 'object')[0]) == 0:
            st.write(f'There exist {len(np.where(IMU.isnull().sum() != 0)[0])} columns with missing Values...')
            for i in range(len(np.where(IMU.isnull().sum() != 0)[0])):
                st.write(f'{IMU.columns[np.where(IMU.isnull().sum() != 0)[0][i]]} has {IMU.isnull().sum().iloc[np.where(IMU.isnull().sum() != 0)[0][i]]} missing values!')
        else:
            obj_num = len(np.where(IMU.dtypes == 'object')[0])
            st.write(f'There exist {obj_num} columns with non-integer/non-float values. Please fix them first!')

    if selected == "Plot":
        st.header('Plot of IMU Data')
        st.subheader('IMU Rough')
        Acceleration, AngularPosition = IMU_write(file=uploaded_file)
        IMU_plot(Acceleration=Acceleration, AngularPosition=AngularPosition)

        st.subheader('Quarternions')
        Quaternion_plot(IMU=IMU)

        st.subheader('IMU Calibrated')
        Acceleration_calibrated, AngularPosition_calibrated = Calibration(Acceleration=Acceleration, AngularPosition=AngularPosition, cal_period=1)
        IMU_plot(Acceleration=Acceleration_calibrated, AngularPosition=AngularPosition_calibrated)

        st.subheader('IMU Calibrated & Filtered')
        Acceleration_filtered, AngularPosition_filtered = IMU_filter(Acceleration=Acceleration_calibrated, AngularPosition=AngularPosition_calibrated, w_c=1*pi)
        IMU_plot(Acceleration=Acceleration_filtered, AngularPosition=AngularPosition_filtered)

    if selected == "Parameters":
        st.header("Parameters Dashboar")

    if selected == "Settings":
        st.header("General Settings")    

# else:
#     if selected == "Home":
#         st.header("IMU Module")
#         st.write("Hello, welcome to this page!")  
#     if selected == "Data":
#         st.header("IMU Data")
#     if selected == "Info":
#         st.header("Summary of Data Info")
#     if selected == "Plot":
#         st.header('Plot of IMU Data')
#     if selected == "Parameters":
#         st.header("Parameters Dashboar")
#     if selected == "Settings":
#         st.header("General Settings") 

# except ValueError:
#     if selected == "Upload":
#         st.write("No file has been chosen!")