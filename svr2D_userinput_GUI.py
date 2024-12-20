import tkinter as tk
from tkinter import messagebox
from tkinter import filedialog
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, ShuffleSplit, StratifiedKFold
import joblib
import threading
import re
import time

# def parseInputData():
#     filePaths = filedialog.askopenfilenames(defaultextension=".csv", filetypes=[("CSV files", "*.csv"), ("Text files", "*.txt")])
#     # file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
#     if filePaths:  # Ensure the user has selected a valid path
#         print(f'File paths: {filePaths[0]} ')
        
#         global received_data
        
#         received_data = pd.DataFrame(pd.read_csv(filePaths[0])).to_numpy()
#         print(f'shape of data: {received_data.shape}')
#         print(f'x range:{min(received_data[:,0])}-{max(received_data[:,0])}\n\
#               y range:{min(received_data[:,1])}-{max(received_data[:,1])}\n\
#               x offset range:{min(received_data[:,2])}-{max(received_data[:,2])}\n\
#               y offset range:{min(received_data[:,3])}-{max(received_data[:,3])}\n')
        
#     else:
#         messagebox.showerror("Error", "no files found")


# ~~~~~~~~~~~~~~~~~~~GLOBAL VARIABLES~~~~~~~~~~~~~~~~~~~
offset_set_1 = np.array([(1.111, 5.111), (1.121, 5.121), (1.181, 5.101), (1.151, 5.151), (1.121, 5.171),
                         (1.211, 5.211), (1.221, 5.221), (1.281, 5.201), (1.251, 5.251), (1.221, 5.271),
                         (1.011, 5.011), (1.021, 5.021), (1.081, 5.001), (1.051, 5.051), (1.021, 5.071),
                         (1.411, 5.411), (1.421, 5.421), (1.481, 5.401), (1.451, 5.451), (1.421, 5.471),
                         (1.511, 5.511), (1.521, 5.521), (1.581, 5.501), (1.551, 5.551), (1.521, 5.571)])
# offset_set_2 = np.array([(1.115, 5.115), (1.125, 5.125), (1.185, 5.105), (1.155, 5.155), (1.125, 5.175),
#                          (1.215, 5.215), (1.225, 5.225), (1.285, 5.205), (1.255, 5.255), (1.225, 5.275),
#                          (1.015, 5.015), (1.025, 5.025), (1.085, 5.005), (1.055, 5.055), (1.025, 5.075),
#                          (1.415, 5.415), (1.425, 5.425), (1.485, 5.405), (1.455, 5.455), (1.425, 5.475),
#                          (1.515, 5.515), (1.525, 5.525), (1.585, 5.505), (1.555, 5.555), (1.525, 5.575)])
# offset_set_3 = np.array([(1.119, 5.119), (1.129, 5.129), (1.189, 5.109), (1.159, 5.159), (1.129, 5.179),
#                          (1.219, 5.219), (1.229, 5.229), (1.289, 5.209), (1.259, 5.259), (1.229, 5.279),
#                          (1.019, 5.019), (1.029, 5.029), (1.089, 5.009), (1.059, 5.059), (1.029, 5.079),
#                          (1.419, 5.419), (1.429, 5.429), (1.489, 5.409), (1.459, 5.459), (1.429, 5.479),
#                          (1.519, 5.519), (1.529, 5.529), (1.589, 5.509), (1.559, 5.559), (1.529, 5.579)])
iteration_number = 1

modelReady = False # has model been trained alrdy? loaded alrdy? used to (dis)/allow save csv button
received_data = None
predicted_data = None


def parseInputData():
    filePath = filedialog.askopenfilename(defaultextension=".txt", filetypes=[("Text files", "*.txt"), ("CSV files", "*.csv")])
    with open(filePath, 'r') as file:
            lines = file.readlines()
    print("reading lines...")

    general_params = {}
    data_section = []
    
    # parse General section
    in_general_section = False
    in_data_section = False
    for line in lines:
        if '[General]' in line:
            in_general_section = True
            print("General found")
            continue
        elif '[Data um]' in line:
            in_general_section = False
            in_data_section = True
            print("Data um found")
            continue
        elif in_general_section:
            match = re.match(r'(\w+)=(\d+)', line.strip())
            if match:
                key, value = match.groups()
                general_params[key] = int(value)
        elif in_data_section:
            if line.strip():  # skip empty lines
                data_section.append(line.strip())

    # extract general_params
    rows = general_params.get('Rows', 0)
    cols = general_params.get('Cols', 0)
    x_int = general_params.get('xInt', 0)
    y_int = general_params.get('yInt', 0)

    # parse xerr and yerr values
    xerr_list = []
    yerr_list = []
    xerr = np.zeros((rows, cols))
    yerr = np.zeros((rows, cols))

    for line in lines:
        line = line.strip() #strip trailing spaces

        # check if 'xerr' is in line, then extract value
        if 'xerr' in line:
            # find position of 'xerr=' and extract value after it
            xerr_start = line.find('xerr=') + len('xerr=')
            xerr_end = line.find(' ', xerr_start)
            if xerr_end == -1:  # if no space, value goes till the end of the line
                xerr_end = len(line)
            xerr = float(line[xerr_start:xerr_end].strip())
            xerr_list.append(xerr)

        # vheck if 'yerr' is in line and extract value
        if 'yerr' in line:
            # find position of 'yerr=' and extract value after it
            yerr_start = line.find('yerr=') + len('yerr=')
            yerr_end = line.find(' ', yerr_start)
            if yerr_end == -1:
                yerr_end = len(line)
            yerr = float(line[yerr_start:yerr_end].strip())
            yerr_list.append(yerr)

    offset_set = np.array(list(zip(xerr_list, yerr_list)))
    print(offset_set.shape)
    print(offset_set)
    
    # compute min/max for x and y coordinates
    min_x = 0
    max_x = (cols - 1) * x_int
    min_y = 0
    max_y = (rows - 1) * y_int

    # calculate interval size for x and y
    interval_x = x_int
    interval_y = y_int

    # return all computed values
    print(f'found values: {min_x}, {max_x}, {min_y}, {max_y}, {interval_x}, {interval_y}, {xerr}, {yerr}')
    min_x_entry.delete(0,'end')
    min_x_entry.insert(0, min_x)
    max_x_entry.delete(0,'end')
    max_x_entry.insert(0, max_x)
    min_y_entry.delete(0,'end')
    min_y_entry.insert(0, min_y)
    max_y_entry.delete(0,'end')
    max_y_entry.insert(0, max_y)
    intvl_x_entry.delete(0,'end')
    intvl_x_entry.insert(0, interval_x)
    intvl_y_entry.delete(0,'end')
    intvl_y_entry.insert(0, interval_y)

    global offset_set_1
    offset_set_1 = offset_set
    return min_x, max_x, min_y, max_y, interval_x, interval_y, xerr, yerr

# Function to search best SVR parameters
def searchBestSVRParams(svrmodel, input, targetOutput, useShuffleSplit=False, n_splits=5, test_size=0.2):
    startTime = time.time()
    param_grid = {
    'C': [0.1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6],  # [1e6],  # Trying a range of C values (penalty)
    'gamma': [0.001, 0.01, 0.1, 0.5, 1, 10, 100 ],  # [ 0.1],  # Trying different gamma values (imptce of each pt)
    # 'kernel': ['rbf', 'poly', 'sigmoid']  # Radial basis function kernel (commonly used for SVR)
    }
    # use GridSearchCV with X-fold cross-validation. X = value set under 'cv' parameter
    #scoring: higher score=better params. negMSE is because MSE is error. lower value=better.
    # other scoring methods: r2, mean_absolute_error, neg_mean_absolute_error. use 'accuracy' when doing classification in GridSearchCV (not applicable for SVR type applications like this one here)
    if useShuffleSplit:
        # ShuffleSplit with specified number of splits and test size
        cv = ShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=42)
        print(f"Using ShuffleSplit with {n_splits} splits and test size {test_size}")
    else:
        # regular KFold cross-validation (default 5-fold)
        cv = n_splits  
        print(f"Using {n_splits}-fold cross-validation")
    # grid_search = GridSearchCV(svrmodel, param_grid, cv=cv, scoring='neg_mean_squared_error', verbose=2, n_jobs=-1)
    grid_search = RandomizedSearchCV(svrmodel, param_grid, n_iter=10, cv=cv, scoring='neg_mean_squared_error', verbose=2, n_jobs=-1)
    
    print(" define grid search across range of svr params done")
    # fit model with data
    grid_search.fit(input, targetOutput)  # Fit on x-offsets or y-offsets separately
    print("fitting input against all svr params defined done")
    # Get best params
    best_params = grid_search.best_params_
    print(f"Best C val: {best_params['C']}, \
           Best Gamma: {best_params['gamma']},")
            #  Best Kernel: {best_params['kernel']}")

    endTime = time.time() 
    executionTime = endTime - startTime 
    print(f"execution time: {executionTime:.2f} seconds")
    return best_params['C'], best_params['gamma']

# Function to save the model
def savemodelas(modelname, savedfilename):
    joblib.dump(modelname, savedfilename)

def loadmodel(savedfilename): #usage: modelName = loadmodel("testmodel.pkl"), and then modelName.predict((3,2)) or smth
    return joblib.load(savedfilename) 
def retrieveCoordsAndOffsets():
    try:
        # Gather user inputs
        min_x = float(min_x_entry.get())
        max_x = float(max_x_entry.get())
        intvl_x = float(intvl_x_entry.get())
        min_y = float(min_y_entry.get())
        max_y = float(max_y_entry.get())
        intvl_y = float(intvl_y_entry.get())
        new_x = float(new_x_entry.get())
        new_y = float(new_y_entry.get())

        # Create coordinate arrays
        x_coords = np.arange(min_x, max_x + intvl_x, intvl_x)
        y_coords = np.arange(min_y, max_y + intvl_y, intvl_y)
        print(f'curr xcoords: {x_coords}\ncurr ycoords: {y_coords}')
        # Validate the new_x and new_y input coordinates
        if new_x > max_x or new_x < min_x:
            messagebox.showerror("Invalid input", f'Invalid X value of {new_x}, should be between {min_x}-{max_x}')
            new_x_entry.config(fg="#f00")
            return None, None, None
        else:
            new_x_entry.config(fg="#000")

        if new_y > max_y or new_y < min_y:
            messagebox.showerror("Invalid input", f'Invalid Y value of {new_y}, should be between {min_y}-{max_y}')
            new_y_entry.config(fg="#f00")
            return None, None, None
        else:
            new_y_entry.config(fg="#000")

        # Stack all offsets into single array, then combine all into single input array
        X, Y = np.meshgrid(x_coords, y_coords)
        coordinate_pairs = np.column_stack((X.ravel(), Y.ravel()))

        # Standardize coordinates
        scaler = StandardScaler()
        coords_scaled = scaler.fit_transform(coordinate_pairs)
        
        return coordinate_pairs, coords_scaled, scaler, (min_x, max_x, min_y, max_y), (new_x, new_y), (intvl_x, intvl_y)
    
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred during coordinate preparation: {str(e)}")
        return None, None, None

    
def train_model():
    # try:
        # Gather user inputs
        coordinate_pairs, coords_scaled, scaler, (min_x, max_x, min_y, max_y), (new_x, new_y), (intvl_x, intvl_y) = retrieveCoordsAndOffsets()
        iteration_number = int(iteration_number_entry.get())
        model_filename = filename_entry.get()
        print(f'params: {min_x}, {max_x}, {iteration_number}, {model_filename}')
        
        coords_scaled_nSets = np.vstack([coords_scaled] * iteration_number)

        # Create offsets based on user input (dummy values for the example)
        offsets = []
        for _ in range(iteration_number): # just loop iteration_number times
            # offsets.append(np.random.rand(len(x_coords) * len(y_coords), 2))  # Generate random offsets
            # ^ should append like this, but for now we hardcode the offsets.
            print("offset set appended to offsets array")
        # stack offsets
        offset_set = offset_set_1#np.vstack([offset_set_1, offset_set_2, offset_set_3])

        # Create SVR models and search for best parameters
        svr_x = SVR(kernel='rbf')
        svr_y = SVR(kernel='rbf')
        if customHyperParams.get() == 0: # cue the spaghetti code...
            print(f'customHyperParams is 0. disabling hyperparams frame')
            bestC_x, bestG_x = searchBestSVRParams(svr_x, coords_scaled_nSets, offset_set[:, 0], useShuffleSplit = True)
            bestC_y, bestG_y = searchBestSVRParams(svr_y, coords_scaled_nSets, offset_set[:, 1], useShuffleSplit = True)

            X_C_entry.config(state=tk.NORMAL)#enable so that i can change the values
            X_G_entry.config(state=tk.NORMAL)
            Y_C_entry.config(state=tk.NORMAL)
            Y_G_entry.config(state=tk.NORMAL)
            X_C_entry.delete(0,'end')
            X_C_entry.insert(0, bestC_x)
            X_G_entry.delete(0,'end')
            X_G_entry.insert(0, bestG_x)
            Y_C_entry.delete(0,'end')
            Y_C_entry.insert(0, bestC_y)
            Y_G_entry.delete(0,'end')
            Y_G_entry.insert(0, bestG_y)
            
            X_C_entry.config(state=tk.DISABLED)#re-disable
            X_G_entry.config(state=tk.DISABLED)
            Y_C_entry.config(state=tk.DISABLED)
            Y_G_entry.config(state=tk.DISABLED)
        else:
            print(f'customHyperParams is 1. getting values from hyperparams frame')
            bestC_x, bestG_x = float(X_C_entry.get()), float(X_G_entry.get())
            bestC_y, bestG_y = float(Y_C_entry.get()), float(Y_G_entry.get())
        svr_x = SVR(kernel='rbf', C=bestC_x, gamma=bestG_x)
        svr_y = SVR(kernel='rbf', C=bestC_y, gamma=bestG_y)

        # Fit models
        svr_x.fit(coords_scaled_nSets, offset_set[:, 0])
        svr_y.fit(coords_scaled_nSets, offset_set[:, 1])

        # save models
        savemodelas(svr_x, model_filename + "_offsetX.pkl")
        savemodelas(svr_y, model_filename + "_offsetY.pkl")

        # test case, predict offsets for a single new set of (x, y) coordinates
        new_coords = np.array([[new_x, new_y]])
        new_coords_scaled = scaler.transform(new_coords)
        
        # Predict the x and y offset 
        predicted_x_offset = svr_x.predict(new_coords_scaled)
        predicted_y_offset = svr_y.predict(new_coords_scaled)

        resultLabel.config(text=f"Predicted x-offset: {predicted_x_offset[0]}\nPredicted y-offset: {predicted_y_offset[0]}")

        # test case, predict offsets for a new range of (x, y) coordinates
        x_vals = np.linspace(min_x, max_x , int(max_x)) # generate x coordinates from 0 to max X - intvl of 100, 100 points.
        y_vals = np.linspace(min_y, max_y , int(max_y))
        X_grid, Y_grid = np.meshgrid(x_vals, y_vals)
        coords_grid = np.column_stack((X_grid.ravel(), Y_grid.ravel()))
        coords_grid_scaled = scaler.transform(coords_grid)

        # predict offsets
        predicted_x_offsets = svr_x.predict(coords_grid_scaled).reshape(X_grid.shape)
        predicted_y_offsets = svr_y.predict(coords_grid_scaled).reshape(Y_grid.shape)

        # get index of predicted offset
        x_index = np.where(np.isclose(x_vals, new_x, atol = 0.8))[0]  # rough index of new_x in autogenerated x_vals range, w absolute tolerance range specified
        y_index = np.where(np.isclose(y_vals, new_y, atol = 0.8))[0]  # rough index of new_y in autogenerated y_vals range
        if x_index.size == 0 or y_index.size == 0:
            raise ValueError(f"New X of {new_x} or New Y of {new_y} not found within the grid.")
        # Get first (and hopefully only) match
        x_index = x_index[0]
        y_index = y_index[0]  
        print(f'x index: {x_index}|y index: {y_index}')
        # Generate and display plots
        print("time to generate plots")
        generate_plots(coordinate_pairs, offset_set, x_index, y_index, new_x, new_y, x_vals, y_vals, predicted_x_offset, predicted_y_offset, predicted_x_offsets, predicted_y_offsets)
        global modelReady
        modelReady = True
        saveButton.config(state=tk.NORMAL) # enable savebutton once done
    # except Exception as e:
        # messagebox.showerror("Error", f"An error occurred: {str(e)}")

def predict_model():
    try:
        coordinate_pairs, coords_scaled, scaler, (min_x, max_x, min_y, max_y), (new_x, new_y), (intvl_x, intvl_y) = retrieveCoordsAndOffsets()
        model_filename = filename_entry.get()
        
        
        # if loading another model, use the following line:
        svr_x = loadmodel(model_filename + "_offsetX.pkl")
        svr_y = loadmodel(model_filename + "_offsetY.pkl")

        # Standardize coordinates
        scaler = StandardScaler()
        coords_scaled = scaler.fit_transform(coordinate_pairs)
        coords_scaled_nSets = np.vstack([coords_scaled] * iteration_number)
        
        # stack offsets
        offset_set = offset_set_1#np.vstack([offset_set_1, offset_set_2, offset_set_3])
        
        # Fit models
        svr_x.fit(coords_scaled_nSets, offset_set[:, 0])
        svr_y.fit(coords_scaled_nSets, offset_set[:, 1])

        # test case, predict offsets for a single new set of (x, y) coordinates
        new_coords = np.array([[new_x, new_y]])
        new_coords_scaled = scaler.transform(new_coords)
        
        # Predict the x and y offset 
        predicted_x_offset = svr_x.predict(new_coords_scaled)
        predicted_y_offset = svr_y.predict(new_coords_scaled)

        resultLabel.config(text=f"Predicted x-offset: {predicted_x_offset[0]}\nPredicted y-offset: {predicted_y_offset[0]}")
        
        # test case, predict offsets for a new range of (x, y) coordinates
        x_vals = np.linspace(min_x, max_x, int(max_x))
        y_vals = np.linspace(min_y, max_y, int(max_y))
        X_grid, Y_grid = np.meshgrid(x_vals, y_vals)
        coords_grid = np.column_stack((X_grid.ravel(), Y_grid.ravel()))
        coords_grid_scaled = scaler.transform(coords_grid)

        # Predict offsets for the entire grid
        predicted_x_offsets = svr_x.predict(coords_grid_scaled).reshape(X_grid.shape)
        predicted_y_offsets = svr_y.predict(coords_grid_scaled).reshape(Y_grid.shape)

        # reshape and return in same format as offset_set_1
        predicted_offsets = np.column_stack((
            predicted_x_offsets.flatten(),
            predicted_y_offsets.flatten()
        ))
        global predicted_data 
        predicted_data = np.column_stack((X_grid.flatten(), Y_grid.flatten(), predicted_offsets))
        
        print(f'shape of offset:{offset_set_1.shape}|SHAPE OF X: {predicted_x_offsets.shape}||shape of predicted offset: {predicted_offsets.shape}')

        # get index of predicted offset
        x_index = np.where(np.isclose(x_vals, new_x, atol = 0.8))[0]  # rough index of new_x in autogenerated x_vals range, w absolute tolerance range specified
        y_index = np.where(np.isclose(y_vals, new_y, atol = 0.8))[0]  # rough index of new_y in autogenerated y_vals range
        if x_index.size == 0 or y_index.size == 0:
            raise ValueError(f"New coordinates {new_x} or {new_y} not found in the grid.")
        # Get first (and hopefully only) match
        x_index = x_index[0]
        y_index = y_index[0]  
        print(f'x index: {x_index}|y index: {y_index}')
        # Generate and display plots
        generate_plots(coordinate_pairs, offset_set, x_index, y_index, new_x, new_y, x_vals, y_vals, predicted_x_offset, predicted_y_offset, predicted_x_offsets, predicted_y_offsets)
        global modelReady
        modelReady = True
        saveButton.config(state=tk.NORMAL) # enable savebutton once done
    except Exception as e:                  
        messagebox.showerror("Error", f"An error occurred: {str(e)}")

# Function to generate plots
def generate_plots(coordinate_pairs, offset_set, x_index, y_index, new_x, new_y, x_vals, y_vals, predicted_x_offset, predicted_y_offset, predicted_x_offsets, predicted_y_offsets):
    # Clear previous plot if any
    for widget in plotFrame.winfo_children():
        widget.destroy()  # Removes old plots from plot_frame
    
    X_grid, Y_grid = np.meshgrid(x_vals, y_vals)
    fig = plt.figure(figsize=(9, 8))

    # Plot predicted x-offsets (3D Surface)
    ax1 = fig.add_subplot(321, projection='3d')
    # print(f'{offset_set_1[:,0]}\n\n{offset_set_1[:,1]}')
    ax1.scatter(coordinate_pairs[:, 0], coordinate_pairs[:, 1], offset_set_1[:, 0], c='r', label='offset dataset 1', s=10)
    # ax1.scatter(coordinate_pairs[:, 0], coordinate_pairs[:, 1], offset_set_2[:, 0], c='b', label='offset dataset 2', s=10)
    # ax1.scatter(coordinate_pairs[:, 0], coordinate_pairs[:, 1], offset_set_3[:, 0], c='g', label='offset dataset 3', s=10)
    ax1.scatter(new_x, new_y, predicted_x_offset, c='c', label=f'Predicted offset at {new_x, new_y}')
    ax1.plot_surface(X_grid, Y_grid, predicted_x_offsets, cmap='viridis', edgecolor='none', alpha=0.7)
    ax1.set_xlabel('X coordinate')
    ax1.set_ylabel('Y coordinate')
    ax1.set_zlabel('Predicted x-offset')
    ax1.set_title('SVR - X offset prediction')
    ax1.legend(bbox_to_anchor=(0,1))


    # 2D Plot for X offset vs X coordinate
    ax11 = fig.add_subplot(323)
    ax11.scatter(coordinate_pairs[:, 0], offset_set_1[:, 0], c='r', s=10)
    # ax11.scatter(coordinate_pairs[:, 0], offset_set_2[:, 0], c='b', s=10)
    # ax11.scatter(coordinate_pairs[:, 0], offset_set_3[:, 0], c='g', s=10)

    # Plot predicted x-offsets over x_vals, sliced along New Y value
    ax11.plot(x_vals, predicted_x_offsets[y_index, :], color='c', label="Predicted x-offsets") # minus 1 due to indexing. round off if it's decimal points, as this is just for visualisation
    # ^ instead of guessing, fix the following: y_index = np.where(Y_vals == new_y)[0][0]  # Index of Y = 6
    ax11.scatter(new_x, predicted_x_offset, c='c', label=f'Predicted Offset at {new_x, new_y}')
    ax11.set_xlabel('X coordinate')
    ax11.set_ylabel('Offset')
    ax11.set_title('SVR - X offset prediction')
    ax11.legend(bbox_to_anchor=(0,1))

    # 2D Plot for X offset vs Y coordinate
    ax12 = fig.add_subplot(325)
    ax12.scatter(coordinate_pairs[:, 1], offset_set_1[:, 0], c='r', s=10)
    # ax12.scatter(coordinate_pairs[:, 1], offset_set_2[:, 0], c='b', s=10)
    # ax12.scatter(coordinate_pairs[:, 1], offset_set_3[:, 0], c='g', s=10)

    # Plot predicted x-offsets over y_vals,  sliced along New X value
    ax12.plot(y_vals, predicted_x_offsets[:, x_index], color='c', label="Predicted x-offsets")
    ax12.scatter(new_y, predicted_x_offset, c='c', label=f'Predicted Offset at {new_x, new_y}')
    ax12.set_xlabel('Y coordinate')
    ax12.set_ylabel('Offset')
    ax12.set_title('SVR - X offset prediction')
    # ax12.legend(loc="upper left")

    # Predicted y-offsets (3D Surface)
    ax2 = fig.add_subplot(322, projection='3d')
    ax2.scatter(coordinate_pairs[:, 0], coordinate_pairs[:, 1], offset_set_1[:, 1], c='r', s=10)
    # ax2.scatter(coordinate_pairs[:, 0], coordinate_pairs[:, 1], offset_set_2[:, 1], c='b', s=10)
    # ax2.scatter(coordinate_pairs[:, 0], coordinate_pairs[:, 1], offset_set_3[:, 1], c='g', s=10)
    ax2.scatter(new_x, new_y, predicted_y_offset, c='c', label=f'Predicted Offset at {new_x, new_y}')
    ax2.plot_surface(X_grid, Y_grid, predicted_y_offsets, cmap='viridis', edgecolor='none', alpha=0.7)
    ax2.set_xlabel('X coordinate')
    ax2.set_ylabel('Y coordinate')
    ax2.set_zlabel('Predicted y-offset')
    ax2.set_title('SVR - Y offset prediction')
    # ax2.legend(loc="upper left")

    # 2D Plot for Y offset vs X coordinate
    ax21 = fig.add_subplot(324)
    ax21.scatter(coordinate_pairs[:, 0], offset_set_1[:, 1], c='r', s=10)
    # ax21.scatter(coordinate_pairs[:, 0], offset_set_2[:, 1], c='b', s=10)
    # ax21.scatter(coordinate_pairs[:, 0], offset_set_3[:, 1], c='g', s=10)

    # Plot predicted y-offsets over x_vals
    ax21.plot(x_vals, predicted_y_offsets[y_index, :], color='c', label="Predicted y-offsets")
    ax21.scatter(new_x, predicted_y_offset, c='c', label=f'Predicted Offset at {new_x, new_y}')
    ax21.set_xlabel('X coordinate')
    ax21.set_ylabel('Offset')
    ax21.set_title('SVR - Y offset prediction')
    # ax21.legend()

    # 2D Plot for Y offset vs Y coordinate
    ax22 = fig.add_subplot(326)
    ax22.scatter(coordinate_pairs[:, 1], offset_set_1[:, 1], c='r', s=10)
    # ax22.scatter(coordinate_pairs[:, 1], offset_set_2[:, 1], c='b', s=10)
    # ax22.scatter(coordinate_pairs[:, 1], offset_set_3[:, 1], c='g', s=10)

    # Plot predicted y-offsets over y_vals
    ax22.plot(y_vals, predicted_y_offsets[:, x_index], color='c', label="Predicted y-offsets")
    ax22.scatter(new_y, predicted_y_offset, c='c', label=f'Predicted Offset at {new_x, new_y}')
    ax22.set_xlabel('Y coordinate')
    ax22.set_ylabel('Offset')
    ax22.set_title('SVR - Y offset prediction')
    # ax22.legend()

    plt.tight_layout()
    # plt.show()

    # Display the plot in the Tkinter window
    canvas = FigureCanvasTkAgg(fig, master=plotFrame)
    canvas.draw()
    canvas.get_tk_widget().pack()

# GUI setup
root = tk.Tk()
root.title("SVR Model Training")

def on_close():
    root.quit()

def saveToCSV():
    global predicted_data
    if predicted_data is None:
        messagebox.showerror("Error", f"nothing to save! Please train or predict the model first.")
        return
    df = pd.DataFrame(predicted_data, columns=['x', 'y', 'predicted_x_offset', 'predicted_y_offset'])
    file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
    if file_path:  # Ensure the user has selected a valid path
        df.to_csv(file_path, index=False)  # Save the DataFrame to CSV
        print(f"Data saved to {file_path}")
        messagebox.showinfo("Data saved", f"Data saved to {file_path}")
    else:
        print("Save operation was cancelled.")
        messagebox.showinfo("Save operation cancelled", "Save operation was canceled.")

def onCheckboxToggle():
    
    global X_C_label, X_C_entry, X_G_label, X_G_entry, Y_C_label, Y_C_entry, Y_G_label, Y_G_entry  # declare as global so they can be retrieved in train_model
    if customHyperParams.get() == 1:
        print(f"Checkbutton is selected. customHyperParams value:{customHyperParams.get()}")
        for child in hyperparams_frame.winfo_children():
            child.configure(state=tk.NORMAL)

    else:
        
        print(f"Checkbutton is deselected. customHyperParams value:{customHyperParams.get()}")
        for child in hyperparams_frame.winfo_children():
            child.configure(state=tk.DISABLED)



root.protocol("WM_DELETE_WINDOW", on_close)

# create frame for input fields
input_frame = tk.Frame(root)
input_frame.grid(row=0, column=0, padx=10, pady=30, sticky="n")
hyperparams_frame = tk.Frame(input_frame)
hyperparams_frame.grid(row=15, column=0, padx=10, pady=5, sticky="n")
terminal_frame = tk.Frame(input_frame)
terminal_frame.grid(row=16, column=0, padx=10, pady=5, sticky="n")
# Define labels and entries for parameters
tk.Label(input_frame, text="Input data:").grid(row=0, column=0, sticky='w')
offsets_entry = tk.Button(input_frame, text="Select Files", command=parseInputData)
offsets_entry.grid(row=0, column=1)

tk.Label(input_frame, text="Min X:").grid(row=1, column=0, sticky='w')
min_x_entry = tk.Entry(input_frame)
min_x_entry.grid(row=1, column=1)
min_x_entry.insert(0, "0")

tk.Label(input_frame, text="Max X:").grid(row=2, column=0, sticky='w')
max_x_entry = tk.Entry(input_frame)
max_x_entry.grid(row=2, column=1)
max_x_entry.insert(0, "400")

tk.Label(input_frame, text="Interval X:").grid(row=3, column=0, sticky='w')
intvl_x_entry = tk.Entry(input_frame)
intvl_x_entry.grid(row=3, column=1)
intvl_x_entry.insert(0, "100")

tk.Label(input_frame, text="Min Y:").grid(row=4, column=0, sticky='w')
min_y_entry = tk.Entry(input_frame)
min_y_entry.grid(row=4, column=1)
min_y_entry.insert(0, "0")

tk.Label(input_frame, text="Max Y:").grid(row=5, column=0, sticky='w')
max_y_entry = tk.Entry(input_frame)
max_y_entry.grid(row=5, column=1)
max_y_entry.insert(0, "400")

tk.Label(input_frame, text="Interval Y:").grid(row=6, column=0, sticky='w')
intvl_y_entry = tk.Entry(input_frame)
intvl_y_entry.grid(row=6, column=1)
intvl_y_entry.insert(0, "100")

tk.Label(input_frame, text="Iteration number:").grid(row=7, column=0, sticky='w')
iteration_number_entry = tk.Entry(input_frame)
iteration_number_entry.grid(row=7, column=1)
iteration_number_entry.insert(0, "1")

tk.Label(input_frame, text="New X:").grid(row=8, column=0, sticky='w')
new_x_entry = tk.Entry(input_frame)
new_x_entry.grid(row=8, column=1)
new_x_entry.insert(0, "30")

tk.Label(input_frame, text="New Y:").grid(row=9, column=0, sticky='w')
new_y_entry = tk.Entry(input_frame)
new_y_entry.grid(row=9, column=1)
new_y_entry.insert(0, "6")

tk.Label(input_frame, text="Filename for model:").grid(row=10, column=0, sticky='w')
filename_entry = tk.Entry(input_frame)
filename_entry.grid(row=10, column=1)
filename_entry.insert(0, "testGUI")

# Result label for predicted offsets
resultLabel = tk.Label(input_frame, text="")
resultLabel.grid(row=11, column=1)

# Train button
trainButton = tk.Button(input_frame, text="Train Model", command=train_model)
trainButton.grid(row=12, column=0)

# Predict button
predictButton = tk.Button(input_frame, text="Predict", command=predict_model)
predictButton.grid(row=12, column=1)

plotFrame = tk.Frame(root)
plotFrame.grid(row=0, column=1, padx=10, pady=10)

saveButton = tk.Button(input_frame, text="Save to CSV", command=saveToCSV, state=tk.DISABLED)
saveButton.grid(row=13, column=1)

customHyperParams = tk.IntVar()
checkButton = tk.Checkbutton(input_frame, text="Custom Select SVR Hyperparams", onvalue=1, offvalue=0, variable=customHyperParams, command=onCheckboxToggle)
checkButton.grid(row=14,column=0)

X_C_label = tk.Label(hyperparams_frame, text="Best C value (X):", state=tk.DISABLED)
X_C_label.grid(row=15, column=0, sticky='w') #sticky=w:stick to west of grid
X_C_entry = tk.Entry(hyperparams_frame)
X_C_entry.grid(row=15, column=1)
X_C_entry.insert(0, "1000000")
X_C_entry.config(state=tk.DISABLED)
X_G_label = tk.Label(hyperparams_frame, text="Best Gamma (X):", state=tk.DISABLED)
X_G_label.grid(row=16, column=0, sticky='w')
X_G_entry = tk.Entry(hyperparams_frame)
X_G_entry.grid(row=16, column=1)
X_G_entry.insert(0, "0.1")
X_G_entry.config(state=tk.DISABLED)

Y_C_label = tk.Label(hyperparams_frame, text="Best C value (Y):", state=tk.DISABLED)
Y_C_label.grid(row=17, column=0, sticky='w')
Y_C_entry = tk.Entry(hyperparams_frame)
Y_C_entry.grid(row=17, column=1)
Y_C_entry.insert(0, "1000000")
Y_C_entry.config(state=tk.DISABLED)
Y_G_label = tk.Label(hyperparams_frame, text="Best Gamma (Y):", state=tk.DISABLED)
Y_G_label.grid(row=18, column=0, sticky='w')
Y_G_entry = tk.Entry(hyperparams_frame)
Y_G_entry.grid(row=18, column=1)
Y_G_entry.insert(0, "0.1")
Y_G_entry.config(state=tk.DISABLED)

terminal_label = tk.Label(terminal_frame, text="debugging output", state=tk.NORMAL)
# Run the application
root.mainloop()