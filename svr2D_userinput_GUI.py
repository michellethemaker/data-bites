import tkinter as tk
from tkinter import messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import joblib

# Define function to train the model and save it
offset_set_1 = np.array([(111.1, 511.1), (112.1, 512.1), (118.1, 510.1), (115.1, 515.1), (112.1, 517.1),
                         (121.1, 521.1), (122.1, 522.1), (128.1, 520.1), (125.1, 525.1), (122.1, 527.1),
                         (101.1, 501.1), (102.1, 502.1), (108.1, 500.1), (105.1, 505.1), (102.1, 507.1),
                         (141.1, 541.1), (142.1, 542.1), (148.1, 540.1), (145.1, 545.1), (142.1, 547.1),
                         (151.1, 551.1), (152.1, 552.1), (158.1, 550.1), (155.1, 555.1), (152.1, 557.1)])
offset_set_2 = np.array([(111.5, 511.5), (112.5, 512.5), (118.5, 510.5), (115.5, 515.5), (112.5, 517.5),
                         (121.5, 521.5), (122.5, 522.5), (128.5, 520.5), (125.5, 525.5), (122.5, 527.5),
                         (101.5, 501.5), (102.5, 502.5), (108.5, 500.5), (105.5, 505.5), (102.5, 507.5),
                         (141.5, 541.5), (142.5, 542.5), (148.5, 540.5), (145.5, 545.5), (142.5, 547.5),
                         (151.5, 551.5), (152.5, 552.5), (158.5, 550.5), (155.5, 555.5), (152.5, 557.5)])
offset_set_3 = np.array([(111.9, 511.9), (112.9, 512.9), (118.9, 510.9), (115.9, 515.9), (112.9, 517.9),
                         (121.9, 521.9), (122.9, 522.9), (128.9, 520.9), (125.9, 525.9), (122.9, 527.9),
                         (101.9, 501.9), (102.9, 502.9), (108.9, 500.9), (105.9, 505.9), (102.9, 507.9),
                         (141.9, 541.9), (142.9, 542.9), (148.9, 540.9), (145.9, 545.9), (142.9, 547.9),
                         (151.9, 551.9), (152.9, 552.9), (158.9, 550.9), (155.9, 555.9), (152.9, 557.9)])
iteration_number = 3



# Function to search best SVR parameters
def searchBestSVRParams(svrmodel, input, targetOutput):
    param_grid = {
    'C': [1e6],  # Trying a range of C values (penalty)
    'gamma': [ 0.1],  # Trying different gamma values (imptce of each pt)
    # 'kernel': ['rbf', 'poly', 'sigmoid']  # Radial basis function kernel (commonly used for SVR)
    }
    # use GridSearchCV with X-fold cross-validation. X = value set under 'cv' parameter
    #scoring: higher score=better params. negMSE is because MSE is error. lower value=better.
    # other scoring methods: r2, mean_absolute_error, neg_mean_absolute_error. use 'accuracy' when doing classification in GridSearchCV (not applicable for SVR type applications like this one here)
    grid_search = GridSearchCV(svrmodel, param_grid, cv=5, scoring='neg_mean_squared_error', verbose=2)
    print(" define grid search across range of svr params done")
    # Fit model with data
    grid_search.fit(input, targetOutput)  # Fit on x-offsets or y-offsets separately
    print("fitting input against all svr params defined done")
    # Get best params
    best_params = grid_search.best_params_
    print(f"Best C val: {best_params['C']}, \
           Best Gamma: {best_params['gamma']},")
            #  Best Kernel: {best_params['kernel']}")
    return best_params['C'], best_params['gamma']

# Function to save the model
def savemodelas(modelname, savedfilename):
    joblib.dump(modelname, savedfilename)

def loadmodel(savedfilename): #usage: modelName = loadmodel("testmodel.pkl"), and then modelName.predict((3,2)) or smth
    return joblib.load(savedfilename) 

def predict_model():
    try:
        min_x = float(min_x_entry.get())
        max_x = float(max_x_entry.get())
        intvl_x = float(intvl_x_entry.get())
        min_y = float(min_y_entry.get())
        max_y = float(max_y_entry.get())
        intvl_y = float(intvl_y_entry.get())
        new_x = float(new_x_entry.get())
        new_y = float(new_y_entry.get())
        model_filename = filename_entry.get()
        
        x_coords = np.arange(min_x, max_x, intvl_x)
        y_coords = np.arange(min_y, max_y, intvl_y)#use arange for floating point; range is for intvls only


        # Stack all offsets into single array, then combine all into single input array
        X, Y = np.meshgrid(x_coords, y_coords)
        # Stack X and Y into pairs of coordinates
        coordinate_pairs = np.column_stack((X.ravel(), Y.ravel()))
        # if loading another model, use the following line:
        svr_x = loadmodel(model_filename + "_offsetX.pkl")
        svr_y = loadmodel(model_filename + "_offsetY.pkl")

        # Standardize coordinates
        scaler = StandardScaler()
        coords_scaled = scaler.fit_transform(coordinate_pairs)
        coords_scaled_nSets = np.vstack([coords_scaled] * iteration_number)
        
        # stack offsets
        offset_set = np.vstack([offset_set_1, offset_set_2, offset_set_3])
        
        # Fit models
        svr_x.fit(coords_scaled_nSets, offset_set[:, 0])
        svr_y.fit(coords_scaled_nSets, offset_set[:, 1])

        # test case, predict offsets for a single new set of (x, y) coordinates
        new_coords = np.array([[new_x, new_y]])
        new_coords_scaled = scaler.transform(new_coords)
        
        # Predict the x and y offset 
        predicted_x_offset = svr_x.predict(new_coords_scaled)
        predicted_y_offset = svr_y.predict(new_coords_scaled)

        result_label.config(text=f"Predicted x-offset: {predicted_x_offset[0]}\nPredicted y-offset: {predicted_y_offset[0]}")
        
        # test case, predict offsets for a new range of (x, y) coordinates
        x_vals = np.linspace(min_x, max_x - intvl_x, int(max_x))
        y_vals = np.linspace(min_y, max_y - intvl_y, int(max_y))
        X_grid, Y_grid = np.meshgrid(x_vals, y_vals)
        coords_grid = np.column_stack((X_grid.ravel(), Y_grid.ravel()))
        coords_grid_scaled = scaler.transform(coords_grid)

        # Predict offsets for the entire grid
        predicted_x_offsets = svr_x.predict(coords_grid_scaled).reshape(X_grid.shape)
        predicted_y_offsets = svr_y.predict(coords_grid_scaled).reshape(Y_grid.shape)

        # Generate and display plots
        generate_plots(coordinate_pairs, offset_set, new_x, new_y, x_vals, y_vals, predicted_x_offset, predicted_y_offset, predicted_x_offsets, predicted_y_offsets)
    except Exception as e:                  
        messagebox.showerror("Error", f"An error occurred: {str(e)}")
    
def train_model():
    # try:
        # Gather user inputs
        
        min_x = float(min_x_entry.get())
        max_x = float(max_x_entry.get())
        intvl_x = float(intvl_x_entry.get())
        min_y = float(min_y_entry.get())
        max_y = float(max_y_entry.get())
        intvl_y = float(intvl_y_entry.get())
        iteration_number = int(iteration_number_entry.get())
        new_x = float(new_x_entry.get())
        new_y = float(new_y_entry.get())
        model_filename = filename_entry.get()
        print(f'params: {min_x}, {max_x}, {iteration_number}, {model_filename}')
        # Create coordinate arrays
        x_coords = np.arange(min_x, max_x, intvl_x)
        y_coords = np.arange(min_y, max_y, intvl_y)
        

        # Stack all offsets into single array, then combine all into single input array
        X, Y = np.meshgrid(x_coords, y_coords)
        # Stack X and Y into pairs of coordinates
        coordinate_pairs = np.column_stack((X.ravel(), Y.ravel()))

        # Standardize coordinates
        scaler = StandardScaler()
        coords_scaled = scaler.fit_transform(coordinate_pairs)
        coords_scaled_nSets = np.vstack([coords_scaled] * iteration_number)

        # Create offsets based on user input (dummy values for the example)
        offsets = []
        for _ in range(iteration_number):
            # offsets.append(np.random.rand(len(x_coords) * len(y_coords), 2))  # Generate random offsets
            # ^ should append like this, but for now we hardcode the offsets.
            print("offset set appended to offsets array")
        # stack offsets
        offset_set = np.vstack([offset_set_1, offset_set_2, offset_set_3])

        # Create SVR models and search for best parameters
        svr_x = SVR(kernel='rbf')
        svr_y = SVR(kernel='rbf')
        if customHyperParams == 0:
            bestC_x, bestG_x = searchBestSVRParams(svr_x, coords_scaled_nSets, offset_set[:, 0])
            bestC_y, bestG_y = searchBestSVRParams(svr_y, coords_scaled_nSets, offset_set[:, 1])
        else:
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

        result_label.config(text=f"Predicted x-offset: {predicted_x_offset[0]}\nPredicted y-offset: {predicted_y_offset[0]}")

        # test case, predict offsets for a new range of (x, y) coordinates
        x_vals = np.linspace(min_x, max_x - intvl_x, int(max_x)) # generate x coordinates from 0 to max X - intvl of 100, 100 points.
        y_vals = np.linspace(min_y, max_y - intvl_y, int(max_y))
        X_grid, Y_grid = np.meshgrid(x_vals, y_vals)
        coords_grid = np.column_stack((X_grid.ravel(), Y_grid.ravel()))
        coords_grid_scaled = scaler.transform(coords_grid)

        # predict offsets gi
        predicted_x_offsets = svr_x.predict(coords_grid_scaled).reshape(X_grid.shape)
        predicted_y_offsets = svr_y.predict(coords_grid_scaled).reshape(Y_grid.shape)

        # Generate and display plots
        print("time to generate plots")
        generate_plots(coordinate_pairs, offset_set, new_x, new_y, x_vals, y_vals, predicted_x_offset, predicted_y_offset, predicted_x_offsets, predicted_y_offsets)

    # except Exception as e:
        # messagebox.showerror("Error", f"An error occurred: {str(e)}")


# Function to generate plots
def generate_plots(coordinate_pairs, offset_set, new_x, new_y, x_vals, y_vals, predicted_x_offset, predicted_y_offset, predicted_x_offsets, predicted_y_offsets):
    # Clear previous plot if any
    for widget in plot_frame.winfo_children():
        widget.destroy()  # Removes old plots from plot_frame
    
    X_grid, Y_grid = np.meshgrid(x_vals, y_vals)
    fig = plt.figure(figsize=(8, 8))

    # Plot predicted x-offsets (3D Surface)
    ax1 = fig.add_subplot(321, projection='3d')
    # print(f'{offset_set_1[:,0]}\n\n{offset_set_1[:,1]}')
    ax1.scatter(coordinate_pairs[:, 0], coordinate_pairs[:, 1], offset_set_1[:, 0], c='r', label='offset dataset 1')
    ax1.scatter(coordinate_pairs[:, 0], coordinate_pairs[:, 1], offset_set_2[:, 0], c='b', label='offset dataset 2')
    ax1.scatter(coordinate_pairs[:, 0], coordinate_pairs[:, 1], offset_set_3[:, 0], c='g', label='offset dataset 3')
    ax1.scatter(new_x, new_y, predicted_x_offset, c='c', label=f'Predicted {new_x, new_y} Point')
    ax1.plot_surface(X_grid, Y_grid, predicted_x_offsets, cmap='viridis', edgecolor='none', alpha=0.7)
    ax1.set_xlabel('X coordinate')
    ax1.set_ylabel('Y coordinate')
    ax1.set_zlabel('Predicted x-offset')
    ax1.set_title('SVR - X offset prediction')
    ax1.legend(bbox_to_anchor=(0,1))


    # 2D Plot for X offset vs X coordinate
    ax11 = fig.add_subplot(323)
    ax11.scatter(coordinate_pairs[:, 0], offset_set_1[:, 0], c='r')
    ax11.scatter(coordinate_pairs[:, 0], offset_set_2[:, 0], c='b')
    ax11.scatter(coordinate_pairs[:, 0], offset_set_3[:, 0], c='g')

    # Plot predicted x-offsets over x_vals, sliced along New Y value
    ax11.plot(x_vals, predicted_x_offsets[int(new_y) - 1, :], color='c', label="Predicted x-offsets") # minus 1 due to indexing. round off if it's decimal points, as this is just for visualisation
    # ^ instead of guessing, fix the following: y_index = np.where(Y_vals == new_y)[0][0]  # Index of Y = 6
    ax11.scatter(new_x, predicted_x_offset, c='c', label=f'Predicted {new_x, new_y} Point')
    ax11.set_xlabel('X coordinate')
    ax11.set_ylabel('Offset')
    ax11.set_title('SVR - X offset prediction')
    ax11.legend(bbox_to_anchor=(0,1))

    # 2D Plot for X offset vs Y coordinate
    ax12 = fig.add_subplot(325)
    ax12.scatter(coordinate_pairs[:, 1], offset_set_1[:, 0], c='r')
    ax12.scatter(coordinate_pairs[:, 1], offset_set_2[:, 0], c='b')
    ax12.scatter(coordinate_pairs[:, 1], offset_set_3[:, 0], c='g')

    # Plot predicted x-offsets over y_vals,  sliced along New X value
    ax12.plot(y_vals, predicted_x_offsets[:, int(new_x) - 1], color='c', label="Predicted x-offsets")
    ax12.scatter(new_y, predicted_x_offset, c='c', label=f'Predicted {new_x, new_y} Point')
    ax12.set_xlabel('Y coordinate')
    ax12.set_ylabel('Offset')
    ax12.set_title('SVR - X offset prediction')
    # ax12.legend(loc="upper left")

    # Predicted y-offsets (3D Surface)
    ax2 = fig.add_subplot(322, projection='3d')
    ax2.scatter(coordinate_pairs[:, 0], coordinate_pairs[:, 1], offset_set_1[:, 1], c='r')
    ax2.scatter(coordinate_pairs[:, 0], coordinate_pairs[:, 1], offset_set_2[:, 1], c='b')
    ax2.scatter(coordinate_pairs[:, 0], coordinate_pairs[:, 1], offset_set_3[:, 1], c='g')
    ax2.scatter(new_x, new_y, predicted_y_offset, c='c', label=f'Predicted {new_x, new_y} Point')
    ax2.plot_surface(X_grid, Y_grid, predicted_y_offsets, cmap='viridis', edgecolor='none', alpha=0.7)
    ax2.set_xlabel('X coordinate')
    ax2.set_ylabel('Y coordinate')
    ax2.set_zlabel('Predicted y-offset')
    ax2.set_title('SVR - Y offset prediction')
    # ax2.legend(loc="upper left")

    # 2D Plot for Y offset vs X coordinate
    ax21 = fig.add_subplot(324)
    ax21.scatter(coordinate_pairs[:, 0], offset_set_1[:, 1], c='r')
    ax21.scatter(coordinate_pairs[:, 0], offset_set_2[:, 1], c='b')
    ax21.scatter(coordinate_pairs[:, 0], offset_set_3[:, 1], c='g')

    # Plot predicted y-offsets over x_vals
    ax21.plot(x_vals, predicted_y_offsets[int(new_y) - 1, :], color='c', label="Predicted y-offsets")
    ax21.scatter(new_x, predicted_y_offset, c='c', label=f'Predicted {new_x, new_y} Point')
    ax21.set_xlabel('X coordinate')
    ax21.set_ylabel('Offset')
    ax21.set_title('SVR - Y offset prediction')
    # ax21.legend()

    # 2D Plot for Y offset vs Y coordinate
    ax22 = fig.add_subplot(326)
    ax22.scatter(coordinate_pairs[:, 1], offset_set_1[:, 1], c='r')
    ax22.scatter(coordinate_pairs[:, 1], offset_set_2[:, 1], c='b')
    ax22.scatter(coordinate_pairs[:, 1], offset_set_3[:, 1], c='g')

    # Plot predicted y-offsets over y_vals
    ax22.plot(y_vals, predicted_y_offsets[:, int(new_x) - 1], color='c', label="Predicted y-offsets")
    ax22.scatter(new_y, predicted_y_offset, c='c', label=f'Predicted {new_x, new_y} Point')
    ax22.set_xlabel('Y coordinate')
    ax22.set_ylabel('Offset')
    ax22.set_title('SVR - Y offset prediction')
    # ax22.legend()

    plt.tight_layout()
    # plt.show()

    # Display the plot in the Tkinter window
    canvas = FigureCanvasTkAgg(fig, master=plot_frame)
    canvas.draw()
    canvas.get_tk_widget().pack()

# GUI setup
root = tk.Tk()
root.title("SVR Model Training")

def on_close():
    root.quit()

def onCheckboxToggle():
    global X_C_label, X_C_entry, X_G_label, X_G_entry, Y_C_label, Y_C_entry, Y_G_label, Y_G_entry  # declare as global so they can be retrieved in train_model
    if customHyperParams.get() == 1:
        print("Checkbutton is selected")
        
        X_C_label = tk.Label(input_frame, text="Best C value (X):")
        X_C_label.grid(row=14, column=0, sticky='w') #sticky=w:stick to west of grid
        X_C_entry = tk.Entry(input_frame)
        X_C_entry.grid(row=14, column=1)
        X_C_entry.insert(0, "1000000")
        X_G_label = tk.Label(input_frame, text="Best Gamma (X):")
        X_G_label.grid(row=15, column=0, sticky='w')
        X_G_entry = tk.Entry(input_frame)
        X_G_entry.grid(row=15, column=1)
        X_G_entry.insert(0, "0.1")


        Y_C_label = tk.Label(input_frame, text="Best C value (Y):")
        Y_C_label.grid(row=16, column=0, sticky='w')
        Y_C_entry = tk.Entry(input_frame)
        Y_C_entry.grid(row=16, column=1)
        Y_C_entry.insert(0, "1000000")
        Y_G_label = tk.Label(input_frame, text="Best Gamma (Y):")
        Y_G_label.grid(row=17, column=0, sticky='w')
        Y_G_entry = tk.Entry(input_frame)
        Y_G_entry.grid(row=17, column=1)
        Y_G_entry.insert(0, "0.1")
    else:
        print("Checkbutton is deselected")
        X_C_label.grid_forget()
        X_C_entry.grid_forget()
        X_G_label.grid_forget()
        X_G_entry.grid_forget()
        Y_C_label.grid_forget()
        Y_C_entry.grid_forget()
        Y_G_label.grid_forget()
        Y_G_entry.grid_forget()


root.protocol("WM_DELETE_WINDOW", on_close)

# create frame for input fields
input_frame = tk.Frame(root)
input_frame.grid(row=0, column=0, padx=10, pady=10)

# Define labels and entries for parameters
tk.Label(input_frame, text="Min X:").grid(row=0, column=0, sticky='w')
min_x_entry = tk.Entry(input_frame)
min_x_entry.grid(row=0, column=1)
min_x_entry.insert(0, "0")

tk.Label(input_frame, text="Max X:").grid(row=1, column=0, sticky='w')
max_x_entry = tk.Entry(input_frame)
max_x_entry.grid(row=1, column=1)
max_x_entry.insert(0, "500")

tk.Label(input_frame, text="Interval X:").grid(row=2, column=0, sticky='w')
intvl_x_entry = tk.Entry(input_frame)
intvl_x_entry.grid(row=2, column=1)
intvl_x_entry.insert(0, "100")

tk.Label(input_frame, text="Min Y:").grid(row=3, column=0, sticky='w')
min_y_entry = tk.Entry(input_frame)
min_y_entry.grid(row=3, column=1)
min_y_entry.insert(0, "0")

tk.Label(input_frame, text="Max Y:").grid(row=4, column=0, sticky='w')
max_y_entry = tk.Entry(input_frame)
max_y_entry.grid(row=4, column=1)
max_y_entry.insert(0, "500")

tk.Label(input_frame, text="Interval Y:").grid(row=5, column=0, sticky='w')
intvl_y_entry = tk.Entry(input_frame)
intvl_y_entry.grid(row=5, column=1)
intvl_y_entry.insert(0, "100")

tk.Label(input_frame, text="Iteration number:").grid(row=6, column=0, sticky='w')
iteration_number_entry = tk.Entry(input_frame)
iteration_number_entry.grid(row=6, column=1)
iteration_number_entry.insert(0, "3")

tk.Label(input_frame, text="Input data:").grid(row=7, column=0, sticky='w')
offsets_entry = tk.Entry(input_frame)
offsets_entry.grid(row=7, column=1)
offsets_entry.insert(0, "3")

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
result_label = tk.Label(input_frame, text="")
result_label.grid(row=11, column=1)

# Train button
train_button = tk.Button(input_frame, text="Train Model", command=train_model)
train_button.grid(row=12, column=0)

# Predict button
predict_button = tk.Button(input_frame, text="Predict", command=predict_model)
predict_button.grid(row=12, column=1)

plot_frame = tk.Frame(root)
plot_frame.grid(row=0, column=1, padx=10, pady=10)

customHyperParams = tk.IntVar()
checkButton = tk.Checkbutton(root, text="choose svr hyperparams", onvalue=1, offvalue=0, variable=customHyperParams, command=onCheckboxToggle)
checkButton.grid(row=13,column=0)


# Run the application
root.mainloop()

