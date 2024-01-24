# # import pandas as pd
# # import numpy as np
# # import time
# # import matplotlib.pyplot as plt
# # from tree.base import DecisionTree
# # from metrics import *

# # np.random.seed(42)
# # num_average_time = 100  # Number of times to run each experiment to calculate the average values


# # # Function to create fake data (take inspiration from usage.py)
# # # ...
# # # Function to calculate average time (and std) taken by fit() and predict() for different N and P for 4 different cases of DTs
# # # ...
# # # Function to plot the results
# # # ...
# # # Other functions
# # # ...
# # # Run the functions, Learn the DTs and Show the results/plots
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
from tree.base import DecisionTree  # Assuming this module is available
from metrics import *  # Assuming this module is available
import json
import matplotlib.pyplot as plt
from statistics import mean
from statistics import stdev

np.random.seed(42)
num_average_time = 100

def create_fake_data(N, P, inp, out ):
    """Creates fake data with N samples and P features of the specified type."""
    if inp+out == "RIRO":
        X = pd.DataFrame(np.random.randn(N, P))
        y = pd.Series(np.random.randn(N))

    elif inp+out == "RIDO":
        X = pd.DataFrame(np.random.randn(N, P))
        y = pd.Series(np.random.randint(5, size=N), dtype="category")

    elif inp+out == "DIRO":
        X = pd.DataFrame({i: pd.Series(np.random.randint(2, size=N), dtype="category") for i in range(P)})
        y = pd.Series(np.random.randn(N))
    elif inp+out == "DIDO":
        X = pd.DataFrame({i: pd.Series(np.random.randint(2, size=N), dtype="category") for i in range(P)})
        y = pd.Series(np.random.randint(5, size=N), dtype="category")

    else:
        raise ValueError("Invalid data_type: {}".format(inp+out))
    return X, y

#Function to plot Avg time vs Size of Dataset graph keeping features constant 
def plot_results_varying_N(N,M):
    for inp in ["RI", "DI"]:
      for out in ["RO", "DO"]:
        avg_fit_list=[]
        avg_predict_list=[]
        for n in range(1,N+1):
            list_fit=[]
            list_predict=[]
            for k in range(0,100):
                    X,y=create_fake_data(n,M,inp,out)
                    s_time=time.time()
                    tree=DecisionTree(criterion="information_gain")
                    tree.fit(X,y)
                    e_time=time.time()
                    tree.predict(X)
                    p_time=time.time()
                    list_fit.append(e_time-s_time)
                    list_predict.append(p_time-e_time)
            avg_fit=mean(list_fit)
            avg_predict=mean(list_predict)
            avg_fit_list.append(avg_fit)
            avg_predict_list.append(avg_predict)
        dataset_length=list(range(1,N+1))
        z = np.polyfit(dataset_length, avg_fit_list, 1)
        p = np.poly1d(z)
        plt.scatter(dataset_length,avg_fit_list,c=avg_fit_list, cmap = plt.cm.plasma)
        plt.title("Size vs Time Plot of Decision Tree")
        plt.xlabel("Dataset Size")
        plt.ylabel("Time taken to fit")
        plt.plot(dataset_length, avg_fit_list)
        plt.show()

        z = np.polyfit(dataset_length, avg_predict_list, 1)
        p = np.poly1d(z)
        plt.scatter(dataset_length,avg_predict_list,c=avg_predict_list, cmap = plt.cm.plasma)
        plt.title("Size vs Time Plot of Decision Tree")
        plt.xlabel("Dataset Size")
        plt.ylabel("Time taken to Predict")
        plt.plot(dataset_length, avg_predict_list)
        plt.show()  

#Function to plot Standard Deviation of time vs Size of Dataset graph keeping features constant 
def plot_results_dev_varying_N(N,M):
    for inp in ["RI", "DI"]:
      for out in ["RO", "DO"]:
        avg_fit_list=[]
        avg_predict_list=[]
        for n in range(1,N+1):
            list_fit=[]
            list_predict=[]
            for k in range(0,100):
                    X,y=create_fake_data(n,M,inp,out)
                    s_time=time.time()
                    tree=DecisionTree(criterion="information_gain")
                    tree.fit(X,y)
                    e_time=time.time()
                    tree.predict(X)
                    p_time=time.time()
                    list_fit.append(e_time-s_time)
                    list_predict.append(p_time-e_time)
            avg_fit=stdev(list_fit)
            avg_predict=stdev(list_predict)
            avg_fit_list.append(avg_fit)
            avg_predict_list.append(avg_predict)
        dataset_length=list(range(1,N+1))
        z = np.polyfit(dataset_length, avg_fit_list, 1)
        p = np.poly1d(z)
        plt.scatter(dataset_length,avg_fit_list,c=avg_fit_list, cmap = plt.cm.plasma)
        plt.title("Size vs Standard Deviation Time Plot of Decision Tree")
        plt.xlabel("Dataset Size")
        plt.ylabel("Standard Deviation of Time taken to fit")
        plt.plot(dataset_length, avg_fit_list)
        plt.show()

        z = np.polyfit(dataset_length, avg_predict_list, 1)
        p = np.poly1d(z)
        plt.scatter(dataset_length,avg_predict_list,c=avg_predict_list, cmap = plt.cm.plasma)
        plt.title("Size vs Standard Deviation Time Plot of Decision Tree")
        plt.xlabel("Dataset Size")
        plt.ylabel("Standard Deviation of Time taken to Predict")
        plt.plot(dataset_length, avg_predict_list)
        plt.show() 
         
#Function to plot Avg time vs Number of features keeping Size of dataset constant
def plot_results_varying_M(N,M):
    for inp in ["RI", "DI"]:
      for out in ["RO", "DO"]:
        avg_fit_list=[]
        avg_predict_list=[]
        for m in range(1,M+1):
            list_fit=[]
            list_predict=[]
            for k in range(0,100):
                    X,y=create_fake_data(N,m,inp,out)
                    s_time=time.time()
                    tree=DecisionTree(criterion="information_gain")
                    tree.fit(X,y)
                    e_time=time.time()
                    tree.predict(X)
                    p_time=time.time()
                    list_fit.append(e_time-s_time)
                    list_predict.append(p_time-e_time)
            avg_fit=mean(list_fit)
            avg_predict=mean(list_predict)
            avg_fit_list.append(avg_fit)
            avg_predict_list.append(avg_predict)
        dataset_length=list(range(1,M+1))
        z = np.polyfit(dataset_length, avg_fit_list, 1)
        p = np.poly1d(z)
        plt.scatter(dataset_length,avg_fit_list,c=avg_fit_list, cmap = plt.cm.plasma)
        plt.title("Number of Features vs Time Plot of Decision Tree")
        plt.xlabel("Number of Features")
        plt.ylabel("Time taken to fit")
        plt.plot(dataset_length, avg_fit_list)
        plt.show()

        z = np.polyfit(dataset_length, avg_predict_list, 1)
        p = np.poly1d(z)
        plt.scatter(dataset_length,avg_predict_list,c=avg_predict_list, cmap = plt.cm.plasma)
        plt.title("Number of Features vs Time Plot of Decision Tree")
        plt.xlabel("Number of Features")
        plt.ylabel("Time taken to Predict")
        plt.plot(dataset_length, avg_predict_list)
        plt.show()  

#Function to plot Standard Deviation of time vs Number of Features  keeping Size of dataset constant
def plot_results_dev_varying_M(N,M):
    for inp in ["RI", "DI"]:
      for out in ["RO", "DO"]:
        avg_fit_list=[]
        avg_predict_list=[]
        for m in range(1,M+1):
            list_fit=[]
            list_predict=[]
            for k in range(0,100):
                    X,y=create_fake_data(N,m,inp,out)
                    s_time=time.time()
                    tree=DecisionTree(criterion="information_gain")
                    tree.fit(X,y)
                    e_time=time.time()
                    tree.predict(X)
                    p_time=time.time()
                    list_fit.append(e_time-s_time)
                    list_predict.append(p_time-e_time)
            avg_fit=stdev(list_fit)
            avg_predict=stdev(list_predict)
            avg_fit_list.append(avg_fit)
            avg_predict_list.append(avg_predict)
        dataset_length=list(range(1,M+1))
        z = np.polyfit(dataset_length, avg_fit_list, 1)
        p = np.poly1d(z)
        plt.scatter(dataset_length,avg_fit_list,c=avg_fit_list, cmap = plt.cm.plasma)
        plt.title("Number of Features vs Standard Deviation of Time Plot of Decision Tree")
        plt.xlabel("Number of Features")
        plt.ylabel("Standard Deviation of Time taken to fit")
        plt.plot(dataset_length, avg_fit_list)
        plt.show()

        z = np.polyfit(dataset_length, avg_predict_list, 1)
        p = np.poly1d(z)
        plt.scatter(dataset_length,avg_predict_list,c=avg_predict_list, cmap = plt.cm.plasma)
        plt.title("Number of Features vs Standard Deviation of Time Plot of Decision Tree")
        plt.xlabel("Number of Features")
        plt.ylabel("Standard Deviation of Time taken to Predict")
        plt.plot(dataset_length, avg_predict_list)
        plt.show()  

#FOR 3D PLOTS
def run_experiment(X, y, tree_type):
    """Runs the experiment for a given dataset and tree type, measuring time for fitting and predicting."""
    learning_time = []
    predict_time = []

    for _ in (range(num_average_time)):
        start_time = time.time()
        tree = DecisionTree(criterion="information_gain")
        tree.fit(X, y)
        end_time = time.time()
        learning_time.append(end_time - start_time)

        start_time = time.time()
        y_hat = tree.predict(X)
        end_time = time.time()
        predict_time.append(end_time - start_time)

    return float(np.mean(learning_time)), float(np.mean(predict_time))

#PLOT THE 3D GRAPHS
#task will take values "predict" or "learning"
def plot_3D(json_path,tree_type,task):
        with open(json_path, "r") as f:
            data = json.load(f)
            
            times = data[f"{task}_times"] 
        z = []
        for i in times:
            for j in range(len(i)):
                i[j] = i[j] * 1000
            z.extend(i)

        x = []
        y = []
        for j in range(1, 7):
            for i in range(6, 42):
                x.append(j)
        for j in range(1, 7):
            for i in range(6, 42):
                y.append(i)


                
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.plot_trisurf(x, y, z, cmap='viridis', linewidth=0.5)
        plt.title(f"{tree_type}_{task}")
        plt.xlabel("N")
        plt.ylabel("M")
        ax.set_zlabel("Time(in us)")

        plt.show()


def main():
    """Runs the experiments for all four cases of decision trees."""
    for inp in ["RI", "DI"]:
      for out in ["RO", "DO"]:

            tree_type = f"{inp}{out}"  # RIRO, RIDO, DIRO, DIDO

            learning_times = []
            predict_times = []
            for N in tqdm(range(1, 7), desc = "Outer Loop"):
                temp_learning_time, temp_predict_time = [], []
                for P in tqdm(range(6, 42), desc = "Inner Loop"):
                    X, y = create_fake_data(N, P, inp, out)
                    
                    learning_time, predict_time = run_experiment(X, y, tree_type)
                    temp_learning_time.append(learning_time)  # Append individual times
                    temp_predict_time.append(predict_time)
                learning_times.append(temp_learning_time)
                predict_times.append(temp_predict_time)
            with open(f"results_{tree_type}.json", "w") as f:
                json.dump({
                    "learning_times": learning_times,
                    "predict_times": predict_times
                }, f)

    # Save plots
    
    for inp in ["RI", "DI"]:
      for out in ["RO", "DO"]:
            json_path = f"results_{inp}{out}.json"
            tree_type = f"{inp}{out}"
            plot_3D(json_path, tree_type=inp+out, task='predict')
            plot_3D(json_path, tree_type=inp+out, task='learning')


# Run either the experiments or load from JSON
if __name__ == "__main__":
    choice = input("Run experiments (e) or plot from JSON (p)? ")
    if choice.lower() == "e":
        main()
    elif choice.lower() == "p":
            for inp in ["RI", "DI"]:
              for out in ["RO", "DO"]:
                tree_type = f"{inp}{out}"
                json_path = f"results_{inp}{out}.json"
                plot_3D(json_path, tree_type=inp+out, task='predict')
                plot_3D(json_path, tree_type=inp+out, task='learning')
                
    else:
        print("Invalid choice.")