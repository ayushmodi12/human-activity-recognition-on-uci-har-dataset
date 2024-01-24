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



# import pandas as pd
# import numpy as np
# import time
# import matplotlib.pyplot as plt
# from tree.base import DecisionTree
# from metrics import *
# from tqdm import tqdm


# np.random.seed(42)
# num_average_time = 100

# # Learn DTs 
# # ...
# # 
# # Function to calculate average time (and std) taken by fit() and predict() for different N and P for 4 different cases of DTs
# # ...
# # Function to plot the results
# # ..
# # Function to create fake data (take inspiration from usage.py)
# # ...
# # ..other functions


# """ Case: RIRO"""

# learning_time = list()
# predict_time = list()

# for Ni in tqdm(range(1,7)):
#     for step in tqdm(range(6,42)):
#         N = Ni
#         P = step
#         X = pd.DataFrame(np.random.randn(N, P))
#         y = pd.Series(np.random.randn(N))

#         predict_time_lst = []
#         for j in range(num_average_time):
#             start_time = time.time()
#             tree = DecisionTree(criterion="information_gain")
#             tree.fit(X, y)
#             end_time = time.time()
                
#             learning_time.append(end_time-start_time)

#             start_time = time.time()
#             y_hat = tree.predict(X)
#             end_time = time.time()
#             predict_time_lst.append(end_time-start_time)
#         predict_time.append(float(np.mean(predict_time_lst)))
# plt.plot(list(learning_time))
# plt.ylabel('RIRO : Fit time', fontsize=16)
# plt.show()

# plt.plot(list(predict_time))
# plt.ylabel('RIRO : Predict time', fontsize=16)
# plt.show()



# """ Case: RIDO"""

# learning_time = list()
# predict_time = list()

# for Ni in tqdm(range(1,7)):
#     for step in tqdm(range(6,42)):
#         N = Ni
#         P = step
#         X = pd.DataFrame(np.random.randn(N, P))
#         y = pd.Series(np.random.randint(P, size = N), dtype="category")
            
#         start_time = time.time()
#         tree = DecisionTree(criterion="information_gain")
#         tree.fit(X, y)
#         end_time = time.time()
            
#         learning_time.append(end_time-start_time)

#         start_time = time.time()
#         y_hat = tree.predict(X)
#         end_time = time.time()
            
#         predict_time.append(end_time-start_time)

# plt.plot(list(learning_time))
# plt.ylabel('RIDO : Fit time', fontsize=16)
# plt.show()

# plt.plot(list(predict_time))
# plt.ylabel('RIDO : Predict time', fontsize=16)
# plt.show()

# """ Case: DIRO"""

# # learning_time = list()
# # predict_time = list()

# # for Ni in range(1,7):
# #     for step in range(6,42):
# #         N = Ni
# #         P = step
# #         X = pd.DataFrame({i:pd.Series(np.random.randint(P, size = N), dtype="category") for i in range(5)})
# #         y = pd.Series(np.random.randn(N))
            
# #         start_time = time.time()
# #         tree = DecisionTree(criterion="information_gain")
# #         tree.fit(X, y)
# #         end_time = time.time()
            
# #         learning_time.append(end_time-start_time)

# #         start_time = time.time()
# #         y_hat = tree.predict(X)
# #         end_time = time.time()
            
# #         predict_time.append(end_time-start_time)


# # plt.plot(list(learning_time))
# # plt.ylabel('DIRO : Fit time', fontsize=16)
# # plt.show()

# # plt.plot(list(predict_time))
# # plt.ylabel('DIRO : Predict time', fontsize=16)
# # plt.show()

# """ Case: DIDO"""

# learning_time = list()
# predict_time = list()

# for Ni in tqdm(range(1,7)):
#     for step in tqdm(range(6,42)):
#         N = Ni
#         P = step
#         X = pd.DataFrame({i:pd.Series(np.random.randint(P, size = N), dtype="category") for i in range(5)})
#         y = pd.Series(np.random.randint(P, size = N) , dtype="category")
            
#         start_time = time.time()
#         tree = DecisionTree(criterion="information_gain")
#         tree.fit(X, y)
#         end_time = time.time()
            
#         learning_time.append(end_time-start_time)

#         start_time = time.time()
#         y_hat = tree.predict(X)
#         end_time = time.time()
            
#         predict_time.append(end_time-start_time)

# plt.plot(list(learning_time))
# plt.ylabel('DIDO : Fit time', fontsize=16)
# plt.show()

# plt.plot(list(predict_time))
# plt.ylabel('DIDO : Predict time', fontsize=16)
# plt.show()

# import pandas as pd
# import numpy as np
# import time
# import matplotlib.pyplot as plt
# from tree.base import DecisionTree
# from metrics import *

# np.random.seed(42)
# num_average_time = 100  # Number of times to run each experiment to calculate the average values


# # Function to create fake data (take inspiration from usage.py)
# # ...
# # Function to calculate average time (and std) taken by fit() and predict() for different N and P for 4 different cases of DTs
# # ...
# # Function to plot the results
# # ...
# # Other functions
# # ...
# # Run the functions, Learn the DTs and Show the results/plots

import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
from tree.base import DecisionTree  # Assuming this module is available
from metrics import *  # Assuming this module is available

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
        # Explain theabove line


    else:
        raise ValueError("Invalid data_type: {}".format(inp+out))
    return X, y

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

def plot_results(learning_time, predict_time, label):
    """Plots the results for learning and prediction time and saves the plots as images."""
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(learning_time)
    plt.xlabel("Number of samples and features (combined)")
    plt.ylabel("Fit time (seconds)")
    plt.title("Fit time for {}".format(label))

    plt.subplot(1, 2, 2)
    plt.plot(predict_time)
    plt.xlabel("Number of samples and features (combined)")
    plt.ylabel("Predict time (seconds)")
    plt.title("Predict time for {}".format(label))

    # Save the plot as a PNG image
    plt.savefig(f"{label}_results.png")

    # Close the plot to avoid displaying it
    plt.close()


def plot_from_json(tree_type):
    with open("results.json", "r") as f:
        data = json.load(f)
        learning_times = data["learning_times"]
        predict_times = data["predict_times"]

    # Create separate line plots
    for i, (learning_time, predict_time) in enumerate(zip(learning_times, predict_times)):
        plot_results(learning_time, predict_time, f"{tree_type}_{i+1}")


def main():
    """Runs the experiments for all four cases of decision trees."""
    for inp in ["RI", "DI"]:
        for out in ["RI", "DI"]:

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
        for out in ["RI", "DI"]:
            tree_type = f"{inp}{out}"
            plot_from_json(f"results_{tree_type}.json")



# Run either the experiments or load from JSON
if __name__ == "__main__":
    choice = input("Run experiments (e) or plot from JSON (p)? ")
    if choice.lower() == "e":
        main()
    elif choice.lower() == "p":
        for inp in ["RI", "DI"]:
            for out in ["RI", "DI"]:
                tree_type = f"{inp}{out}"
                plot_from_json(f"results_{tree_type}.json")
    else:
        print("Invalid choice.")