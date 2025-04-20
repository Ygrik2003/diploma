import src.examples.navier_stockes as navier_stockes

config = {"Lx": 20}
task = navier_stockes.ProblemTaskData("D:/work/diploma/data/vector_field_data.csv")
print(len(task.domain()))
print(len(task.wall()))
print(len(task.inflow()))
print(len(task.outflow()))

import matplotlib.pyplot as plt


def plot_data_loader_x_components(
    data_loader, group_names=None, title="DataLoader X Components"
):
    all_x = []
    all_y = []

    for batch in data_loader:
        print(batch)
        all_x.append(batch[0][0][0])
        all_y.append(batch[0][0][1])

    plt.figure(figsize=(10, 6))
    plt.scatter(all_x, all_y, alpha=0.6)

    plt.xlabel("X Component Value")
    plt.ylabel("Point Index")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


plot_data_loader_x_components(task.domain(), "domain")
plot_data_loader_x_components(task.wall(), "wall")
plot_data_loader_x_components(task.inflow(), "inflow")
plot_data_loader_x_components(task.outflow(), "outflow")
