import matplotlib.pyplot as plt
import numpy as np

tensor_size = (1, 50, 20, 13)
batch_dim = 0
channel_dim = 1

my_np_tensor = np.random.random(tensor_size)
# print (my_np_tensor)

# def get_coords(tensor, channel_dim = 1):
#     tensor = np.squeeze(tensor)
#     return tensor


def plot_layer(tensor, ax, layer=0, channel_dim=0):
    tensor = np.squeeze(tensor)

    xs = np.array(range(tensor.shape[-1]))
    ys = np.array(range(tensor.shape[-2]))

    zs = tensor[layer,:,:]

    XS, YS = np.meshgrid(xs, ys)
    ZS = np.ones_like(XS)*layer*2
    ZS = ZS + zs

    #the size of the markers

    ax.plot_surface(XS, YS, ZS, cmap='spring')

    return xs, ys


fig = plt.figure()
ax = plt.axes(projection='3d')
# ax.set_aspect('equal')

for layer in range(my_np_tensor.shape[channel_dim]):
    xs, ys = plot_layer(my_np_tensor, ax, layer=layer)

X, Y = xs, ys
Z = np.array(range(my_np_tensor.shape[channel_dim]))

max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() / 2.0

mid_x = (X.max()+X.min()) * 0.5
mid_y = (Y.max()+Y.min()) * 0.5
mid_z = (Z.max()+Z.min()) * 0.5

ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, mid_y + max_range)
ax.set_zlim(mid_z - max_range, mid_z + max_range)

fig.tight_layout()

plt.show()

