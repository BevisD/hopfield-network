import numpy as np
import matplotlib.pyplot as plt

from network import HopfieldNetwork

model = HopfieldNetwork(256)

img_1 = np.load("images/img_1.npy")
img_2 = np.load("images/img_2.npy")
img_3 = np.load("images/img_3.npy")
img_4 = np.load("images/img_4.npy")

imgs = [img_1, img_2, img_3, img_4]

model.train(imgs)

masks = [2 * np.random.binomial(1, 0.7, (16, 16)) - 1 for _ in imgs]
starts = [mask * img for mask, img in zip(masks, imgs)]
outs = [model.predict(start) for start in starts]

fig, axs = plt.subplots(4, 3, figsize=(6, 8))

for ax in axs.flatten():
    ax.set_xticks([])
    ax.set_yticks([])

axs[0, 0].set_title("Original")
axs[0, 1].set_title("Altered")
axs[0, 2].set_title("Recalled")

for row, (img, start, out) in enumerate(zip(imgs, starts, outs)):
    axs[row, 0].imshow(img, cmap="gray")
    axs[row, 1].imshow(start, cmap="gray")
    axs[row, 2].imshow(out, cmap="gray")

plt.tight_layout()
plt.show()
