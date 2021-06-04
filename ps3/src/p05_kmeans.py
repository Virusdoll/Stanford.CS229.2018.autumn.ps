from matplotlib.image import imread
import matplotlib.pyplot as plt
import numpy as np

K = 16 # cluster numbers
MAX_IT = 50 # max iterations
CON_TH = 1 # convergence threshold
img_path = '../data/peppers-large.tiff'

def kmeans(k, x, max_it, con_th):
    m, n = x.shape

    # initial cluster centroids mu
    # choose k x
    np.random.seed(1)
    mu = np.random.randint(255, size=(k, n))
    loss = None
    it = 0

    # repeat until convergence
    while True:
        # update c
        c = np.argmin(np.array([
            np.linalg.norm(x - mu[j], axis=1)**2
            for j in range(k)
        ]), axis=0) # shape (m,)
        
        # update mu
        for j in range(k):
            x_j = x[np.where(c == j)]
            if x_j.shape[0] > 0:
                mu[j] = np.mean(x_j, axis=0)
        
        # if convergence
        prev_loss = loss
        loss = sum(np.linalg.norm(x - mu[c], axis=1)**2) / m
        it += 1
        print(it, loss)
        if prev_loss == None:
            continue
        if np.abs(loss - prev_loss) < con_th or it == max_it:
            break

    # replace x with mu
    x = mu[c]

    return x

def compress(img):
    m, n, d = img.shape
    img_compress = kmeans(
        K, img.reshape((m*n, d)), MAX_IT, CON_TH
    ).reshape((m, n, d))
    return img_compress

if __name__ == '__main__':
    # read image
    # shape (512, 512, 3)
    img = imread(img_path)
    img_compress = compress(img)
    plt.imshow(img_compress)
    plt.show()

