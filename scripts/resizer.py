from skimage import data, color
from skimage.transform import rescale, resize, downscale_local_mean


for fold in range(1, 4):
    folder_name = f"fold{fold}_images"
    folder_path = os.path.join(CAFE_PATH, folder_name)
    # get the list of all fold images and append to train data batches
    train_data_batches.append([resize(imread(os.path.join(folder_path,im)), (32,32),anti_aliasing=True) for im in os.listdir(folder_path)])
    # get the labels
    label_path = os.path.join(CAFE_PATH,f"part_{fold}_label_array_emotion.npy")
    train_data_labels.append(np.load(label_path))


