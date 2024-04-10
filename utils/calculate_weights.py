from tqdm import tqdm
import numpy as np

def calculate_weigths_labels(dataset, dataloader, num_classes):
    # Create an instance from the data loader
    z = np.zeros((num_classes,))
    # Initialize tqdm
    # tqdm_batch = tqdm(dataloader)
    tqdm_batch = tqdm(dataloader)
    print('Calculating classes weights')
    for step, (inputs_train, mask0_train, mask1_train, _, _) in enumerate(tqdm_batch):
    # for sample in tqdm_batch:
    #     y = mask0_train
    #     y = y.detach().cpu().numpy()
    #     mask = (y >= 0) & (y < num_classes)
    #     labels = y[mask].astype(np.uint8)
    #     count_l = np.bincount(labels, minlength=num_classes)
    #     z += count_l
        y1 = mask1_train
        y1= y1.detach().cpu().numpy()
        mask1 = (y1 >= 0) & (y1 < num_classes)
        labels1 = y1[mask1].astype(np.uint8)
        count_2 = np.bincount(labels1, minlength=num_classes)
        z += count_2
    tqdm_batch.close()
    total_frequency = np.sum(z)
    class_weights = []
    for frequency in z:
        class_weight = 1 / (np.log(1.02 + (frequency / total_frequency)))
        class_weights.append(class_weight)
    ret = np.array(class_weights)
    classes_weights_path = dataset+'_classes_weights2.npy'
    np.save(classes_weights_path, ret)
    return ret

if __name__ == "__main__":
    from dataset_pcd_race import PCD
    from torch.utils.data import DataLoader
    dataset_train=DataLoader(PCD('H:/2020/race/dataset/change_detection_train/train'),num_workers=1,batch_size=1,shuffle=True)
    result=calculate_weigths_labels('cd',dataset_train,7)
    print(result)


