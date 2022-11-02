from fileinput import filename
import numpy as np
import h5py
import os


def merge_dataset(file_1, file_2, save_dir=None, new_file_name=None):
    # TODO: Is 1 the generated dataset? -> assumed: yes
    input_file_1 = h5py.File(file_1, 'r')
    input_file_2 = h5py.File(file_2, 'r')

    if save_dir is None:
        # TODO: Make sure to use the location of the generated data!!!
        save_dir = os.path.dirname(file_1)
    
    key_len_1 = []
    for key in input_file_1.keys():
        key_len_1.append(len(input_file_1[key]))
    key_len_1 = np.array(key_len_1)
    
    assert np.all(key_len_1 == key_len_1[0])
    
    key_len_2 = []
    for key in input_file_2.keys():
        key_len_2.append(len(input_file_2[key]))
    key_len_2 = np.array(key_len_2)
    
    assert np.all(key_len_2 == key_len_2[0])
    assert np.all(key_len_1 == key_len_2)

    file1_name = os.path.splitext(os.path.basename(file_1))[0]
    file2_name = os.path.splitext(os.path.basename(file_2))[0]

    if new_file_name is None:
        new_file_name = 'merged_'+ file1_name + '_' + file2_name + '.hdf5'
        
    new_file_path = os.path.join(save_dir, new_file_name)
        
    new_file = h5py.File(new_file_path, 'w')

    shuffle_order = np.arange(key_len_1[0]+key_len_2[0])
    np.random.shuffle(shuffle_order)


    for key in input_file_1.keys():
        data1 = input_file_1[key][:]
        data2 = input_file_2[key][:]
        data = np.concatenate([data1, data2])
        new_file.create_dataset(key, data=data[shuffle_order])

    truth1 = np.zeros(key_len_1[0])
    truth2 = np.ones(key_len_2[0])
    truth = np.concatenate([truth1, truth2])
    new_file.create_dataset('label', data=truth[shuffle_order])

    new_file.close()
    
    return new_file_path
    
def split_dataset(file_path, val_fraction=0.2, test_fraction=0.2, save_dir=None, file_name=None):
    if save_dir is None:
        save_dir = os.path.dirname(file_path)
    if file_name is None:
        file_name = os.path.basename(file_path)
        
    input_file = h5py.File(file_path, 'r')

    key_len = []
    for key in input_file.keys():
        key_len.append(len(input_file[key]))
    key_len = np.array(key_len)

    assert np.all(key_len==key_len[0])

    cut_index_val = int(val_fraction * key_len[0])
    cut_index_test = cut_index_val + int(test_fraction * key_len[0])
    
    train_path = os.path.join(save_dir, "train_"+file_name)
    val_path = os.path.join(save_dir, "validation_"+file_name)
    test_path = os.path.join(save_dir, "test_"+file_name)
    
    train_file = h5py.File(train_path, 'w')
    val_file = h5py.File(val_path, 'w')
    test_file = h5py.File(test_path, 'w')

    for key in input_file.keys():
        val_file.create_dataset(key, data=input_file[key][:cut_index_val])
        test_file.create_dataset(key, data=input_file[key][cut_index_val:cut_index_test])
        train_file.create_dataset(key, data=input_file[key][cut_index_test:])

    train_file.close()
    test_file.close()
    val_file.close()
    
    return train_path, val_path, test_path

def prepare_classifier_datasets(original_dataset, generated_dataset, save_dir):
    
    path_merged = merge_dataset(file_1=generated_dataset, file_2=original_dataset,
                                save_dir=save_dir)
    
    return split_dataset(path_merged, save_dir=save_dir, file_name="data.hdf5")
    

# merges the three cls sets into one set  
def main():
    for particle in ["piplus", "eplus", "gamma"]:
        
        print(f"starting {particle}")

        file_1 = os.path.join("..", "Datasets", particle, "train_" + "cls_" + particle + ".hdf5")
        file_2 = os.path.join("..", "Datasets", particle, "val_" + "cls_" + particle + ".hdf5")
        file_3 = os.path.join("..", "Datasets", particle, "test_" + "cls_" + particle + ".hdf5")
    
        input_file_1 = h5py.File(file_1, 'r')
        input_file_2 = h5py.File(file_2, 'r')
        input_file_3 = h5py.File(file_3, 'r')
        
        print(f"datasets opened")

        new_file = h5py.File(os.path.join("..", "Datasets", particle, "cls_" + particle + ".hdf5"), "w")
        
        for key in input_file_1.keys():
            data1 = input_file_1[key][:]
            data2 = input_file_2[key][:]
            data3 = input_file_3[key][:]
            data = np.concatenate([data1, data2, data3])
            new_file.create_dataset(key, data=data)       
            
        print("new file created")
            
        input_file_1.close()
        input_file_2.close()
        input_file_3.close()
        new_file.close()
        
        
    

if __name__=='__main__':
    main()
