import os
import shutil
import pandas as pd


class PreparingDatasets:
    def __init__(self, framework):
        self.framework = framework

    def transfer_directory_items(self, in_dir, out_dir, transfer_list, mode='cp', remove_out_dir=False):
        print(f'starting to copying/moving from {in_dir} to {out_dir}')
        if remove_out_dir or os.path.isdir(out_dir):
            os.system(f'rm -rf {out_dir}; mkdir -p {out_dir}')
        if mode == 'cp':
            for name in transfer_list:
                shutil.copy(os.path.join(in_dir, name), out_dir)
        elif mode == 'mv':
            for name in transfer_list:
                shutil.move(os.path.join(in_dir, name), out_dir)
        else:
            raise ValueError(f'{mode} is not supported, supported modes: mv and cp')
        print(f'finished copying/moving from {in_dir} to {out_dir}')

    def dir_train_test_split(self, in_dir, out_dir, test_size=0.3, result_names=('train', 'val'), mode='cp',
                             remove_out_dir=False):
        from sklearn.model_selection import train_test_split
        list_ = os.listdir(in_dir)
        train_name, val_name = train_test_split(list_, test_size=test_size)
        self.transfer_directory_items(in_dir,
                                      os.path.join(out_dir, result_names[0]),
                                      train_name, mode=mode,
                                      remove_out_dir=remove_out_dir)
        self.transfer_directory_items(in_dir,
                                      os.path.join(out_dir, result_names[1]),
                                      val_name, mode=mode,
                                      remove_out_dir=remove_out_dir)

    def create_directory(self):
        if self.framework == 'pytorch':
            try:
                os.makedirs('./covid-19/train/covid')
                os.makedirs('./covid-19/train/normal')
                os.makedirs('./covid-19/val/covid')
                os.makedirs('./covid-19/val/normal')
                os.makedirs('./covid-19/test/covid')
                os.makedirs('./covid-19/test/normal')
                os.makedirs('./covid_splitting/train')
                os.makedirs('./covid_splitting/val')
            except:
                pass
        else:
            print("Wrong Type of Framework!")

    def preparing_datasets(self, split_size=0.3):
        df = pd.read_csv("./Chest_xray_Corona_Metadata.csv")

        train_data = df[df['Dataset_type'] == 'TRAIN']
        test_data = df[df['Dataset_type'] == 'TEST']

        if self.framework == 'pytorch':
            self.create_directory()
            self.dir_train_test_split('./Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/train/',
                                      './covid_splitting/', test_size=split_size)

            # for train data
            # COVID
            train_pnemonia = './covid-19/train/covid/'
            source_train = "./covid_splitting/train"
            move_train_pnemonia = train_data[train_data['Label_2_Virus_category'] == 'COVID-19'][
                'X_ray_image_name'].values
            for i in move_train_pnemonia:
                try:
                    path = os.path.join(source_train, i)
                    shutil.copy(path, train_pnemonia)
                    print("File ", i, " Successfully copied to ", train_pnemonia)
                except FileNotFoundError:
                    pass

            # Normal
            train_normal = './covid-19/train/normal/'
            move_train_normal = train_data[train_data.Label == 'Normal']['X_ray_image_name'].values
            for i in move_train_normal:
                try:
                    path = os.path.join(source_train, i)
                    shutil.copy(path, train_normal)
                    print("File ", i, " Successfully copied to ", train_normal)
                except FileNotFoundError:
                    pass

            # for val data
            # COVID
            val_pnemonia = './covid-19/val/covid/'
            source_val = "./covid_splitting/val/"
            move_train_pnemonia = train_data[train_data['Label_2_Virus_category'] == 'COVID-19'][
                'X_ray_image_name'].values
            for i in move_train_pnemonia:
                try:
                    val_path = os.path.join(source_val, i)
                    shutil.copy(val_path, val_pnemonia)
                    print("File ", i, " Successfully copied to ", val_pnemonia)
                except FileNotFoundError:
                    pass

            # Normal
            val_normal = './covid-19/val/normal/'
            move_train_normal = train_data[train_data.Label == 'Normal']['X_ray_image_name'].values
            for i in move_train_normal:
                try:
                    val_path = os.path.join(source_val, i)
                    shutil.copy(val_path, val_normal)
                    print("File ", i, " Successfully copied to ", val_normal)
                except FileNotFoundError:
                    pass

            # for test data
            # COVID
            test_pnemonia = './covid-19/test/covid/'
            source_test = "./Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/test/"
            move_test_pnemonia = test_data[test_data['Label'] == 'Pnemonia']['X_ray_image_name'].values

            for i in move_test_pnemonia:
                path2 = os.path.join(source_test, i)
                shutil.copy(path2, test_pnemonia)
                print("File ", i, " Successfully copied to ", test_pnemonia)

            test_normal = './covid-19/test/normal/'
            move_test_normal = test_data[test_data.Label == 'Normal']['X_ray_image_name'].values
            for i in move_test_normal:
                path3 = os.path.join(source_test, i)
                shutil.copy(path3, test_normal)
                print("File ", i, " Successfully copied to ", test_normal)

        else:
            print("Wrong Type of Framework")
