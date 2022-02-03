# Avatar-Builder-train-crop
Repository to crop facial features and train the cropped images. 
** Please find the images link bellow **
https://drive.google.com/drive/folders/1FdxTUUDt2GRjqskM70901av-qf2VTlEJ?usp=sharing

## Descriptions

- ```align_save_img.py```

Align and save the images where faces were detected. (Change directory files name according to yours. ```database/CHICAGO_MIX```


- ```save_cropped.py```

Create the cropped facial features from detected face in the image. You have to uncomment which facial features you want to save place the image in ```database/CHICAGO_MIX```

- ```create_dirs.ipynb```

Seperate the cropped facial feature images into subdirectories corresponding to their labeled shape. The dataframe  ```database/csv_dir/hairchicago2.csv``` is used for all the facial features except hair. For hairstyle classification please use ```database/csv_dir/df_hairstyle.csv```

- ```create_save_aug.ipynb```

Performs data augmentation to boost number of available images ```database/CHICAGO_MIX```

- ```CNN_model2.ipynb```

Train the model once you cropped the images and did data augmentation
