

![The-active-learning-loop-In-active-machine-learning-data-from-experiments-informs-a](https://user-images.githubusercontent.com/69680607/182359622-76a4c70e-c834-478f-8449-74121c705c01.png)



# Covid-19-detection

Virus
![virus_1](https://user-images.githubusercontent.com/69680607/182359720-dfdf021d-6d36-4b3c-8d9d-8846ab3dd8e6.jpeg)


Normal
![normal_1](https://user-images.githubusercontent.com/69680607/182359731-92966de8-6cc4-4e9b-b04f-6a1ca4be3a0f.jpeg)

## Tensorflow version
in Tensorflow-keras directory run this code:
``` python train.py ```

<br/>

## Pytorch version
in Pytorch directory run this code:
``` python train_torch.py ```

<br/>

## Pytorch Active Learning version
in Active Learning (Pytorch) directory run this code:
``` python train_queries.py ```

<br/>

### Note about the parameters in main running file:<br/>
You can change the hyperparameters in the main running file.<br/>
You can also add more models in the models directory using our importing format<br/>
You can use any datasets and using datasets directory files you can split and preprocess the datasets.<br/>

<br/>

## Results
We achieved 98% validation accuracy using InceptionResnetV2 model and could reach around 0.0002 total validation loss in Active learning and normal classifiers in Pytorch and Tensorflow.
