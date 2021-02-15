import logging
from absl import app, flags
import os
Project_Header = '''|--------------------------------------------------------------------------------------------|
|Diabetic Retinopathy Detection Project - Team05                                             |
|--------------------------------------------------------------------------------------------|
|Team Members :                                                                              |
|1.  Ram Sabarish Obla Amar Bapu     |st169693|  email ID:  st169693@stud.uni-stuttgart.de   |
|2.  Swetha Lakshmana Murthy         |st169481|  email ID:  st169481@stud.uni-stuttgart.de   |
|--------------------------------------------------------------------------------------------|
The dataset used here is HAPT dataset. This contains a total of 30 subjects. There raw data is obtained from two interial
sensors, accelerometer and gyroscope. Each user performs two experiments
Folder structure of the dataset
    ROOT_FOLDER(/home/user/.../HAPT Data Set)
       |-------- RawData
       |            |------ acc_exp01_user01.txt
       |            |           
       |            |------ gyro_exp61_user30.txt                 
       |            |------ labels.txt
       |                       
       |                         
       |
       | -------- Train               
       |             |
       |             | ----- X_train.txt
       |             | ----- y_train.txt
       |             | ----- subject_id_train.txt
       |
       |-------- Test               
       |             |
       |             | ----- X_test.txt
       |             | ----- y_test.txt
       |             | ----- subject_id_test.txt
       |
       |
       |------- activity_labels.txt
       ........
       ........
    ```

'''

def main(argv):
    print('Main Function')

if __name__ == "__main__":
    # Importing the programs according to the sequence as mentioned in the README.md
    print(Project_Header)
    import input_pipeline.data_preprocessing_visualization
    import models.layers
    import evaluation.eval
    import evaluation.metrics
    app.run(main)
