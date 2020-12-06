#setup
!pip3 -r install requirements
#After loading the data, change the $DATA_PATH in DataProcessing.py to the directory containing the data.

#Train the model with this command: !python3 main.py:
It will run:
        data_preprocessing
        load_data & data_transformer
        model definition
        train-test
