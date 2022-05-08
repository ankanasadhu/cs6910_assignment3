# Part 1 : Using vanilla sequence to sequence model for transliteration

* The models for RNN, LSTM and GRU were implemented using the keras and tensorflow.
* The hyperparmater tuning was done using wandb sweeps.
* The best model was tested against the test data to calculate the test accuracy.

# Libraries used

* Keras library and tensorflow was used for building the encoder, decoder models for RNN, LSTM and GRU.
* Wandb library was used for finding the best validation accuracy and best hyperparameter configuration for the model.
* Pandas library was used to read the data files.

# How to use
* The following command can be run to execute wandb sweep:
    * wandb sweep sweep.yaml
    * wandb run agent <agent_url>
* This runs the wandb config file and the model starts training with random configuration for the hyperparameters.

# Description of the files

* ### data_prepare.py 
    * The file contains a PrepareData class that preprocesses the data and prepares data matrices that can be fed to the model. It also returns the tokenizer objects for both input and target values.
* ### model_.py
    * The file contains encoder and decoder models for RNN, LSTM and GRU.
    * The function create_model is used to return an encoder decoder model for the chosen cell type.
* ### run_inference.py
    * This file contains a function run_inferencing, that takes in the trained model, retrieves the decoder and for each decoder, the output from the previous decoder is passed instead of the original input.
    * Then the new decoder model obtained is returned.
* ### train.py
    * This file contains the train method that is used to train models for various hyperparemeter configuration and log the loss and accuracy using wandb.
* ### calc_accuracy.py
    * This file coontains the calculate_accuracy function that takes in the model, its predictions and ground truth and calculates the accuracy for the model.
    * The function also writes the right and wrong predictions made by the model in two separate files.
* ### make_predictions.py
    * This file contains predict function that uses the encoder decoder model to make predictions for an input.
    * This function is used inside the calc_accuracy file.

* ### sweep.yaml
    * This file contains all the possbile hyperparamter configurations for building the model.
* ### predictions_vanilla folder
    * The folder contains two output files:
        * correct.txt : contains all the correct predictions made by the current model.
        * wrong.txt : contains all the wroing predictions made by the current model.

# References

* https://keras.io/examples/nlp/lstm_seq2seq/
* https://www.tensorflow.org/addons/tutorials/networks_seq2seq_nmt
* https://towardsdatascience.com/seq2seq-model-in-tensorflow-ec0c557e560f

