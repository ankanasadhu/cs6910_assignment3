# Part 2 : Using attention based sequence to sequence model for transliteration

* The best model (GRU in our case) was chosen by looking at the highest validation accuracy on the vanilla sequence to sequence model.
* The hyperparmater tuning was done for GRU using wandb sweeps.
* The best model was tested against the test data to calculate the test accuracy and log the required output files.

# Libraries used

* Keras library and tensorflow was used for building the encoder, decoder models for GRU.
* Wandb library was used for finding the best validation accuracy and best hyperparameter configuration for the model.
* Pandas library was used to read the data files.
* matplotlib and seaborn libraries were used to generate the attention heatmaps.
* numpy library was used for matrix and vector operations.


# How to use
* The current confguration for the train.py file's train function is train(1) where 1 = train_for_one (this can be seen at the end of the train.py file).
* This means that the file will train only the best GRU model and recieve the output.
* If this value is set to 0, then the train function will run for all the hyperparameter configurations in sweep.yaml file. 
* If train_for_one  == 0, the following command should be run to execute wandb sweep:
    * wandb sweep sweep.yaml
    * wandb run agent <agent_url>
* This runs the wandb config file and the model starts training with random configuration for the hyperparameters.

# Description of the files

* ### make_dataset.py 
    * The file contains a PrepareData class that preprocesses the data and prepares data matrices that can be fed to the model. It also returns the tokenizer objects for both input and target values.
* ### encoders.py
    * The file contains an encoder class for GRU that generates the encoder model for GRU.
* ### decoders.py
    * The file contains a decoder class for GRU that generates the decoder model for GRU.
* ### attention.py
    * This file contains the Attention class that implements the Bahdanau Attention.
* ### train.py
    * **train** method: Used to train models for various hyperparemeter configuration and log the loss and accuracy using wandb.
    * **validation** method : Makes predictions using the run_inference function and returns the accuracy for test or validation. It also stores the correct and incorrect predictions in two separate files.
    * **train_step** method : Trains the model for each batch of data by computing the gradients and applying the gradients to each learnable parameter in the model. The method uses teacher forcing to train the model.
    * **masked_loss** method : This method calculates loss using two sequences of same length.
    * **run_inference** method : This method is used during output prediction and calculation of the attention weights. It does not use teacher forcing. 
    * **random_test_words** method: generates 10 random words and passes it to gen_plot method for attention weights claculation. 
    * **gen_plot** method : generates the attention weights using the inference model and passes it to make_plot for saving the attention plots as images.
    * **make_plot** method: creates the predictions_attention folder if it is not present and then stores the attention heatmap as an image for all the randomly chosen words.
    * **cstr**, **get_clr**, **print_color** methods : These methods are used to generate the colors and html text for the attention weights of the chosen words, that is later saved as an html file.
    * **connectivity** method : This method saves the attention visualization for each chosen word. It stores the html text in the connectivity.html file. 


* ### sweep.yaml
    * This file contains all the possbile hyperparamter configurations for building the model.
* ### predictions_attention folder
    * The main folder contain **connectivity.html** file which contains the attention visualization for all the randomly chosen words from the test data. 
    * Inside the main folder, there are two folders named GRU and type_GRU_.
    * The GRU folder contains two output files for the validation data:
        * correct.txt : contains all the correct predictions made by the current model.
        * wrong.txt : contains all the wroing predictions made by the current model.
    * The type_GRU_ contains two output files for the test data and also contains the attention heatmaps for the randomly chosen words.

# References

* https://keras.io/examples/nlp/lstm_seq2seq/
* https://www.tensorflow.org/text/tutorials/nmt_with_attention
* https://distill.pub/2019/memorization-in-rnns/#appendix-autocomplete
* https://towardsdatascience.com/visualising-lstm-activations-in-keras-b50206da96ff
* https://towardsdatascience.com/natural-language-generation-part-2-gpt-2-and-huggingface-f3acb35bc86a
* https://colab.research.google.com/github/sarthakmalik/GPT2.Training.Google.Colaboratory/blob/master/Train_a_GPT_2_Text_Generating_Model_w_GPU.ipynb
* https://www.tensorflow.org/addons/tutorials/networks_seq2seq_nmt
* https://towardsdatascience.com/seq2seq-model-in-tensorflow-ec0c557e560f