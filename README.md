# CNN_test
We use the TensorFlow library to achieve a type of CNN called AlexNet. To optimize the running time with large training data, we choose a stochastic gradient descent algorithm to calculate the cost function. The training data are the dog and cat pictures provided by Microsoft. See this link https://www.microsoft.com/en-us/download/confirmation.aspx?id=54765.

To see the result, please look at the picture(document in the file with png document name). To save the testing time, I only used thirty-two pictures for training and twelve pictures for validation. Generally, the Alex net is efficient.(Now updated to 400 pictures for training data and 200 pictures for validation)

To use this code, please download alex_net.py. You need to put training data and validation data in a same folder. In the training data and validation data folder, you create folders for different classes, such as dog class and cat class. Then, you put your training pictures in this class. 
                                                        
                                     /mainfolder/validation(or training data)/class(dog and cat)
If you put your dog picture in ( .../data/validation_data/dog), then your directory for validation is ( .../data/validation_data)
