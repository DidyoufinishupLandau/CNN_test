"""
Created on Mon May  2 02:16:24 2022
University of Manchester
@author: Pan Zhang

Try to implement the Alex Net with tensorflow library

"""

import sys
import tensorflow as tf
import matplotlib.pyplot as plt

TRAIN_DATA_DIRECTORY = (
    "C:/Users/jd\Desktop/data/kagglecatsanddogs_3367a/data/train"  # path
)
VALIDATION_DATA_DIRECTORY = (
    "C:/Users/jd\Desktop/data/kagglecatsanddogs_3367a/data/val"  # path validation
)

IMAGE_SHAPE = (227, 227, 3)  # shape of the picture

CLASS_NUMBER = 1000  #the dimension of the vector after the final fully
                        #connected layers


def data_load(dimension):
    dimension = int(dimension)
    """
    load data(picture)

    Parameters
    ----------
    dimension : int
        The dimension of the matrix(Use dimension*dimension matrix to describe
                                    the picture)

    Returns
    -------
    train_data : get the data from the training sample
    validation_data : get the data from the validation sample

    The neural network will compare this two group of data to get the accuracy at
    the last steps.

    The types of data are hard to tell. They are hyperparameter.
    class_names : string
        The name of the training groups(class, we have label so this is not
                                        clustering)
    """

    train_data = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DATA_DIRECTORY,
        label_mode="categorical",
        seed=123,
        image_size=(dimension, dimension)
    )
    validation_data = tf.keras.utils.image_dataset_from_directory(
        VALIDATION_DATA_DIRECTORY,
        label_mode="categorical",
        seed=123,
        image_size=(dimension, dimension)
    )
    #class names, in this case, dog and cat
    class_names = train_data.class_names
    return train_data, validation_data, class_names


def model_load(image_shape=IMAGE_SHAPE, class_number=CLASS_NUMBER):
    ###########################
    "Neural network"
    "noting that their is no padding in Alex network"
    ###########################
    model = tf.keras.models.Sequential(
        [  # normalization
            tf.keras.layers.experimental.preprocessing.Rescaling(
                1.0 / 255, input_shape=image_shape
            ),
            # 11 * 11 convolution layer. 96 channel. strides = 3
            tf.keras.layers.Conv2D(96, (11, 11), strides=3, activation="relu"),
            # followed by a max pooling with size 3 * 3 stride = 2
            tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=2),
            tf.keras.layers.Conv2D(256, (5, 5), activation="relu"),
            tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=2),
            # three same convolution
            tf.keras.layers.Conv2D(384, (3, 3), activation="relu"),
            tf.keras.layers.Conv2D(384, (3, 3), activation="relu"),
            tf.keras.layers.Conv2D(256, (3, 3), activation="relu"),
            tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=2),
            #here the dimension of the layer is 6*6* 256
            # flatten the 2D network
            tf.keras.layers.Flatten(),
            # The fully connected layer
            #9216 because 256 * 6 * 6 = 9216
            tf.keras.layers.Dense(9216, activation="relu"),
            tf.keras.layers.Dense(4096, activation="relu"),
            tf.keras.layers.Dense(4096, activation="relu"),
            tf.keras.layers.Dense(class_number, activation="softmax")
        ]
    )
    model.summary()
    ##########################
    "classification procedure"
    ##########################
    # calcualte the cost function, use gradient descent to get the final answer
    # here sgd is the stochastic gradient descent, which is an optimized gradient descent
    # algorithm
    # to use sgd, our s=epochs should better larger than 100 even if the sgd here
    # is further optimized by momentum

    # we also can use RMSprop and adam methods.

    model.compile(
        optimizer="sgd", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return model


def train_model(epochs):
    """


    Parameters
    ----------
    epochs : int
        Number of trainning

    Returns
    -------
    train_result : the information of training. Stored in hash

    """
    get_train_data, get_validation_data, get_class_names = data_load(227)
    get_model = model_load(class_number=len(get_class_names))
    train_result = get_model.fit(
        get_train_data, validation_data=get_validation_data, epochs=epochs
    )
    return train_result


def plot(
    data_array,
    plot_name,
    x_label,
    y_label,
    title,
    multimethod=True,
    compare=False,
    mark_x=False,
    xMax=False
):
    """
    The main plot function. It can plot three types of function which should
    be enough in this project.
    1. Plot a single function
    set multimethod = False
    2. Plot a compare plot function
    set multimethod = True
    set compare = True
    3. plot a multiple function

    set multimethod = True
    set compare = False

    Parameters
    ----------
    data array : array
        the y axis values
    plot_name : string "string" for single function
                or array contain string has the form["string","string"]
                The legend of the plot

    x_label : string "string" for single function
                or array contain string has the form["string","string"]

                The description of x axis
    y_label : string "string" for single function
                or array contain string has the form["string","string"]

                The description of y axis
    title : string
    the name of the plot

    multimethod : bool, optional
    switch

    compare : bool, optional
        switch. The default is False.

    mark_x : bool, optional
        mark a x point. The default is False.
    xlim : bool, optional
        set a plot range of x

    Returns
    -------
    None.

    """
    try:
        fig = plt.figure(figsize=(10, 8))
        main_plot = fig.add_subplot(111)
        main_plot.set_title(title, fontsize="17")
        if multimethod:
            if not compare:
                main_plot.set_xlabel(x_label, fontsize="15")
                main_plot.set_ylabel(y_label, fontsize="15")
                for i in range(len(data_array)):
                    main_plot.plot(data_array)
                plt.legend(plot_name)
            if compare:
                main_plot.set_xlabel(x_label, fontsize="15")
                main_plot.set_ylabel(y_label, fontsize="15")
                main_plot.plot(data_array[0])
                main_plot.plot(data_array[1], "k--")
                plt.legend(plot_name)
        if not multimethod:
            main_plot.plot(data_array)
            main_plot.set_xlabel(x_label, fontsize="15")
            main_plot.set_ylabel(y_label, fontsize="15")
            plt.legend(plot_name)
        if isinstance(mark_x, float):
            plt.axvline(x=mark_x, color="k", ls="--")
        if isinstance(xMax, float):
            plt.xlim(0, xMax)
        plt.show()
        return None
    except TypeError:
        print("Encounter Type error for some reason")
        sys.exit()
    except ValueError:
        print("encounter Value Error for some reason")
        sys.exit()
    except AttributeError:
        print("encounter attributeError for some reason")
        sys.exit()


if __name__ == "__main__":

    get_train_result = train_model(epochs=30)

    #read training information
    loss = get_train_result.history["loss"]
    accuracy = get_train_result.history["accuracy"]
    validation_loss = get_train_result.history["val_loss"]
    validation_accuracy = get_train_result.history["val_accuracy"]

    ###############
    "plot"
    ###############
    data_array = [validation_loss, loss]
    PLOT_NAME = ["validation_loss", "loss"]
    plot(
        data_array,
        PLOT_NAME,
        "epochs",
        "Cross Entropy",
        "Training and Validation loss",
        compare=True
    )

    data_array = [validation_accuracy, accuracy]
    PLOT_NAME = ["validation_accuracy", "accuracy"]
    plot(
        data_array,
        PLOT_NAME,
        "epochs",
        "Accuracy",
        "Training and Validation accuracy",
        compare=True
    )
