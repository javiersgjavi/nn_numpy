# Neural Network with Numpy 

Hi! This is a personal project where I decided to implement neural networks and an optimizer from scratch, simply for the fun of it ✌️🤓. I wanted to experiment and enjoy the process of creating an implementation entirely from scratch. Here’s a bit more about what I’ve done.

## What's in This Project?
- handmade.py: A handmade neural network using NumPy.
- pytorch.py: A neural network using PyTorch, to compare with the manual implementation.

## Requirements
Before running the scripts, make sure to have these libraries installed:

```
pip install numpy scikit-learn torch
```
## What Did I Do in Each File?
### handmade.py

In this file, I implemented everything from scratch:

- bce_loss: A function to calculate binary cross-entropy loss.
- IrisDataset: Class to load and prepare the famous Iris dataset.
- NN: Class for the neural network, including forward and backward functions.
- Adam: Class that implements the Adam optimizer.

Here I train the network and at the end, show the loss and accuracy.

### pytorch.py
This is much shorter and sweeter thanks to PyTorch:

- IrisDataset: Class to load and prepare the data.
- NN: Neural network using torch.nn tools.

I train the network and also show the loss and accuracy.

## How to Run the Scripts
### Running handmade.py

```
python handmade.py
```

You will see the loss decreasing every 100 epochs and the final accuracy of the model.

### Running pytorch.py

```
python pytorch.py
```

Similar to the previous one, but using PyTorch to make it easier.



That’s all! Thanks for checking out my project. If you have any suggestions or comments, they are more than welcome. 😊