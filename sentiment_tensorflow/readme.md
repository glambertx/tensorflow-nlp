# Classification of medical text documents

Final project for CS 224D explores the use of deep learning natural language processing for capturing implicit and explicit judgments based on documented symptom text from the Vaccine Adverse Event Reporting System (VAERS). Convolutional neural networks with random initialization and pre-trained word vector embeddings are compared to a baseline logistic regression model for accuracy in classifying seriousness of adverse events in VAERS.

## Getting Started

CNN model (sentiment_tensorflow) was run using Python 2.7.10 on Ubuntu 14.04

### Installing

First TensorFlow needs to be installed (we used the GPU version for this analysis due to large speed increases)

```
https://www.tensorflow.org/versions/r0.9/get_started/os_setup.html
```

Additionally you will need Numpy and SciPy

```
sudo apt-get install python-numpy python-scipy

```

End with an example of getting some data out of the system or using it for a little demo

## Running the deep learning model

```
python train.py
```

## Built With

* GPU - Nvidia GeForce Titan X

## Authors

* **Greg Lambert** 

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Kim, Y. (2014). Convolutional neural networks for sentence classification

  http://emnlp2014.org/papers/pdf/EMNLP2014181.pdf
