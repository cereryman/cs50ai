# cereryman

# Traffic #

## Table of contents ##
* [Purpose](#purpose)
* [Approach](#approach)
* [Models](#models)
* [Summary of Results](#summary-of-results)
* [Conclusion](#conclusion)
* [References](#references)

## Purpose ##
To develop a neural network using TensorFlow to identify which traffic sign appears in a photograph.

## Approach ##
The process started by using the neural network shown in the lecture (this is henceforth, referred to as the "baseline").

From there, I varied various parameters and varied the number of layers to see the impact on accuracy. 
Then, online, I researched more potential ways of enhancing accuracy.

*Note: Processing speeds were not considered during this process.*

## Models ##
Below shows a list of the deviation from the baseline network that were tried. Results are shown in the "Summary of Results"
1. Experimented with changing the dropout rate to 0.2 → The effects of accuracy were a negligible decrease.
2. Added a second Convolutional layer with 64 filters → This time, the improvement was considerable. 
3. Added a third Convolutional layer with 128 filters → The improvements were minor which led me to try other things.
4. Added a Dense hidden layer with 256 filters (no dropout) → Again, the improvements were minor which lead me to try other things.
5. Added a dropout function after every dense layer to reduce over-fitting → Again, minor improvements...

From here, I found it difficult to  improve the accuracy, this led me to search the web for different approaches which were, perhaps, not showcased in the lecture.

*Note: I also realized that I had forgotten to restore the dropout to 0.5. I promptly did so...*

6. According to reference #1, the best way reduce error is a combination of dropout and max_norm. This was implemented as per reference #2. I also restored the dropout to 0.5. → This led to a small improvement.
7. Tried batch normalization as per reference #3 → This provided another small improvement.

## Summary of Results ##
| #   |              Implementation              | Accuracy (%) |
|-----|:----------------------------------------:|:------------:|
| N/A | Baseline from Lecture                    | 5.76         |
| 1   | Change Drop-out                          | 5.51         |
| 2   | Add 2nd Convolutional Layer              | 96.41        |
| 3   | Add 3rd Convolutional Layer              | 97.89        |
| 4   | Add Dense Hidden Layer                   | 98.15        |
| 5   | Add Dropout after Dense Layers           | 98.28        |
| 6   | Add Max-norm and Restore Drop-out to 0.5 | 98.85        |
| 7   | Add Batch Normalization                  | 99.52        |

## Conclusion ##
The most accurate model was #7, with an accuracy of 99.52%.

## References ##
1. <https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf>
2. <https://www.tensorflow.org/api_docs/python/tf/keras/constraints/MaxNorm>
3. <https://www.machinecurve.com/index.php/2020/01/15/how-to-use-batch-normalization-with-keras/>
