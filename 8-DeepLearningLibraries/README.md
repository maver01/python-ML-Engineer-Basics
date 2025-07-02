# DL

| **Architecture**                           | **Description**                                                                   | **Use Cases**                                  |
| ------------------------------------------ | --------------------------------------------------------------------------------- | ---------------------------------------------- |
| **Fully Connected (MLP)**                  | Dense layers where each neuron is connected to all neurons in the previous layer. | Tabular data, basic classification/regression. |
| **Convolutional Neural Networks (CNNs)**   | Use convolutional layers to learn spatial hierarchies.                            | Image and video recognition.                   |
| **Recurrent Neural Networks (RNNs)**       | Process sequences step by step with hidden state.                                 | Time series, text sequences.                   |
| **Long Short-Term Memory (LSTM)**          | A variant of RNNs designed to better capture long-term dependencies.              | Text, speech, time series forecasting.         |
| **Transformers**                           | Use self-attention to model global dependencies in sequences.                     | NLP, translation, large language models.       |
| **Autoencoders**                           | Learn compressed representations by reconstructing inputs.                        | Dimensionality reduction, anomaly detection.   |
| **GANs (Generative Adversarial Networks)** | Learn to generate synthetic data.                                                 | Image synthesis, data augmentation.            |

## CNN

- Conv layer: scans the input image with filters, increase matrix channels.
- Max pool layer: Reduces channels, keeps the most relevant values.
- Fully connected layers: Flattens the channels into a vector.
- Forward: Apply all these steps together, compute predictions.
- Backwards pass (backpropagation): Compute gradients of the loss with respect to each paraemter, to update the weights.
- Optimizer: use backpropagation to update the weights.
