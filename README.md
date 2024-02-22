## Chatbot: Conversation with AI

This repository contains the code for a chatbot built from scratch using Keras and TensorFlow. You can train it on your own intent data and interact with it in a text-based conversation.

**Features:**

* Predicts user intent based on input text.
* Responds with pre-defined messages chosen from relevant intent categories.
* Uses saved model, tokenizer, and label encoder for efficiency.
* User experience enhanced with color-coded conversation flow.

**Getting Started:**

1. **Dependencies:** Make sure you have the required libraries installed (`tensorflow`, `keras`, `sklearn`, `colorama`).
2. **Data:** Prepare your training data as a JSON file in the `intent.json` format (see example in the repository).
3. **Training:** Run the `main.py` script to train the model on your data, saving the results as `chat_model`, `tokenizer.pickle`, and `label_encoder.pickle`.
4. **Testing:** Run the `request.py` script to interact with the trained chatbot. Type "quit" to exit.

**Note:** This is a basic implementation, and further development is possible for more advanced features and functionalities.

**Additional Information:**

* The `main.py` script defines the model architecture, training process, and saving methods.
* The `request.py` script implements the user interaction loop and utilizes saved resources for prediction and response retrieval.
* Feel free to experiment with different intent data and explore options for more dynamic response generation.

**I hope this readme helps you understand and use this chatbot project!**
