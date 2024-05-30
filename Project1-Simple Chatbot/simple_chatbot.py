import nltk
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle

# Download necessary NLTK data
nltk.download('punkt')

class SimpleMLChatbot:
    def __init__(self):
        # Initialize training data
        self.questions = [
            "how are you", "what is your name", "what do you do",
            "how old are you", "where do you live"
        ]
        self.responses = [
            "I'm just a bot, but I'm doing great! How about you?",
            "I'm called SimpleBot. What's your name?",
            "I chat with people to help them with basic questions.",
            "I was created recently, so I'm quite new!",
            "I live in the cloud, where all the data resides."
        ]
        
        # Vectorize the questions
        self.vectorizer = CountVectorizer()
        X_train = self.vectorizer.fit_transform(self.questions)
        
        # Train the model
        self.model = MultinomialNB()
        self.model.fit(X_train, self.responses)
        
        # Load context from previous session if exists
        try:
            with open('context.pkl', 'rb') as f:
                self.context = pickle.load(f)
        except FileNotFoundError:
            self.context = {}
        
    def save_context(self):
        with open('context.pkl', 'wb') as f:
            pickle.dump(self.context, f)
    
    def greet(self):
        return "Hello! I'm your simple ML chatbot. How can I help you today?"

    def farewell(self):
        self.save_context()
        return "Goodbye! Have a great day!"

    def respond_to_question(self, user_input):
        # Transform user input to the same vector space
        X_user = self.vectorizer.transform([user_input])
        
        # Predict the response
        predicted_response = self.model.predict(X_user)
        return predicted_response[0]

    def ask_questions(self):
        questions = [
            "What is your favorite color?",
            "What is your hobby?",
            "What is your favorite food?"
        ]

        user_responses = []
        for question in questions:
            response = input(question + " ")
            user_responses.append(response)
            self.context[question] = response

        return "Thanks for sharing! I have noted your preferences."

    def recall_context(self):
        if self.context:
            context_summary = " ".join([f"{key.split()[-1]} is {value}" for key, value in self.context.items()])
            return f"I remember you mentioned your favorite {context_summary}."
        else:
            return "I don't have any prior context to recall."

    def error_handling(self):
        return "I'm sorry, I didn't understand that. Can you please rephrase?"

    def chat(self):
        print(self.greet())
        
        while True:
            user_input = input("> ")
            if user_input.lower() in ["bye", "exit", "quit"]:
                print(self.farewell())
                break
            elif user_input.lower() == "recall":
                print(self.recall_context())
            else:
                try:
                    response = self.respond_to_question(user_input)
                    print(response)
                    if "your name" in user_input.lower():
                        name = input("Nice to meet you! What's your name? ")
                        self.context['name'] = name
                        print(f"Nice to meet you, {name}!")
                    elif any(keyword in user_input.lower() for keyword in ["favorite color", "hobby", "favorite food"]):
                        self.ask_questions()
                except:
                    print(self.error_handling())

if __name__ == "__main__":
    bot = SimpleMLChatbot()
    bot.chat()
