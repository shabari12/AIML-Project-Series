import nltk
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle

# Download necessary NLTK data
nltk.download('punkt')

class AdmissionChatbot:
    def __init__(self):
        # Initialize training data for admission-related questions
        self.questions = [
            "what are the admission procedures", 
            "what are the admission requirements", 
            "what are the admission deadlines",
            "how can i apply for admission",
            "what documents are needed for admission"
        ]
        self.responses = [
            "The admission procedures include filling out an online application, submitting required documents, and paying the application fee.",
            "The admission requirements include a completed application form, transcripts, standardized test scores, and recommendation letters.",
            "The admission deadlines are typically in early January for regular admission and November for early admission.",
            "You can apply for admission by visiting our website and filling out the online application form.",
            "The documents needed for admission include your transcripts, test scores, recommendation letters, and a personal statement."
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
        return "Hello! I'm your admission assistant chatbot. How can I help you with your college admission queries today?"

    def farewell(self):
        self.save_context()
        return "Goodbye! If you have more questions, feel free to ask anytime."

    def respond_to_question(self, user_input):
        # Transform user input to the same vector space
        X_user = self.vectorizer.transform([user_input])
        
        # Predict the response
        predicted_response = self.model.predict(X_user)
        return predicted_response[0]

    def ask_questions(self):
        questions = [
            "What is your name?",
            "What program are you interested in?",
            "When do you plan to apply?"
        ]

        user_responses = []
        for question in questions:
            response = input(question + " ")
            user_responses.append(response)
            self.context[question] = response

        return "Thank you for the information! This will help me assist you better."
    

    def recall_context(self):
        if self.context:
            context_summary = ", ".join([f"{key[:-1]} is {value}" for key, value in self.context.items() if len(key.split()) > 1])
            return f"I remember you mentioned that {context_summary}."
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
                    if any(keyword in user_input.lower() for keyword in ["apply", "requirements", "procedures", "deadlines", "documents"]):
                        self.ask_questions()
                except:
                    print(self.error_handling())

if __name__ == "__main__":
    bot = AdmissionChatbot()
    bot.chat()
