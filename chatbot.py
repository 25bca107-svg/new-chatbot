import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Training data: Questions paired with target intents
KNOWLEDGE_BASE = {
    "greeting": {
        "patterns": ["hello", "hi", "hey", "good morning", "good evening", "what's up"],
        "responses": ["Hello! How can I help you today?", "Hey there! What can I do for you?", "Hi! How's it going?"]
    },
    "help": {
        "patterns": ["i need help", "can you help me", "what can you do", "assistance required"],
        "responses": ["I am a simple NLP chatbot! Ask me basic questions or say goodbye."]
    },
    "name": {
        "patterns": ["what is your name", "who are you", "what should i call you"],
        "responses": ["I am QuickBot, an NLP-powered Python chatbot!"]
    },
    "goodbye": {
        "patterns": ["bye", "see you later", "goodbye", "exit", "quit"],
        "responses": ["Goodbye! Have a great day!", "See you later!", "Bye! Take care."]
    }
}

def train_nlp_model():
    corpus = []
    labels = []
    
    for intent, data in KNOWLEDGE_BASE.items():
        for pattern in data["patterns"]:
            corpus.append(pattern)
            labels.append(intent)
            
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(corpus)
    
    return vectorizer, tfidf_matrix, corpus, labels

def get_response(user_input, vectorizer, tfidf_matrix, labels, threshold=0.2):
    # Vectorize user input
    input_vector = vectorizer.transform([user_input])
    
    # Calculate similarity scores
    similarities = cosine_similarity(input_vector, tfidf_matrix)[0]
    best_match_idx = similarities.argmax()
    best_score = similarities[best_match_idx]
    
    # Return response if confidence score beats threshold
    if best_score >= threshold:
        matched_intent = labels[best_match_idx]
        return random.choice(KNOWLEDGE_BASE[matched_intent]["responses"])
    else:
        return "I'm not quite sure I understand. Could you rephrase that?"

def run_chatbot():
    vectorizer, tfidf_matrix, corpus, labels = train_nlp_model()
    print("🤖 Chatbot Ready! (Type 'quit' or 'bye' to exit)\n" + "-"*45)
    
    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
            
        response = get_response(user_input, vectorizer, tfidf_matrix, labels)
        print(f"Bot: {response}")
        
        if user_input.lower() in ["bye", "goodbye", "quit", "exit"]:
            break

if __name__ == "__main__":
    run_chatbot()
