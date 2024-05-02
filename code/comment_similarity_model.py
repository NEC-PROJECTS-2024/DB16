import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class CommentSimilarityModel:
    def __init__(self, comments_file):
        self.image_path = './static/Images'
        self.df = pd.read_csv(comments_file)
        self.df.columns = ['image_name', 'comment']

        self.df['comment'] = self.df['comment'].str.replace(',', '')

        self.vectorizer = TfidfVectorizer()
        self.all_comments_vector = self.vectorize_comments(self.df['comment'])



    def vectorize_comments(self, comments):
        return self.vectorizer.fit_transform(comments.fillna(''))

    def calculate_cosine_similarity(self, text_vector, image_vector):
        return cosine_similarity(text_vector, image_vector)[0][0]

    def find_most_similar_comment(self, input_comment):
        input_comment_vector = self.vectorizer.transform([input_comment])

        similarities = []
        for i, row in self.df.iterrows():
            similarity = self.calculate_cosine_similarity(input_comment_vector, self.all_comments_vector[i])
            similarities.append((row['comment'], row['image_name'], similarity))

        similarities.sort(key=lambda x: x[2], reverse=True)
        
        if similarities and similarities[0][2] >= 0.3:
            return similarities[0]  # Extract highest similarity
        else:
            return None

    def get_most_similar_comment(self, input_comment):
        result = self.find_most_similar_comment(input_comment)
        if result:
            comment, image_name, similarity = result
            return comment, image_name, similarity
        else:
            return "No image matched", "", 0.0
        

    
   