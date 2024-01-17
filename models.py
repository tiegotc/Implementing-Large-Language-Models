from transformers import pipeline
generator = pipeline("sentiment-analysis")
import warnings
warnings.filterwarnings('ignore')

class LLMS:
    def __init__(self, input_sentence: str, review_list: list):
        self.input_txt = input_sentence
        self.review_list = review_list

    def sentiment_analyser(self):
        print('Running sentiment analysis...\n')
        generator = pipeline("sentiment-analysis")
        print('-----------------------\n')
        print('input text:', self.input_txt[0])
        print('Sentiment:', generator(self.input_txt)[0]['label'])
        print('Score:', round(generator(self.input_txt)[0]['score'], 2) * 100, '\n')
        print('------------------------\n')

    def zeroshot_classifier(self, 
                            task="zero-shot-classification",
                            clabels=["education", "politics", "business"]):
        print('Running zero-shot-classification...\n')
        classifier = pipeline(task)
        for review in self.review_list:
            result = classifier(sequences=review, candidate_labels=clabels,)
            print(result)