from transformers import pipeline
generator = pipeline("sentiment-analysis")
import warnings
warnings.filterwarnings('ignore')

class LLMS:
    def __init__(self, input_sentence: str, input_list: list, clabels: list):
        self.input_txt = input_sentence
        self.input_list = input_list
        self.clabels = clabels

    def sentiment_analyser(self):
        print('Running sentiment analysis...\n')
        generator = pipeline("sentiment-analysis")
        print('------------------------\n')
        print('input text:', self.input_txt)
        print('Sentiment:', generator(self.input_txt)[0]['label'])
        print('Score:', round(generator(self.input_txt)[0]['score'], 2) * 100, '\n')
        print('------------------------\n')

    def zeroshot_classifier(self, 
                            task="zero-shot-classification"):
        print('Running zero-shot-classification...\n')
        classifier = pipeline(task)
        result = classifier(sequences=self.input_list, candidate_labels=self.clabels)
        print(result)
