import utils
from models import LLMS
import yaml
import warnings
warnings.filterwarnings('ignore')

with open("./config.yml", 'r') as file:
    inputs = yaml.safe_load(file)

def run():
    print(inputs["review_list"][0])
    model = LLMS(input_sentence=inputs["input_sentence"], review_list=inputs["review_list"])
    model.sentiment_analyser()
    model.zeroshot_classifier()

if __name__ == "__main__":
    run()