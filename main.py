import utils
from models import LLMS
import yaml
import argparse
import warnings
warnings.filterwarnings('ignore')
parser = argparse.ArgumentParser()

with open("./config.yml", 'r') as file:
    inputs = yaml.safe_load(file)

parser.add_argument('--input_text', 
                    type=str, 
                    default=inputs['input_sentence'],
                    help='Input text to pass into the sentiment analyser.')
parser.add_argument('--review_list', 
                    type=list, 
                    default=inputs['review_list'],
                    help='A list of reviews that you want to check the sentiment for iteratively.')

def run(args):
        model = LLMS(input_sentence=args.input_text, review_list=args.review_list)
        model.sentiment_analyser()
        model.zeroshot_classifier()

if __name__ == "__main__":
    args = parser.parse_args()
    run(args)
