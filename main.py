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
parser.add_argument('--input_list', 
                    type=list, 
                    default=inputs['input_list'],
                    help='A list of academia related sentences that the model will try to classify iteratively.')
parser.add_argument('--clabels', 
                    type=list, 
                    default=inputs['clabels'],
                    help='A list of academia related classes to bucket the input_list items into.')

def run(args):
        model = LLMS(input_sentence=args.input_text, input_list=args.input_list, clabels=args.clabels)
        model.sentiment_analyser()
        model.zeroshot_classifier()

if __name__ == "__main__":
    args = parser.parse_args()
    run(args)
