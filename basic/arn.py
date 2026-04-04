import argparse
parser=argparse.ArgumentParser(
    description="The prohrame print the name of my dog "
)
parser.add_argument('-c','--color',metavar='color',required=True,help='the colour to search for ')

args=parser.parse_args()
print(args.color)