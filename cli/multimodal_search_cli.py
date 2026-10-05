import argparse
import os

from lib.multimodal_search import verify_image_embedding
def main()->None:

    parser = argparse.ArgumentParser(description='Multimodal Search')
    subparser = parser.add_subparsers(dest="command",help="Available commands")

    verify_image = subparser.add_parser("verify_image_embed",help = "Verifies the image embeddings")
    verify_image.add_argument("--image",help="The path of the image file")

    args= parser.parse_args()

    image_path = args.image or getattr(args, "sub_image", None)
    
    if not image_path:
            parser.error("--image is required")

    match args.command:
        case "verify_image_embed":
            verify_image_embedding(image_path)













if __name__ == "__main__":
    main()

    
    

    