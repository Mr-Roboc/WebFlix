import argparse
import os

from lib.multimodal_search import verify_image_embedding,image_search_command
def main()->None:

    parser = argparse.ArgumentParser(description='Multimodal Search')
    subparser = parser.add_subparsers(dest="command",help="Available commands")

    verify_image = subparser.add_parser("verify_image_embed",help = "Verifies the image embeddings")
    verify_image.add_argument("--image",help="The path of the image file")

    search_image = subparser.add_parser("image_search",help= "Search for images")
    search_image.add_argument("--image", help ="The path of image file")


    args= parser.parse_args()

    image_path = args.image or getattr(args, "sub_image", None)
    
    if not image_path:
            parser.error("--image is required")

    match args.command:
        case "verify_image_embed":
            verify_image_embedding(image_path)

        case "image_search":
              results = image_search_command(image_path)

              for idx,result in enumerate(results,1):
                   print(f"{idx}. {result['title']} (similarity): {result['score']} \n{result['description'][:100]}")














if __name__ == "__main__":
    main()

    
    

    