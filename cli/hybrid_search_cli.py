import argparse

def main()->None:
    parser= argparse.ArgumentParser(description= "Hybrid Search")

    

    parser.add_subparsers(dest="command", help = "available commands")

    args = parser.parse_args()

    match args.command:
        case _:
            parser.print_help()




if __name__ == "__main__":
    main()

