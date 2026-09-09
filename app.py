import argparse

from privacy_mesh.demo import launch


def main():
    parser = argparse.ArgumentParser(prog="privacy-mesh-app")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--quick", action="store_true", help="fast low-epoch runs")
    parser.add_argument("--no-prewarm", action="store_true", help="skip running all experiments on boot")
    args = parser.parse_args()
    launch(host=args.host, port=args.port, prewarm=not args.no_prewarm, quick=args.quick)


if __name__ == "__main__":
    main()
