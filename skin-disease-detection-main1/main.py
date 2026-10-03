"""Command-line entry point for a single DermaAssist prediction."""

from __future__ import annotations

import argparse

from PIL import Image

from predictor import predict


def main() -> None:
	parser = argparse.ArgumentParser(description="Classify a skin-lesion image.")
	parser.add_argument("image", help="Path to an image file")
	args = parser.parse_args()

	probabilities = predict(Image.open(args.image))
	for class_name, probability in sorted(
		probabilities.items(), key=lambda item: item[1], reverse=True
	):
		print(f"{class_name}: {probability:.4f}")


if __name__ == "__main__":
	main()
