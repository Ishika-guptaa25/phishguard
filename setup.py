#!/usr/bin/env python3
"""
PHISHING DETECTION SYSTEM - SETUP & FIX SCRIPT
One command to fix all issues and prepare for training
"""

import os
import sys
import subprocess


def check_python():
    """Check Python version"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python {version.major}.{version.minor} - Need 3.8+")
        return False
    print(f"✓ Python {version.major}.{version.minor}")
    return True


def check_packages():
    """Check required packages"""
    packages = ['pandas', 'numpy', 'sklearn', 'streamlit', 'joblib', 'requests', 'plotly']
    missing = []

    for pkg in packages:
        try:
            if pkg == 'sklearn':
                __import__('sklearn')
            else:
                __import__(pkg)
            print(f"✓ {pkg}")
        except ImportError:
            print(f"❌ {pkg}")
            missing.append(pkg)

    if missing:
        print(f"\nInstalling missing packages...")
        cmd = f"pip install {' '.join(missing)}"
        os.system(cmd)
        return True
    return True


def check_dataset():
    """Check dataset exists"""
    if not os.path.exists('dataset_full.csv'):
        print("❌ dataset_full.csv not found!")
        print("Download from: https://www.kaggle.com/datasets/shashwatwork/phishing-website-dataset")
        return False
    print("✓ dataset_full.csv found")
    return True


def convert_dataset():
    """Convert dataset to proper format"""
    print("\nConverting dataset...")
    result = os.system("python convert_kaggle_dataset.py")
    return result == 0


def check_feature_extractor():
    """Check feature extractor works"""
    try:
        from feature_extractor import PhishingFeatureExtractor
        extractor = PhishingFeatureExtractor()
        features, names = extractor.extract_features("https://google.com")
        if len(features) > 20:
            print(f"✓ Feature extractor ({len(features)} features)")
            return True
    except Exception as e:
        print(f"❌ Feature extractor: {e}")
    return False


def main():
    print("\n" + "=" * 70)
    print("PHISHING DETECTION SYSTEM - COMPLETE SETUP")
    print("=" * 70 + "\n")

    print("CHECKING ENVIRONMENT")
    print("-" * 70)

    if not check_python():
        print("\n❌ Python check failed")
        return False

    print("\nChecking packages...")
    if not check_packages():
        print("\n❌ Package check failed")
        return False

    print("\nCHECKING DATASET")
    print("-" * 70)

    if not check_dataset():
        print("\n❌ Dataset not found")
        return False

    if not convert_dataset():
        print("\n❌ Dataset conversion failed")
        return False

    print("\nCHECKING MODEL COMPONENTS")
    print("-" * 70)

    if not check_feature_extractor():
        print("\n❌ Feature extractor check failed")
        return False

    print("\n" + "=" * 70)
    print("✓ ALL CHECKS PASSED!")
    print("=" * 70)
    print("\nREADY FOR TRAINING")
    print("-" * 70)
    print("Next step: python model.py")
    print("\nThis will:")
    print("  • Extract features from all URLs")
    print("  • Train the machine learning model")
    print("  • Save model files to models/")
    print("  • Take 10-15 minutes")
    print()

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)