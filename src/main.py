"""src/main.py — Entry point for Gesture Control Hub."""
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.app import GestureControlApp, SWIPE_GESTURES, APP_VERSION  # noqa: F401


def main():
    GestureControlApp().run()


if __name__ == '__main__':
    main()
