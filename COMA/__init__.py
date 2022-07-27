import os
import sys
ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path = sys.path if ROOT_PATH in sys.path else [ROOT_PATH] + sys.path