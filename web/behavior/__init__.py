# -*- encoding: utf-8 -*-

from flask import Blueprint

behavior_guide = Blueprint('behavior', __name__, url_prefix='/training_jobs')
from ..behavior import train_and_stop, test_and_stop
