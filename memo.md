cd /home/grasswonder/midas_hand/midas-piper-control
.venv/bin/python3 -c "
from scripts.piper_api import PiperAPI
api = PiperAPI(payload='half')
api.connect()
api.disconnect()
"