from vidgear.gears import NetGear
import cv2

options = {"flag": 0, "copy": False, "track": False}

client = NetGear(
    address="192.168.68.122",
    port="5454",
    protocol="tcp",
    pattern=0,
    receive_mode=True,
    logging=True,
    **options
)

while True:
    frame = client.recv()
    if frame is None:
        break
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.imshow("Output Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cv2.destroyAllWindows()
client.close()
