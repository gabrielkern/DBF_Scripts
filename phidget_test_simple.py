from Phidget22.Phidget import *
from Phidget22.Devices.VoltageRatioInput import *
import time

gain = 1
bias = 0

def onVoltageRatioChange(self, voltageRatio):
	force = (voltageRatio - bias) / gain
	print("Force (lbs): " + str(force))

def main():
	voltageRatioInput1 = VoltageRatioInput()

	voltageRatioInput1.setHubPort(0)
	voltageRatioInput1.setChannel(1)

	voltageRatioInput1.setOnVoltageRatioChangeHandler(onVoltageRatioChange)

	voltageRatioInput1.openWaitForAttachment(5000)

	try:
		input("Press Enter to Stop\n")
	except (Exception, KeyboardInterrupt):
		pass

	voltageRatioInput1.close()

main()
