from Phidget22.Phidget import *
from Phidget22.Devices.VoltageRatioInput import *
import time

gain = 11516.71642
bias = -2.4412758659763615e-05

def onVoltageRatioChange(self, voltageRatio):
	force = (voltageRatio - bias) * gain
	print(str(force))

def main():
	voltageRatioInput1 = VoltageRatioInput()

	voltageRatioInput1.setChannel(1)

	voltageRatioInput1.setOnVoltageRatioChangeHandler(onVoltageRatioChange)

	voltageRatioInput1.openWaitForAttachment(5000)

	try:
		input("Press Enter to Stop\n")
	except (Exception, KeyboardInterrupt):
		pass

	voltageRatioInput1.close()

main()