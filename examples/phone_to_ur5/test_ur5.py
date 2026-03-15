import time
import argparse
import numpy as np
from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface

def main():
    parser = argparse.ArgumentParser(description="Test connection to UR5 via RTDE")
    parser.add_argument("--ip", type=str, default="192.168.0.124", help="IP address of the UR5 robot")
    parser.add_argument("--freq", type=float, default=125.0, help="RTDE frequency (default: 125.0)")
    args = parser.parse_args()

    ip = args.ip
    print(f"--- Testing connection to UR5 at {ip} ---")

    try:
        print(f"1. Attempting to connect to RTDE Receive Interface on {ip}...")
        rtde_receive = RTDEReceiveInterface(ip, frequency=args.freq)
        if rtde_receive.isConnected():
            print("Successfully connected to RTDE Receive Interface.")
            
            # Print some telemetry
            q = rtde_receive.getActualQ()
            tcp = rtde_receive.getActualTCPPose()
            
            print(f"\nCurrent Joint Positions (rad): {q}")
            print(f"Current Joint Positions (deg): {[np.degrees(val) for val in q]}")
            print(f"Current TCP Pose (x, y, z, rx, ry, rz): {tcp}")
            
            # Wait a bit to see if connection holds
            time.sleep(1.0)
        else:
            print("RTDE Receive Interface says it's not connected.")
    except Exception as e:
        print(f"FAILED to connect to RTDE Receive: {e}")
        print("\nPossible reasons:")
        print(" - Incorrect IP address.")
        print(" - Robot is not on the same network.")
        print(" - Port 30004 is blocked or not listening.")
        print(" - Another RTDE receive connection is active and blocking.")
        return

    try:
        print(f"\n2. Attempting to connect to RTDE Control Interface on {ip}...")
        rtde_control = RTDEControlInterface(ip)
        if rtde_control.isConnected():
            print("Successfully connected to RTDE Control Interface.")
            # We won't move anything to be safe, but just verifying connection is enough.
            rtde_control.disconnect()
            print("Disconnected from RTDE Control Interface.")
        else:
            print("RTDE Control Interface says it's not connected.")
    except Exception as e:
        print(f"FAILED to connect to RTDE Control: {e}")
        if "already in use" in str(e).lower():
            print("\nCRITICAL: RTDE input registers are already in use.")
            print("Check Teach Pendant -> Installation -> Fieldbus and disable EtherNet/IP, PROFINET, or MODBUS.")

    try:
        print(f"\n3. Attempting to connect to Robotiq Gripper Socket (63352) on {ip}...")
        import socket
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(2.0)
        s.connect((ip, 63352))
        print("Successfully connected to Robotiq Gripper Socket.")
        s.close()
    except Exception as e:
        print(f"FAILED to connect to Robotiq Gripper Socket: {e}")
        print("Note: This is expected if you don't have a Robotiq gripper or if the URCap is not enabled.")

    if 'rtde_receive' in locals():
        rtde_receive.disconnect()
        print("\nDisconnected from RTDE Receive Interface.")

    print("\n--- Test Finished ---")

if __name__ == "__main__":
    main()
